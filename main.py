#!/usr/bin/env python3
"""
This is the implementation of the experiment plan we agreed on:

    - 3 workers with control-theoretic roles
        R = Researcher    (expansion)
        C = Critic        (contraction)
        S = Synthesizer   (dissipation)
    - Synthesizer placement is a SWEPT factor:
        terminal      -- orchestrator picks {R,C}; S fires only at the end
        peer          -- orchestrator picks {R,C,S} every round
        scheduled-k   -- orchestrator picks {R,C}; S forced every k rounds
    - Routing strategy is a SWEPT factor (orchestrator + 3 baselines):
        orchestrator  -- LLM picks the next worker given the workspace
        random        -- uniform random worker
        fixed-R / fixed-C / fixed-S -- always the same worker
        round_robin   -- cycle through allowed workers
    - Replicates per (task, placement, strategy, T_o) cell.
    - Per replicate, we record the routing sequence, the workspace at every
      round, and the final answer; embeddings are taken at every step.

What this script computes (the pilot metrics):
    - Order-1, 2, 3 block entropy of the routing sequence.
    - Frequencies of pre-registered motifs (RCRC, RCSRCS, SSSS, ...).
    - Synthesizer-call inter-arrival times.
    - Mean pairwise cosine distance between final-output embeddings (a
      proxy for output divergence across replicates).
    - Workspace-embedding divergence per round (a finite-time divergence
      rate proxy -- NOT a true Lyapunov exponent; the system is stochastic).



Setup
-----
    pip install openai sentence-transformers numpy
    export GROQ_API_KEY=...

Quick smoke test (~30-50 LLM calls, finishes in a few minutes)
--------------------------------------------------------------
    python orchestrator_pilot.py --pilot

Larger pilot (~hundreds of calls)
---------------------------------
    python orchestrator_pilot.py \
        --temperatures 0.0 0.5 1.0 \
        --placements peer terminal \
        --strategies orchestrator random \
        --k 5

Output: JSON files under ./pilot_results/
    trajectories_<timestamp>.json   -- every run, every workspace, every embedding
    summary_<timestamp>.json        -- per-cell metrics
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from openai import OpenAI

# Local sentence-transformers gives us a frozen, reproducible embedding model
# without an extra API dependency. Loaded lazily so --no-embed runs without it.
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# ============================================================================
# Configuration
# ============================================================================

# Use a dated model string so a paper run is reproducible. Update as needed.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Small (~80 MB), fast on CPU, fixed weights -- good defaults for a pilot.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

WORKERS = ["R", "C", "S"]  # the symbolic alphabet for our routing sequences

# Pre-registering the motifs we'll track BEFORE looking at any data is
# important; adding motifs after the fact is a form of p-hacking.
PRE_REGISTERED_MOTIFS = [
    "RC",       # researcher-critic alternation, length 2
    "RCRC",     # length-4 RC limit cycle
    "RCSRCS",   # the conjectured "healthy three-phase rhythm"
    "RSRS",     # premature consolidation
    "SS",       # synthesizer doublet
    "SSSS",     # degenerate self-feeding
    "RR",       # researcher monoculture
    "CC",       # critic monoculture
]


DEFAULT_TASKS = [
    (
        "In 'Puss in Boots', the fairytale, is Puss the good guy? Give 3 reasons for your answer."
        
        # "A small business owner is choosing between two payroll software "
        # "vendors. Vendor A: $40/employee/mo, US-only support, integrates "
        # "with QuickBooks. Vendor B: $25/employee/mo, 24/7 chat support, "
        # "requires manual export to QuickBooks. The business has 12 "
        # "employees, two of whom work in Canada. Recommend a vendor and "
        # "give 3 reasons."
    ),
]


# ============================================================================
# Worker prompts (the control-theoretic roles)
# ============================================================================
# Each prompt is short and structurally analogous so differences in workspace
# evolution come from the role, not from prose style.

RESEARCHER_PROMPT = """You are the RESEARCHER.
Role: EXPAND the workspace with new information relevant to the user task.
Add claims, evidence, hypotheses, or sub-questions. Do NOT remove or summarize.
Append your contribution under a heading "Researcher (round {round_num}):"
Keep it under 150 words."""

CRITIC_PROMPT = """You are the CRITIC.
Role: CONTRACT the workspace by challenging weak claims and flagging gaps.
Identify specific items in the current workspace that are unsupported, vague,
contradictory, or off-topic. Do NOT add unrelated new content.
Append under "Critic (round {round_num}):"
Keep it under 150 words."""

SYNTHESIZER_PROMPT = """You are the SYNTHESIZER.
Role: COMPRESS and reorganize the workspace into a cleaner state.
Produce a consolidated draft that keeps the strongest content and drops noise.
Your output REPLACES the prior workspace -- preserve what matters.
Output under "Synthesizer (round {round_num}, consolidated):"
Keep it under 250 words."""

ORCHESTRATOR_PROMPT = """You are the ORCHESTRATOR of a multi-agent system.
Workers:
  R (Researcher)  -- expands the workspace with new information
  C (Critic)      -- challenges weak claims, contracts the workspace
  S (Synthesizer) -- compresses the workspace into a consolidated draft

Read the workspace and the user task. Decide which worker should act next.
{worker_set_constraint}

Respond with EXACTLY ONE CHARACTER: {valid_chars}. No explanation. No punctuation."""

FINAL_ANSWER_PROMPT = """You are the FINAL READOUT.
The workspace below is the result of a multi-agent process.
Produce the final answer to the user task using only what's in the workspace.
Be direct and concise."""


# ============================================================================
# LLM call helper (single swap point for backend changes)
# ============================================================================

def call_llm(client, 
             model, 
             system, 
             user, 
             temperature, 
             max_tokens=400):
    """
    client: an instantiated LLM client
    model: model identifier string 
    system: system prompt string 
    user: user prompt string )
    temperature: sampling temperature (float)
    max_tokens: max tokens to generate (int)
    Returns the generated text (str). Raises on persistent failure.
    """
    last_err = None
    for attempt in range(10):
        try:
            time.sleep(2.5)
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            msg = str(e)
            m = re.search(r'try again in ([0-9.]+)s', msg)
            wait = max(float(m.group(1)) + 1.0 if m else 2 ** attempt, 5.0)
            print(f"  [warn] LLM call failed ({type(e).__name__}); "
                  f"retrying in {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


# ============================================================================
# Routing strategies (orchestrator + baselines)
# ============================================================================

def orchestrator_pick(client, 
                      model, 
                      task, 
                      workspace, 
                      allowed_workers, 
                      temperature):
    """
    client: an instantiated LLM client
    model: model identifier string 
    task: the user task string
    workspace: the current workspace string (may be empty)
    allowed_workers: list of worker tokens (subset of ["R","C","S"]) that the orchestrator is allowed to pick from this round
    temperature: sampling temperature (float)
    Returns the chosen worker token (str), i.e., one of "R", "C", "S". 
    """
    valid_chars = ", ".join(allowed_workers)
    constraint = f"You may only choose among: {valid_chars}."

    system = ORCHESTRATOR_PROMPT.format(
        worker_set_constraint=constraint, valid_chars=valid_chars,
    )

    user = (
        f"USER TASK:\n{task}\n\n"
        f"CURRENT WORKSPACE:\n{workspace or '(empty)'}\n\n"
        f"Next worker?"
    )
    out = call_llm(client, 
                   model, 
                   system, 
                   user, 
                   temperature, 
                   max_tokens=4)
    out = out.strip().upper()

    for ch in out:
        if ch in allowed_workers:
            return ch
    # Fallback: first allowed worker. Logged so it's auditable in the JSON.
    print(f"  [warn] orchestrator returned unparseable '{out}', "
          f"defaulting to {allowed_workers[0]}", file=sys.stderr)
    return allowed_workers[0]


def random_pick(allowed_workers, rng):
    return rng.choice(allowed_workers)


def round_robin_pick(allowed_workers, round_num):
    return allowed_workers[round_num % len(allowed_workers)]


# ============================================================================
# Worker invocation and workspace update rule
# ============================================================================

def run_worker(client, 
               model, 
               worker, 
               task, 
               workspace, 
               round_num, 
               worker_temp):
    """
    client: an instantiated LLM client
    model: model identifier string
    worker: one of "R", "C", "S"
    task: the user task string
    workspace: the current workspace string (may be empty)
    round_num: the current round number (int)
    worker_temp: sampling temperature for the worker's response (float)
    Invoke the chosen worker on the current workspace; return its contribution."""
    template = {"R": RESEARCHER_PROMPT,
                "C": CRITIC_PROMPT,
                "S": SYNTHESIZER_PROMPT}[worker]
    # System Prompt
    system = template.format(round_num=round_num)
    # User Prompt
    user = f"USER TASK:\n{task}\n\nCURRENT WORKSPACE:\n{workspace or '(empty)'}"
    return call_llm(client, 
                    model, 
                    system, 
                    user, 
                    worker_temp, 
                    max_tokens=500)


def update_workspace(workspace, 
                     worker, 
                     contribution):
    """
    The dissipative-element design choice:
    - R, C : APPEND the contribution to the workspace.
    - S    : REPLACE the workspace with the synthesizer's consolidated draft.
    """
    contribution = contribution.strip()
    if worker == "S":
        return contribution
    return (workspace + "\n\n" + contribution).strip() if workspace else contribution


# ============================================================================
# Trajectory: data structure + one full run
# ============================================================================

@dataclass
class Trajectory:
    """Everything we record from a single run. Serializable to JSON."""
    task: str
    routing_strategy: str          # orchestrator | random | fixed-X | round_robin
    placement: str                 # terminal | peer | scheduled-3 | scheduled-5
    orchestrator_temp: float
    worker_temp: float
    n_rounds: int
    seed: int
    sequence: list                 # [str], each in {"R","C","S"}
    workspace_per_round: list      # [str]
    final_answer: str = ""
    workspace_embeddings: list = field(default_factory=list)  # [[float]]
    final_embedding: list = field(default_factory=list)
    timestamp: str = ""


def allowed_workers_for(placement, round_num):
    """
    Decide the set of workers the routing strategy may pick from this round.
    For 'scheduled-k', S is FORCED on rounds where round_num > 0 and
    round_num % k == 0; otherwise the orchestrator picks from {R,C}.
    """
    if placement == "peer":
        return ["R", "C", "S"]
    if placement == "terminal":
        return ["R", "C"]              # orchestrator never picks S in-loop
    if placement.startswith("scheduled-"):
        k = int(placement.split("-")[1])
        if round_num > 0 and round_num % k == 0:
            return ["S"]               # forced synthesizer step
        return ["R", "C"]
    raise ValueError(f"Unknown placement: {placement}")


def run_trajectory(
    client, model, task,
    routing_strategy="orchestrator",
    placement="peer",
    orchestrator_temp=0.7,
    worker_temp=0.7,
    n_rounds=8,
    seed=0,
    embedder=None,
):
    """Run one trajectory end-to-end. Returns a populated Trajectory."""
    rng = random.Random(seed)
    sequence, workspaces = [], []
    workspace = ""

    for r in range(n_rounds):
        allowed = allowed_workers_for(placement, r)

        # Pick the next worker according to the routing strategy.
        if len(allowed) == 1:
            # Forced step (e.g., the scheduled-k synthesizer beat).
            worker = allowed[0]
        elif routing_strategy == "orchestrator":
            worker = orchestrator_pick(
                client, model, task, workspace, allowed, orchestrator_temp,
            )
        elif routing_strategy == "random":
            worker = random_pick(allowed, rng)
        elif routing_strategy.startswith("fixed-"):
            forced = routing_strategy.split("-")[1]
            # If the fixed worker isn't allowed in this placement, fall back
            # to the first allowed -- with a warning, since this means the
            # baseline isn't strictly "fixed".
            if forced not in allowed:
                print(f"  [warn] fixed-{forced} not allowed under "
                      f"placement={placement}; using {allowed[0]}",
                      file=sys.stderr)
                worker = allowed[0]
            else:
                worker = forced
        elif routing_strategy == "round_robin":
            worker = round_robin_pick(allowed, r)
        else:
            raise ValueError(f"Unknown routing strategy: {routing_strategy}")

        # Run the chosen worker; update workspace.
        contribution = run_worker(
            client, model, worker, task, workspace, r, worker_temp,
        )
        workspace = update_workspace(workspace, worker, contribution)
        sequence.append(worker)
        workspaces.append(workspace)

    # Terminal-only placement: tack on a single synthesizer pass before the
    # readout, so 'terminal' still gets the benefit of consolidation -- this
    # makes the comparison to peer/scheduled fair.
    if placement == "terminal":
        contribution = run_worker(
            client, model, "S", task, workspace, n_rounds, worker_temp,
        )
        workspace = update_workspace(workspace, "S", contribution)
        sequence.append("S")
        workspaces.append(workspace)

    # Final readout. Temperature 0 keeps the answer extraction near-deterministic
    # so the output divergence we measure is driven by the workspace, not by
    # the readout step.
    final = call_llm(
        client, model, FINAL_ANSWER_PROMPT,
        f"USER TASK:\n{task}\n\nWORKSPACE:\n{workspace}",
        temperature=0.0, max_tokens=400,
    )

    traj = Trajectory(
        task=task,
        routing_strategy=routing_strategy,
        placement=placement,
        orchestrator_temp=orchestrator_temp,
        worker_temp=worker_temp,
        n_rounds=n_rounds,
        seed=seed,
        sequence=sequence,
        workspace_per_round=workspaces,
        final_answer=final,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    # Compute embeddings if a model was loaded.
    if embedder is not None:
        wb = embedder.encode(workspaces, normalize_embeddings=True)
        traj.workspace_embeddings = [v.tolist() for v in wb]
        fb = embedder.encode([final], normalize_embeddings=True)
        traj.final_embedding = fb[0].tolist()

    return traj


# ============================================================================
# Metrics on collections of trajectories
# ============================================================================

def shannon_entropy(symbols):
    """Order-1 entropy of a symbol list, in bits."""
    if not symbols:
        return 0.0
    counts = Counter(symbols)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def block_entropy(seq_str, block_size):
    """Entropy over (overlapping) blocks of length `block_size`, in bits per block."""
    if len(seq_str) < block_size:
        return 0.0
    blocks = [seq_str[i:i + block_size]
              for i in range(len(seq_str) - block_size + 1)]
    return shannon_entropy(blocks)


def motif_frequencies(seq_str, motifs=PRE_REGISTERED_MOTIFS):
    """Count overlapping occurrences of each pre-registered motif in seq_str."""
    out = {}
    for m in motifs:
        c, i = 0, 0
        while True:
            idx = seq_str.find(m, i)
            if idx == -1:
                break
            c += 1
            i = idx + 1   # overlapping count
        out[m] = c
    return out


def synthesizer_call_timings(sequence):
    """Inter-S intervals. e.g. RCSRCS -> [3] (positions 2, 5)."""
    positions = [i for i, w in enumerate(sequence) if w == "S"]
    if len(positions) < 2:
        return []
    return [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]


def cosine(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def pairwise_final_divergence(trajectories):
    """Mean pairwise cosine *distance* (1 - sim) across replicates' final outputs."""
    embs = [t.final_embedding for t in trajectories if t.final_embedding]
    if len(embs) < 2:
        return float("nan")
    dists = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            dists.append(1.0 - cosine(embs[i], embs[j]))
    return float(np.mean(dists))


def trajectory_divergence_over_rounds(trajectories):
    """
    For each round r, the mean pairwise cosine distance between workspace
    embeddings at round r across replicates. Increasing values vs r are a
    finite-time divergence-rate proxy. NOT a true Lyapunov exponent --
    this system is stochastic, not a deterministic flow.
    """
    if not trajectories:
        return []
    n = min(len(t.workspace_embeddings) for t in trajectories) \
        if all(t.workspace_embeddings for t in trajectories) else 0
    out = []
    for r in range(n):
        embs = [t.workspace_embeddings[r] for t in trajectories]
        dists = [1.0 - cosine(embs[i], embs[j])
                 for i in range(len(embs)) for j in range(i + 1, len(embs))]
        out.append(float(np.mean(dists)) if dists else float("nan"))
    return out


def summarize(trajectories):
    """Compute the pilot-metrics dict for one (task, placement, strategy, T) cell."""
    if not trajectories:
        return {"n_replicates": 0}
    seqs = ["".join(t.sequence) for t in trajectories]
    pooled = "".join(seqs)

    # Block entropies pooled across replicates.
    H1 = shannon_entropy(list(pooled))
    H2 = block_entropy(pooled, 2)
    H3 = block_entropy(pooled, 3)

    # Motif counts averaged per replicate.
    motif_means = {m: 0.0 for m in PRE_REGISTERED_MOTIFS}
    for s in seqs:
        for k, v in motif_frequencies(s).items():
            motif_means[k] += v / len(seqs)

    # Pooled inter-S timings.
    timings = []
    for t in trajectories:
        timings.extend(synthesizer_call_timings(t.sequence))

    return {
        "n_replicates": len(trajectories),
        "H1_bits": H1,
        "H2_bits": H2,
        "H3_bits": H3,
        "motif_means": motif_means,
        "s_inter_call_intervals": timings,
        "s_inter_call_mean": float(np.mean(timings)) if timings else None,
        "s_inter_call_std": float(np.std(timings)) if timings else None,
        "final_output_divergence": pairwise_final_divergence(trajectories),
        "workspace_divergence_per_round":
            trajectory_divergence_over_rounds(trajectories),
    }


# ============================================================================
# Experiment driver
# ============================================================================

def deterministic_seed(*parts):
    """A reproducible seed from arbitrary args. Python's hash() is salted; this isn't."""
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def run_experiment(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    embedder = None
    if not args.no_embed:
        if SentenceTransformer is None:
            print("[warn] sentence-transformers not installed; running with "
                  "--no-embed semantics", file=sys.stderr)
        else:
            print(f"Loading embedder: {args.embed_model}")
            embedder = SentenceTransformer(args.embed_model)

    all_trajectories = []
    summary = {}

    # Sweep over (task, placement, strategy, orchestrator_temp).
    for task_idx, task in enumerate(args.tasks):
        for placement in args.placements:
            for strategy in args.strategies:
                # T_o is only meaningful when the orchestrator is the chooser.
                temps = args.temperatures if strategy == "orchestrator" else [0.0]
                for temp in temps:
                    cell_key = f"task{task_idx}|{placement}|{strategy}|T{temp}"
                    cell_trajs = []
                    print(f"\n=== {cell_key} ===")

                    def run_rep(k):
                        seed = deterministic_seed(task_idx, placement, strategy, temp, k)
                        print(f"  rep {k+1}/{args.k} seed={seed} ...",
                              end=" ", flush=True)
                        try:
                            traj = run_trajectory(
                                client, args.model, task,
                                routing_strategy=strategy,
                                placement=placement,
                                orchestrator_temp=temp,
                                worker_temp=args.worker_temp,
                                n_rounds=args.n_rounds,
                                seed=seed,
                                embedder=embedder,
                            )
                            print(f"seq={''.join(traj.sequence)}")
                            return traj
                        except Exception as e:
                            print(f"FAILED: {e}")
                            return None

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        results = ex.map(run_rep, range(args.k))
                    cell_trajs = [t for t in results if t is not None]
                    all_trajectories.extend(cell_trajs)
                    summary[cell_key] = summarize(cell_trajs)

    # Persist trajectories and summary side by side.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    traj_path = out_dir / f"trajectories_{stamp}.json"
    summary_path = out_dir / f"summary_{stamp}.json"

    with traj_path.open("w") as f:
        json.dump([asdict(t) for t in all_trajectories], f, indent=2)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {len(all_trajectories)} trajectories to {traj_path}")
    print(f"Wrote summary to {summary_path}")

    # A small inline report so you can eyeball things before opening the JSON.
    print("\n--- Pilot summary ---")
    for cell, m in summary.items():
        print(f"\n{cell}")
        if m.get("n_replicates", 0) == 0:
            print("  (no successful replicates)")
            continue
        print(f"  H1={m['H1_bits']:.3f}  H2={m['H2_bits']:.3f}  H3={m['H3_bits']:.3f}")
        print(f"  motif means : {m['motif_means']}")
        print(f"  S intervals : mean={m['s_inter_call_mean']} "
              f"std={m['s_inter_call_std']}  raw={m['s_inter_call_intervals']}")
        print(f"  final-output divergence : {m['final_output_divergence']}")
        wpr = m["workspace_divergence_per_round"]
        if wpr:
            print(f"  workspace-divergence-by-round : {[round(x,3) for x in wpr]}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="LLM identifier passed to call_llm.")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--no-embed", action="store_true",
                   help="Skip embeddings (for fast smoke tests).")
    p.add_argument("--out-dir", default="./pilot_results")

    p.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.7],
                   help="Orchestrator temperatures to sweep.")
    p.add_argument("--placements", nargs="+", default=["peer", "terminal"],
                   choices=["peer", "terminal", "scheduled-3", "scheduled-5"])
    p.add_argument("--strategies", nargs="+", default=["orchestrator", "random"],
                   choices=["orchestrator", "random",
                            "fixed-R", "fixed-C", "fixed-S", "round_robin"])
    p.add_argument("--n-rounds", type=int, default=8)
    p.add_argument("--worker-temp", type=float, default=0.7,
                   help="Held FIXED across the sweep so T_o is the only varied stochasticity.")
    p.add_argument("--k", type=int, default=5,
                   help="Replicates per cell. Pilot=2-5; full run aim for 30-50.")

    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)

    p.add_argument("--pilot", action="store_true",
                   help="Tiny smoke test: 1 placement, 2 strategies, k=2, 4 rounds.")

    args = p.parse_args()
    if args.pilot:
        args.placements = ["peer"]
        args.strategies = ["orchestrator", "random"]
        args.temperatures = [0.7]
        args.n_rounds = 4
        args.k = 2
    return args


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: set GROQ_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)
    run_experiment(parse_args())


# python main.py --temperatures 0.0 0.5 1.0 --placements peer --strategies orchestrator random --k 10