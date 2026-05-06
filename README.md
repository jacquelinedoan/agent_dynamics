# Orchestrator-Worker Dynamics

One of the most common multi-agent architectures deployed is one composed of a single Orchestrator agent who calls on multiple worker agents to produce a conclusion, given a user prompt. In this pilot experiment, I seek to formally describe the dynamics of this specific multi-agent system, and explore different regimes such as high-variance routing (the orchestrator choosing workers erratically), fixed-pattern attractors (the orchestrator settling into the same few workers in a feedback loop), and especially regime changes between them. This is important as the world moves to a fully agentic customer journey because of its scalability — if this system can demonstrate high input-output sensitivity, small changes to the user prompt ("close" in the embedding space) can lead to vastly different conclusions, which has unwanted consequences in customer and colleague experience. Attempting to describe these dynamics can help predict and minimize response inconsistencies in deployed Orchestrator agent systems.

Recreate the virtual environment:

```python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
