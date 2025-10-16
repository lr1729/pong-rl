# Pong RL Sandbox

Lightweight reproduction of the classic Atari Pong reinforcement-learning project with a modernized layout. The entire training logic now lives in a single module (`src/pong_rl/__init__.py`) exposing two agents:

- **Policy Gradient (REINFORCE)** – faithful copy of the Karpathy baseline.
- **Proximal Policy Optimization (PPO)** – CNN actor-critic with frame stacking.

Both share preprocessing, checkpointing, and CLI utilities so the repository stays minimal.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Register the Atari ROMs once (if not already available)
pip install "autorom[accept-rom-license]"
AutoROM --accept-license
```

All commands run through the single entry point:

```bash
PYTHONPATH=src python -m pong_rl <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `train --algo pg --episodes 1000` | Train the vanilla policy-gradient agent. |
| `train --algo ppo --total-steps 200000` | Train the PPO agent (1 env, frame stack 4). |
| `test` | Run a fast smoke-test (imports, networks, environment reset). |
| `watch --algo pg --checkpoint checkpoints/pg.pth` | Play back a saved checkpoint (PG or PPO). |
| `demo --algo ppo --checkpoint checkpoints/ppo.pth --port 8080` | Launch the web demo (see below). |

Checkpoints are written to `checkpoints/pg.pth` and `checkpoints/ppo.pth` by default and contain enough state to resume training.

## Project Layout

```
src/pong_rl/__init__.py     # Shared helpers, PG + PPO trainers, CLI
docs/history.md             # Summary of previous experiments and findings
requirements.txt            # Runtime dependencies
```

Everything else from earlier iterations (analysis scripts, PDFs, large markdown reports, and binary checkpoints) has been removed to keep the repo focused.

## Notes & Tips

- Training loops default to moderate run lengths (PG: 1k episodes, PPO: 200k steps). Override with `--episodes` or `--total-steps`.
- The PPO loop logs progress roughly every 10k environment steps; PG logs every 10 episodes.
- To resume from a checkpoint, keep the same `--algo` and point to the existing file via `--checkpoint`.
- Smoke tests will fail if the Atari ROMs are missing; install them with AutoROM as shown above.
- To host an interactive web demo, run `scripts/run_demo.sh` (or `PYTHONPATH=src python -m pong_rl demo ...`). By default the server binds to `0.0.0.0:8080`, so visit `http://localhost:8080/` (or the corresponding proxy URL provided by your host) after starting it.

## Training Artifacts

- Policy-gradient checkpoints save to `checkpoints/pg.pth`.
- PPO checkpoints save to `checkpoints/ppo.pth`.
- Each file includes the model weights, optimizer state, running reward, and reward history so you can resume or analyse training.

For more background on the experiments that led to this trimmed-down version—including GPU utilisation studies, scaling observations, and transformer experiments—see `docs/history.md`.
