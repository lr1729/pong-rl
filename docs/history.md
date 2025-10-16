# Project History & Key Findings

This repository started as an expansive sandbox exploring several reinforcement-learning variants for Atari Pong. The earlier version contained long-form reports, profiling utilities, and multiple experimental agents. Here is a condensed record of what was built and what we learned before consolidating the codebase.

## Implemented Agents

- **Policy Gradient (REINFORCE)**  
  - Matches Andrej Karpathy’s 2016 blog implementation.  
  - Network: 6400 → 200 → 1 MLP, frame differencing for motion.  
  - Sample efficiency: ~8k episodes to consistently beat the built-in AI when trained on GPU.  
  - Key limitation: high variance updates; GPU remains mostly idle.

- **PPO + CNN + Frame Stack**  
  - Actor-critic with three convolutional layers and shared FC trunk (≈1.3M parameters).  
  - Frame stacking (4 frames) restores velocity information lost in single-frame differencing.  
  - Uses Generalized Advantage Estimation and clipped objective.  
  - Sample efficiency improves to roughly 1.5k episodes; wall-clock drops to <1 day on A40.

- **Parallel PPO Prototype** *(removed but recorded)*  
  - Used 64 asynchronous environments to drive throughput on a 96-core machine.  
  - Demonstrated CPU-bound behaviour: even large batches only raised GPU utilisation to ~5%.  
  - Highlighted that Atari simulations, not network size, dominate runtime.

- **Decision Transformer (Offline RL)** *(removed but recorded)*  
  - Collected offline trajectories (up to 1000 games) with parallel envs.  
  - Transformer encoder conditioned on `(return-to-go, state, action)` sequences.  
  - With mostly random data the model underperformed PPO, but served as a template for offline RL exploration.

## Analysis Utilities

- **Environment & GPU Profiling**  
  - `analysis_bottleneck.py` timed each component; environment steps consumed ~70% of per-step time.  
  - GPU memory footprints were tiny (<0.02% of an A40) even for larger networks.  
  - Conclusion: algorithmic improvements and environment parallelism matter more than scaling network width.

- **Architecture Comparison**  
  - Benchmarked MLPs, CNNs, and toy transformers for inference speed and parameter counts.  
  - All models remained compute-light; inference rarely exceeded 1 ms.

- **Documentation Library**  
  - Extensive markdown reports covered scaling laws, online vs. offline trade-offs, GPU utilisation, project recommendations, and team workflows.  
  - The key insights from those documents are distilled into this file.

## Lessons Learned

1. **Environment throughput is the bottleneck.** Larger models barely move the needle compared to better algorithms or more parallel environments.
2. **Upgrade algorithmic foundations first.** Switching from vanilla PG to PPO delivered the largest real-world gain.
3. **Transformers need high-quality data.** Offline methods only shine when trajectories are strong; random-play datasets are insufficient.
4. **Pong has an intrinsic ceiling.** Prior analysis showed that random serves, frame skip latency, and physics quirks cap achievable win rates around 90–95%, even for state-of-the-art agents.
5. **Keep the tooling lean.** Many auxiliary scripts added maintenance overhead without long-term value, motivating the current consolidation.

## Current State

- Training code is unified under `src/pong_rl/__init__.py`.  
- Historical checkpoints, analysis scripts, PDFs, and duplicate documentation have been excised.  
- The `PYTHONPATH=src python -m pong_rl …` interface provides training, smoke tests, and playback with minimal dependencies.

This summary replaces the previous sprawling documentation while preserving the decisions and findings that shaped the final streamlined toolkit.
