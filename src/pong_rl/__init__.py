"""
Unified training utilities for Pong reinforcement learning agents.

This module keeps the project lightweight by exposing just a handful of
commands:

    python -m pong_rl train --algo pg
    python -m pong_rl train --algo ppo
    python -m pong_rl test
    python -m pong_rl watch --checkpoint checkpoints/pg.pth --algo pg

Both training loops share helpers for preprocessing, environment setup,
and checkpointing so that the repository only needs this single file for
the RL logic.
"""

from __future__ import annotations

import argparse
import atexit
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ensure ALE environments are registered with Gymnasium once on import.
gym.register_envs(ale_py)


###############################################################################
# Common helpers
###############################################################################

def make_env(render_mode: Optional[str] = None) -> gym.Env:
    """Create the Pong environment with optional rendering."""
    return gym.make("ALE/Pong-v5", render_mode=render_mode)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Preprocess a raw Atari frame to an 80x80 binary image."""
    processed = frame[35:195]  # Crop to play field
    processed = processed[::2, ::2, 0]  # Downsample and keep red channel
    processed = processed.astype(np.float32)
    processed[processed == 144.0] = 0.0
    processed[processed == 109.0] = 0.0
    processed[processed != 0.0] = 1.0
    return processed


def difference_frame(current: np.ndarray, previous: Optional[np.ndarray]) -> np.ndarray:
    """Return the difference between consecutive frames (motion extraction)."""
    if previous is None:
        return np.zeros_like(current, dtype=np.float32)
    return current.astype(np.float32) - previous.astype(np.float32)


def discount_rewards(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """Compute discounted rewards with reward normalization."""
    discounted = np.zeros(len(rewards), dtype=np.float32)
    running_total = 0.0
    for index in reversed(range(len(rewards))):
        if rewards[index] != 0:
            running_total = 0.0
        running_total = running_total * gamma + rewards[index]
        discounted[index] = running_total
    mean = discounted.mean()
    std = discounted.std()
    if std < 1e-6:
        return discounted - mean
    return (discounted - mean) / std


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    next_value: float,
    dones: Sequence[bool],
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    returns = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * lam * mask * gae
        advantages[step] = gae
        returns[step] = gae + values[step]
        next_value = values[step]
    adv_std = advantages.std()
    advantages = (advantages - advantages.mean()) / (adv_std + 1e-6)
    return advantages, returns


def ensure_dir(path: Path) -> None:
    """Create directory for the given path if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# Network definitions
###############################################################################


class PolicyNetwork(nn.Module):
    """Two-layer policy network used for vanilla policy gradients."""

    def __init__(self, hidden_size: int = 200) -> None:
        super().__init__()
        input_size = 80 * 80
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class CNNActorCritic(nn.Module):
    """Actor-Critic network for PPO."""

    def __init__(self, num_actions: int = 2, frame_stack: int = 4, hidden_size: int = 512) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_shared = nn.Linear(64 * 6 * 6, hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob, value, entropy)."""
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy


###############################################################################
# Policy Gradient training
###############################################################################


@dataclass
class PGConfig:
    """Configuration for policy gradient training."""

    episodes: int = 1000
    batch_size: int = 10
    learning_rate: float = 1e-3
    gamma: float = 0.99
    decay_rate: float = 0.99
    checkpoint: Path = Path("checkpoints/pg.pth")
    log_interval: int = 10
    save_interval: int = 100


def train_policy_gradient(config: PGConfig) -> List[float]:
    """Train the vanilla policy gradient agent."""
    device = _device()
    ensure_dir(config.checkpoint)

    env = make_env()
    policy = PolicyNetwork().to(device)
    optimizer = optim.RMSprop(policy.parameters(), lr=config.learning_rate, alpha=config.decay_rate)

    start_episode = 0
    reward_history: List[float] = []
    if config.checkpoint.exists():
        checkpoint = torch.load(config.checkpoint, map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = int(checkpoint.get("episode", 0))
        reward_history = list(checkpoint.get("reward_history", []))
        print(f"Resumed policy gradient training from episode {start_episode}.")

    observation, _ = env.reset()
    prev_processed: Optional[np.ndarray] = None

    batch_states: List[np.ndarray] = []
    batch_actions: List[int] = []
    batch_rewards: List[float] = []

    running_reward: Optional[float] = None
    recent_rewards: Deque[float] = deque(maxlen=100)

    for episode_idx in range(start_episode, config.episodes):
        episode_states: List[np.ndarray] = []
        episode_actions: List[int] = []
        episode_rewards: List[float] = []
        prev_processed = None
        done = False

        while not done:
            processed = preprocess_frame(observation)
            diff_frame = difference_frame(processed, prev_processed)
            prev_processed = processed

            state_tensor = torch.from_numpy(diff_frame.reshape(1, -1)).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                prob = float(policy(state_tensor).item())

            action = 2 if np.random.rand() < prob else 3
            episode_states.append(diff_frame)
            episode_actions.append(0 if action == 2 else 1)

            observation, reward, terminated, truncated, _ = env.step(action)
            episode_rewards.append(float(reward))
            done = terminated or truncated

        observation, _ = env.reset()

        reward_sum = float(sum(episode_rewards))
        reward_history.append(reward_sum)
        recent_rewards.append(reward_sum)
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

        batch_states.extend(episode_states)
        batch_actions.extend(episode_actions)
        batch_rewards.extend(episode_rewards)

        if (episode_idx + 1) % config.log_interval == 0:
            recent_avg = np.mean(recent_rewards) if recent_rewards else 0.0
            print(
                f"[PG] Episode {episode_idx + 1}/{config.episodes} | "
                f"Episode reward: {reward_sum:+.0f} | "
                f"Running avg: {running_reward:.2f} | "
                f"Recent avg (100): {recent_avg:.2f}"
            )

        if (episode_idx + 1) % config.batch_size == 0 or (episode_idx + 1) == config.episodes:
            states_tensor = torch.from_numpy(np.stack(batch_states)).to(device=device, dtype=torch.float32)
            actions_tensor = torch.from_numpy(np.array(batch_actions, dtype=np.int64)).to(device)
            rewards_tensor = torch.from_numpy(discount_rewards(batch_rewards, config.gamma)).to(device)

            probs = policy(states_tensor).squeeze(1)
            probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
            log_probs = torch.where(
                actions_tensor == 0,
                torch.log(probs),
                torch.log1p(-probs),
            )

            loss = -(log_probs * rewards_tensor).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_states.clear()
            batch_actions.clear()
            batch_rewards.clear()

        if (episode_idx + 1) % config.save_interval == 0 or (episode_idx + 1) == config.episodes:
            torch.save(
                {
                    "episode": episode_idx + 1,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "reward_history": reward_history,
                    "running_reward": running_reward,
                },
                config.checkpoint,
            )
            print(f"[PG] Saved checkpoint to {config.checkpoint}")

    env.close()
    return reward_history


###############################################################################
# PPO training
###############################################################################


@dataclass
class PPOConfig:
    """Configuration parameters for PPO training."""

    total_steps: int = 200_000
    rollout_length: int = 2048
    batch_size: int = 256
    epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    frame_stack: int = 4
    checkpoint: Path = Path("checkpoints/ppo.pth")
    log_interval: int = 10_000
    save_interval: int = 50_000


class FrameStack:
    """Utility to maintain a fixed-length stack of frames."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.frames: Deque[np.ndarray] = deque(maxlen=size)

    def reset(self) -> None:
        self.frames.clear()

    def push(self, frame: np.ndarray) -> None:
        if not self.frames:
            for _ in range(self.size):
                self.frames.append(frame)
        else:
            self.frames.append(frame)

    def as_array(self) -> np.ndarray:
        assert self.frames, "Frame stack is empty. Call push() before as_array()."
        return np.stack(self.frames, axis=0).astype(np.float32)


class DemoSession:
    """Maintain environment state for the interactive demo."""

    def __init__(self, algo: str, checkpoint: Optional[Path]) -> None:
        self.requested_algo = algo
        self.algo = algo
        self.device = _device()
        self.env = make_env(render_mode="rgb_array")
        self.policy: Optional[nn.Module] = None
        self.checkpoint_path: Optional[Path] = None
        self.agent_note: str = ""
        self.prev_processed: Optional[np.ndarray] = None
        self.frame_stack = FrameStack(4)
        self.done = False
        self.total_reward = 0.0
        self.steps = 0

        self._load_policy(checkpoint)
        self.reset()

    def _load_policy(self, checkpoint: Optional[Path]) -> None:
        if self.algo == "random":
            return

        if checkpoint is None:
            checkpoint = Path("checkpoints/pg.pth" if self.algo == "pg" else "checkpoints/ppo.pth")

        if not checkpoint.exists():
            self.agent_note = f"Checkpoint '{checkpoint}' not found. Falling back to random actions."
            self.algo = "random"
            return

        data = torch.load(checkpoint, map_location=self.device)

        if self.algo == "pg":
            self.policy = PolicyNetwork().to(self.device)
        else:
            self.policy = CNNActorCritic().to(self.device)

        self.policy.load_state_dict(data["model_state_dict"])
        self.policy.eval()
        self.checkpoint_path = checkpoint

    def reset(self) -> np.ndarray:
        observation, _ = self.env.reset()
        self.last_obs = observation
        self.prev_processed = None
        self.frame_stack.reset()
        self.frame_stack.push(preprocess_frame(observation))
        self.done = False
        self.total_reward = 0.0
        self.steps = 0
        return observation

    def current_frame(self) -> np.ndarray:
        return self.last_obs

    def status_message(self) -> str:
        agent_names = {
            "random": "Random Actions",
            "pg": "Policy Gradient (REINFORCE)",
            "ppo": "PPO + CNN",
        }
        checkpoint_info = f" | Checkpoint: {self.checkpoint_path}" if self.checkpoint_path else ""
        note = f"\n\n> {self.agent_note}" if self.agent_note else ""
        return (
            f"**Agent:** {agent_names.get(self.algo, self.algo)}{checkpoint_info}\n"
            f"**Episode return:** {self.total_reward:+.0f} | **Steps:** {self.steps}{note}"
        )

    def step(self) -> Tuple[np.ndarray, str]:
        reset_note = ""
        if self.done:
            self.reset()
            reset_note = "Previous episode finished. Starting a new game."

        if self.algo == "random" or self.policy is None:
            env_action = int(self.env.action_space.sample())
        elif self.algo == "pg":
            processed = preprocess_frame(self.last_obs)
            diff = difference_frame(processed, self.prev_processed)
            self.prev_processed = processed
            state_tensor = torch.from_numpy(diff.reshape(1, -1)).to(device=self.device, dtype=torch.float32)
            with torch.no_grad():
                prob = float(self.policy(state_tensor).item())
            env_action = 2 if prob > 0.5 else 3
        else:  # PPO
            state_tensor = torch.from_numpy(self.frame_stack.as_array()).unsqueeze(0).to(
                device=self.device, dtype=torch.float32
            )
            with torch.no_grad():
                logits, _ = self.policy(state_tensor)
                action_idx = torch.argmax(F.softmax(logits, dim=-1), dim=-1).item()
            env_action = 2 if action_idx == 0 else 3

        observation, reward, terminated, truncated, _ = self.env.step(env_action)
        self.last_obs = observation
        self.total_reward += reward
        self.steps += 1
        if self.algo == "ppo":
            self.frame_stack.push(preprocess_frame(observation))

        action_meanings = self.env.unwrapped.get_action_meanings()
        action_label = action_meanings[env_action] if env_action < len(action_meanings) else str(env_action)

        done = terminated or truncated
        if done:
            self.done = True

        message = (
            f"Action: {action_label} | Reward: {reward:+.0f} | "
            f"Episode return: {self.total_reward:+.0f} | Done: {done}"
        )
        if reset_note:
            message = f"{reset_note}\n{message}"
        if done:
            message += " — episode finished. Click Step to start a new game."

        return observation, message

    def close(self) -> None:
        self.env.close()


def ppo_update(
    policy: CNNActorCritic,
    optimizer: optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
) -> None:
    """Perform a PPO update over collected rollout data."""
    dataset_size = states.size(0)
    indices = np.arange(dataset_size)
    device = states.device

    for _ in range(config.epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, config.batch_size):
            end = start + config.batch_size
            batch_idx = indices[start:end]

            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            logits, values = policy(batch_states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(-1), batch_returns)
            loss = actor_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


def train_ppo(config: PPOConfig) -> List[float]:
    """Train PPO agent using a single environment with frame stacking."""
    device = _device()
    ensure_dir(config.checkpoint)

    env = make_env()
    policy = CNNActorCritic(frame_stack=config.frame_stack).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)

    start_step = 0
    reward_history: List[float] = []
    if config.checkpoint.exists():
        checkpoint = torch.load(config.checkpoint, map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("global_step", 0))
        reward_history = list(checkpoint.get("reward_history", []))
        print(f"Resumed PPO training from step {start_step}.")

    frame_stack = FrameStack(config.frame_stack)

    observation, _ = env.reset()
    frame_stack.reset()
    frame_stack.push(preprocess_frame(observation))

    episode_return = 0.0
    recent_returns: Deque[float] = deque(maxlen=100)
    global_step = start_step

    while global_step < config.total_steps:
        rollout_states: List[np.ndarray] = []
        rollout_actions: List[int] = []
        rollout_log_probs: List[float] = []
        rollout_rewards: List[float] = []
        rollout_values: List[float] = []
        rollout_dones: List[bool] = []

        steps_collected = 0
        while steps_collected < config.rollout_length and global_step < config.total_steps:
            state = frame_stack.as_array()
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                action, log_prob, value, _ = policy.sample(state_tensor)

            env_action = 2 if int(action.item()) == 0 else 3
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated

            rollout_states.append(state)
            rollout_actions.append(int(action.item()))
            rollout_log_probs.append(float(log_prob.item()))
            rollout_rewards.append(float(reward))
            rollout_values.append(float(value.item()))
            rollout_dones.append(bool(done))

            episode_return += reward
            frame_stack.push(preprocess_frame(next_obs))
            observation = next_obs
            global_step += 1
            steps_collected += 1

            if done:
                reward_history.append(episode_return)
                recent_returns.append(episode_return)
                observation, _ = env.reset()
                frame_stack.reset()
                frame_stack.push(preprocess_frame(observation))
                episode_return = 0.0

                if global_step >= config.total_steps:
                    break

        last_state = frame_stack.as_array()
        last_state_tensor = torch.from_numpy(last_state).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            _, last_value_tensor = policy(last_state_tensor)
        last_value = float(last_value_tensor.item())

        advantages, returns = compute_gae(
            rewards=rollout_rewards,
            values=rollout_values,
            next_value=last_value,
            dones=rollout_dones,
            gamma=config.gamma,
            lam=config.gae_lambda,
        )

        states_tensor = torch.from_numpy(np.array(rollout_states, dtype=np.float32)).to(device=device, dtype=torch.float32)
        actions_tensor = torch.from_numpy(np.array(rollout_actions, dtype=np.int64)).to(device)
        old_log_probs_tensor = torch.from_numpy(np.array(rollout_log_probs, dtype=np.float32)).to(device)
        advantages_tensor = torch.from_numpy(advantages).to(device)
        returns_tensor = torch.from_numpy(returns).to(device)

        ppo_update(
            policy=policy,
            optimizer=optimizer,
            states=states_tensor,
            actions=actions_tensor,
            old_log_probs=old_log_probs_tensor,
            advantages=advantages_tensor,
            returns=returns_tensor,
            config=config,
        )

        if global_step % config.log_interval == 0:
            recent_avg = np.mean(recent_returns) if recent_returns else 0.0
            print(
                f"[PPO] Step {global_step:,}/{config.total_steps:,} | "
                f"Recent avg return (100): {recent_avg:.2f}"
            )

        if global_step % config.save_interval == 0 or global_step >= config.total_steps:
            torch.save(
                {
                    "global_step": global_step,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "reward_history": reward_history,
                },
                config.checkpoint,
            )
            print(f"[PPO] Saved checkpoint to {config.checkpoint}")

    env.close()
    return reward_history


###############################################################################
# Utilities: smoke tests and evaluation
###############################################################################


def run_smoke_tests() -> None:
    """Run lightweight checks to confirm dependencies and helpers."""
    print("Running smoke tests...")
    device = _device()
    print(f"Detected device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    dummy_frame = np.zeros((210, 160, 3), dtype=np.uint8)
    processed = preprocess_frame(dummy_frame)
    assert processed.shape == (80, 80)
    print("Preprocessing check passed.")

    policy = PolicyNetwork()
    dummy_state = torch.zeros(1, 80 * 80, dtype=torch.float32)
    with torch.no_grad():
        prob = policy(dummy_state)
    assert prob.shape == (1, 1)
    print("Policy network forward check passed.")

    ppo_net = CNNActorCritic()
    dummy_stack = torch.zeros(1, 4, 80, 80, dtype=torch.float32)
    with torch.no_grad():
        logits, value = ppo_net(dummy_stack)
    assert logits.shape == (1, 2)
    assert value.shape == (1, 1)
    print("PPO network forward check passed.")

    try:
        env = make_env()
        obs, _ = env.reset()
        assert obs.shape == (210, 160, 3)
        env.close()
        print("Environment reset check passed.")
    except Exception as exc:  # noqa: BLE001 - best-effort diagnostics
        print(f"Environment check failed: {exc}")
        print("Install ROMs with `pip install \"autorom[accept-rom-license]\" && AutoROM --accept-license`.")
        raise

    print("All smoke tests passed.")


def watch_agent(checkpoint: Path, algo: str, episodes: int, render: bool) -> None:
    """Load a checkpointed agent and watch it play Pong."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = _device()
    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)

    if algo == "pg":
        policy = PolicyNetwork().to(device)
        data = torch.load(checkpoint, map_location=device)
        policy.load_state_dict(data["model_state_dict"])
        policy.eval()
        observation, _ = env.reset()
        prev_processed: Optional[np.ndarray] = None
        for episode in range(episodes):
            done = False
            episode_return = 0.0
            prev_processed = None
            while not done:
                processed = preprocess_frame(observation)
                diff = difference_frame(processed, prev_processed)
                prev_processed = processed

                state_tensor = torch.from_numpy(diff.reshape(1, -1)).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    prob = float(policy(state_tensor).item())
                action = 2 if prob > 0.5 else 3

                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                done = terminated or truncated

            print(f"[PG] Episode {episode + 1}: return={episode_return:+.0f}")
            observation, _ = env.reset()

    elif algo == "ppo":
        policy = CNNActorCritic().to(device)
        data = torch.load(checkpoint, map_location=device)
        policy.load_state_dict(data["model_state_dict"])
        policy.eval()

        for episode in range(episodes):
            observation, _ = env.reset()
            frame_stack = FrameStack(size=4)
            frame_stack.reset()
            frame_stack.push(preprocess_frame(observation))
            done = False
            episode_return = 0.0
            while not done:
                state = frame_stack.as_array()
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    logits, _ = policy(state_tensor)
                    action = torch.argmax(F.softmax(logits, dim=-1), dim=-1).item()
                env_action = 2 if action == 0 else 3
                observation, reward, terminated, truncated, _ = env.step(env_action)
                frame_stack.push(preprocess_frame(observation))
                episode_return += reward
                done = terminated or truncated
            print(f"[PPO] Episode {episode + 1}: return={episode_return:+.0f}")
    else:
        raise ValueError("Algorithm must be 'pg' or 'ppo'.")

    env.close()


def launch_demo(algo: str, checkpoint: Optional[Path], host: str, port: int) -> None:
    """Start an interactive demo server using Gradio."""
    try:
        import gradio as gr  # Lazy import to keep light dependencies for other commands.
    except ImportError as exc:  # pragma: no cover - handled at runtime.
        raise RuntimeError("Gradio is required for the demo. Install dependencies via `pip install -r requirements.txt`.") from exc

    session = DemoSession(algo, checkpoint)
    atexit.register(session.close)

    initial_frame = session.current_frame()
    initial_status = session.status_message()

    def reset_handler() -> Tuple[np.ndarray, str]:
        frame = session.reset()
        status = f"{session.status_message()}\n\nEnvironment reset."
        return frame, status

    def step_handler() -> Tuple[np.ndarray, str]:
        frame, event = session.step()
        status = f"{session.status_message()}\n\n{event}"
        return frame, status

    with gr.Blocks(title="Pong RL Demo") as demo:
        gr.Markdown(
            "## Pong RL Interactive Demo\n"
            "Use the buttons below to let the agent play one step at a time or reset the episode."
        )
        image = gr.Image(
            value=initial_frame,
            image_mode="RGB",
            height=initial_frame.shape[0],
            width=initial_frame.shape[1],
            label="Game View",
            show_label=True,
        )
        status = gr.Markdown(initial_status)
        with gr.Row():
            step_button = gr.Button("Step Agent", variant="primary")
            reset_button = gr.Button("Reset Episode")

        step_button.click(fn=step_handler, outputs=[image, status])
        reset_button.click(fn=reset_handler, outputs=[image, status])

    print(
        f"[demo] Launching Pong demo on http://{host}:{port}. "
        "When running on RunPod, open the provided proxy URL for this port."
    )
    demo.queue().launch(server_name=host, server_port=port, share=False, show_api=False)


###############################################################################
# Command-line interface
###############################################################################


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Pong RL training utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train an agent.")
    train_parser.add_argument("--algo", choices=("pg", "ppo"), default="pg")
    train_parser.add_argument("--episodes", type=int, default=1000, help="Episodes for PG training.")
    train_parser.add_argument("--total-steps", type=int, default=200_000, help="Total steps for PPO training.")
    train_parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path.")

    subparsers.add_parser("test", help="Run smoke tests.")

    watch_parser = subparsers.add_parser("watch", help="Watch a trained agent.")
    watch_parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to load.")
    watch_parser.add_argument("--algo", choices=("pg", "ppo"), required=True, help="Algorithm for the checkpoint.")
    watch_parser.add_argument("--episodes", type=int, default=3, help="Episodes to play.")
    watch_parser.add_argument("--no-render", action="store_true", help="Disable human rendering.")

    demo_parser = subparsers.add_parser("demo", help="Launch the web-based interactive demo.")
    demo_parser.add_argument("--algo", choices=("random", "pg", "ppo"), default="random", help="Agent used in demo.")
    demo_parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint for pg/ppo agents.")
    demo_parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind the server.")
    demo_parser.add_argument("--port", type=int, default=8080, help="Port for the demo server.")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    if args.command == "train":
        if args.algo == "pg":
            config = PGConfig(episodes=args.episodes)
            if args.checkpoint is not None:
                config.checkpoint = args.checkpoint
            reward_history = train_policy_gradient(config)
            print(f"Policy gradient training finished. Episodes: {len(reward_history)}")
        else:
            config = PPOConfig(total_steps=args.total_steps)
            if args.checkpoint is not None:
                config.checkpoint = args.checkpoint
            reward_history = train_ppo(config)
            print(f"PPO training finished. Steps: {config.total_steps}, returns recorded: {len(reward_history)}")

    elif args.command == "test":
        run_smoke_tests()

    elif args.command == "watch":
        watch_agent(
            checkpoint=args.checkpoint,
            algo=args.algo,
            episodes=args.episodes,
            render=not args.no_render,
        )
    elif args.command == "demo":
        launch_demo(
            algo=args.algo,
            checkpoint=args.checkpoint,
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
