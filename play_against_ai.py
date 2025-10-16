#!/usr/bin/env python3
"""
Interactive Web Demo: Play Pong against the trained AI

Usage:
    python3 play_against_ai.py [model_path]

Example:
    python3 play_against_ai.py pong_parallel_ppo_model.pth

Then open browser to: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import base64
from io import BytesIO
from PIL import Image
import sys
import os

gym.register_envs(ale_py)

app = Flask(__name__)
CORS(app)

# Global game state
game_env = None
game_state = {
    'score_player': 0,
    'score_ai': 0,
    'observation': None,
    'done': False,
    'frame_stack': None,
    'info': {}
}

# AI Model
class CNNActorCritic(nn.Module):
    """CNN-based Actor-Critic for Parallel PPO"""
    def __init__(self, num_actions=2, frame_stack=4, hidden_size=512):
        super().__init__()
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_shared = nn.Linear(64 * 6 * 6, hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        return self.actor(x), self.critic(x)

    def get_action(self, x):
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1)  # Greedy for evaluation
        return action.item()

# Load model
ai_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    global ai_model
    try:
        ai_model = CNNActorCritic(num_actions=2, frame_stack=4).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        ai_model.load_state_dict(checkpoint['model_state_dict'])
        ai_model.eval()
        print(f"✅ Loaded model from {model_path}")
        print(f"   Episodes: {checkpoint.get('episodes', 'unknown')}")
        if 'reward_history' in checkpoint:
            rewards = checkpoint['reward_history'][-100:]
            wins = sum(1 for r in rewards if r > 0)
            print(f"   Recent performance: {wins}/100 wins")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"   Will use random AI instead")
        return False

def preprocess_frame(frame):
    """Preprocess frame to 80x80 grayscale"""
    frame = frame[35:195]
    frame = frame[::2, ::2, 0]
    frame = np.array(frame, dtype=np.float32)
    frame[frame == 144] = 0
    frame[frame == 109] = 0
    frame[frame != 0] = 1
    return frame

def init_game():
    """Initialize a new game"""
    global game_env, game_state

    if game_env is None:
        game_env = gym.make("ALE/Pong-v5")

    obs, info = game_env.reset()

    # Initialize frame stack
    processed = preprocess_frame(obs)
    frame_stack = deque([processed] * 4, maxlen=4)

    game_state = {
        'score_player': 0,
        'score_ai': 0,
        'observation': obs,
        'done': False,
        'frame_stack': frame_stack,
        'total_reward': 0,
        'info': info
    }

    return game_state

def get_ai_action():
    """Get action from AI model"""
    if ai_model is None:
        # Random fallback
        return np.random.choice([2, 3])  # UP or DOWN

    # Use model
    stacked = np.array(list(game_state['frame_stack']), dtype=np.float32)
    state_tensor = torch.FloatTensor(stacked).unsqueeze(0).to(device)

    with torch.no_grad():
        action = ai_model.get_action(state_tensor)

    # Map 0/1 to 2/3 (UP/DOWN)
    return 2 if action == 0 else 3

def frame_to_base64(frame):
    """Convert numpy frame to base64 for web display"""
    # Convert to RGB if grayscale
    if len(frame.shape) == 2:
        frame = np.stack([frame] * 3, axis=-1)

    # Convert to uint8
    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)

    # Create PIL image
    img = Image.fromarray(frame)

    # Scale up for visibility
    img = img.resize((640, 480), Image.NEAREST)

    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Serve the game interface"""
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_game():
    """Start a new game"""
    init_game()
    return jsonify({
        'success': True,
        'frame': frame_to_base64(game_state['observation']),
        'score_player': game_state['score_player'],
        'score_ai': game_state['score_ai']
    })

@app.route('/step', methods=['POST'])
def step_game():
    """Execute one game step"""
    data = request.json
    player_action = data.get('action', 0)  # 0=NOOP, 2=UP, 3=DOWN

    # Get AI action
    ai_action = get_ai_action()

    # In Pong, we control the right paddle, opponent is left
    # The environment controls the opponent, we just control our paddle
    # So we only send our action

    obs, reward, terminated, truncated, info = game_env.step(player_action)
    done = terminated or truncated

    # Update frame stack
    processed = preprocess_frame(obs)
    game_state['frame_stack'].append(processed)
    game_state['observation'] = obs
    game_state['done'] = done
    game_state['total_reward'] += reward

    # Update scores based on reward
    if reward > 0:
        game_state['score_player'] += 1
    elif reward < 0:
        game_state['score_ai'] += 1

    return jsonify({
        'success': True,
        'frame': frame_to_base64(obs),
        'score_player': game_state['score_player'],
        'score_ai': game_state['score_ai'],
        'done': done,
        'reward': reward,
        'ai_action': 'UP' if ai_action == 2 else 'DOWN' if ai_action == 3 else 'NOOP'
    })

@app.route('/status', methods=['GET'])
def status():
    """Get current game status"""
    return jsonify({
        'model_loaded': ai_model is not None,
        'game_active': game_state['observation'] is not None,
        'score_player': game_state['score_player'],
        'score_ai': game_state['score_ai'],
        'done': game_state['done']
    })

if __name__ == '__main__':
    # Load model if provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        load_model(model_path)
    else:
        print("⚠️  No model provided, using random AI")
        print("   Usage: python3 play_against_ai.py <model_path>")

    print("\n" + "=" * 70)
    print("🎮 PONG: HUMAN vs AI")
    print("=" * 70)
    print("\n🌐 Starting web server...")
    print("   Open browser to: http://localhost:5000")
    print("\n   Controls:")
    print("     W or ↑ = Move paddle UP")
    print("     S or ↓ = Move paddle DOWN")
    print("\n   Press Ctrl+C to stop server")
    print("=" * 70 + "\n")

    # Create templates directory
    os.makedirs('templates', exist_ok=True)

    app.run(host='0.0.0.0', port=5000, debug=False)
