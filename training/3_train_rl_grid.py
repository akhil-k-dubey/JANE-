import os
import sys

# Ensure repository root is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
from torch.optim import AdamW

from environments.grid_world.physics_engine import GridPhysicsEngine
from environments.grid_world.reward_function import GridRewardFunction
from environments.grid_world.agent_controller import JaneGridAgent

def train_rl_grid(episodes: int = 100, lr: float = 1e-3, device_str: str = "cuda"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Stage 3] Training Jane RL in Grid World on device: {device}")

    physics = GridPhysicsEngine(size=10)
    # Simple obstacles for training
    physics.set_map(
        walls={(3, 3), (3, 4), (3, 5), (6, 5), (6, 6)},
        hazards={(5, 5)}
    )
    reward_fn = GridRewardFunction()
    agent = JaneGridAgent(physics=physics, reward_fn=reward_fn, latent_dim=256)
    agent.pos_encoder.to(device)
    agent.policy_head.to(device)

    optimizer = AdamW(list(agent.pos_encoder.parameters()) + list(agent.policy_head.parameters()), lr=lr)

    success_count = 0
    start = (0, 0)
    goal = (9, 9)

    for ep in range(episodes):
        current_pos = start
        log_probs = []
        rewards = []

        epsilon = max(0.05, 1.0 - ep / 80.0)

        for step in range(35):
            obs = agent.get_observation_tensor(current_pos, goal, device=device)
            latent = torch.relu(agent.pos_encoder(obs))
            logits = agent.policy_head(latent)
            probs = torch.softmax(logits, dim=-1)

            if torch.rand(1).item() < epsilon:
                action = torch.randint(0, 4, (1,)).item()
            else:
                action = probs.argmax(dim=-1).item()

            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor(action, device=device))
            log_probs.append(log_prob)

            next_pos, is_hazard, event = physics.step(current_pos, action)
            reward, done = reward_fn.calculate_reward(next_pos, goal, event, is_hazard)
            rewards.append(reward)
            current_pos = next_pos

            if done:
                break

        if current_pos == goal:
            success_count += 1

        # REINFORCE policy gradient update
        discounted_returns = []
        running_add = 0.0
        for r in reversed(rewards):
            running_add = r + 0.95 * running_add
            discounted_returns.insert(0, running_add)

        returns_t = torch.tensor(discounted_returns, device=device)
        if len(returns_t) > 1 and returns_t.std() > 0:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-7)

        policy_loss = []
        for lp, R in zip(log_probs, returns_t):
            policy_loss.append(-lp * R)

        if policy_loss:
            optimizer.zero_grad()
            total_loss = torch.stack(policy_loss).sum()
            total_loss.backward()
            optimizer.step()

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}/{episodes} - Steps: {len(rewards)} - Total Reward: {sum(rewards):.2f} - Success Rate: {success_count / (ep + 1):.2%}")

    os.makedirs("logs/training_runs", exist_ok=True)
    torch.save({
        "pos_encoder": agent.pos_encoder.state_dict(),
        "policy_head": agent.policy_head.state_dict()
    }, "logs/training_runs/stage3_rl.pt")
    print("[Stage 3] Checkpoint saved to logs/training_runs/stage3_rl.pt")

if __name__ == "__main__":
    train_rl_grid()
