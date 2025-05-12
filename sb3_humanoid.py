import torch
import torch.nn as nn
import gymnasium as gym
import wandb
import os
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from wandb.integration.sb3 import WandbCallback

class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(CustomMLP, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, features_dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)

class PPOTrainer:
    def __init__(self, env_id="Humanoid-v4", num_envs=64, total_timesteps=5_000_000, prev_model_path=None):
        self.env_id = env_id
        self.num_envs = num_envs
        self.total_timesteps = total_timesteps
        self.prev_model_path = prev_model_path
        self.vecnorm_path = "ppo_humanoid_vecnormalize.pkl"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        wandb.init(project="mujoco-humanoid-ppo", sync_tensorboard=True)

        self.envs = self.create_envs()
        self.model = self.load_or_create_model()

    def create_envs(self):
        def make_env(rank):
            def _init():
                env = gym.make(self.env_id)
                return env
            return _init

        envs = SubprocVecEnv([make_env(i) for i in range(self.num_envs)])
        envs = VecMonitor(envs)

        if os.path.exists(self.vecnorm_path):
            envs = VecNormalize.load(self.vecnorm_path, envs)
        else:
            envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)

        return envs

    def load_or_create_model(self):
        if self.prev_model_path and os.path.exists(self.prev_model_path):
            return PPO.load(self.prev_model_path, env=self.envs, device=self.device)
        else:
            return PPO(
                policy="MlpPolicy",
                env=self.envs,
                policy_kwargs={"features_extractor_class": CustomMLP, "features_extractor_kwargs": {"features_dim": 256}},
                n_steps=4096,
                batch_size=1024,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                vf_coef=0.5,
                learning_rate=2.5e-4,
                clip_range=0.2,
                verbose=1,
                tensorboard_log="./ppo_humanoid/",
                device=self.device
            )

    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps, callback=WandbCallback())
        self.model.save("ppo_humanoid")
        self.envs.save(self.vecnorm_path)
        self.envs.close()
        wandb.finish()

class PPOEvaluator:
    def __init__(self, model_path, env_id="Humanoid-v4", video_filename="ppo_humanoid.mp4", vecnorm_path=None, num_envs=8, human=False):
        self.env_id = env_id
        self.video_filename = video_filename
        self.num_envs = num_envs
        self.human = human

        self.envs = SubprocVecEnv([self.make_env(i) for i in range(self.num_envs)])  # SubprocVecEnv 적용

        # Check VecNormalize
        if vecnorm_path is not None and os.path.exists(vecnorm_path):
            self.envs = VecNormalize.load(vecnorm_path, self.envs)
        else:
            self.envs = VecNormalize(self.envs, norm_obs=True, norm_reward=False, clip_obs=10.0)

        self.envs.training = False
        self.model = PPO.load(model_path, env=self.envs, device="cuda" if torch.cuda.is_available() else "cpu")

    def make_env(self, rank):
        def _init():
            return gym.make(self.env_id, render_mode="human" if self.human else "rgb_array")
        return _init

    def evaluate(self, num_steps=2000):
        obs = self.envs.reset()
        total_rewards = np.zeros(self.num_envs)
        frames = []

        for _ in range(num_steps):
            if not self.human:
                frame = self.get_combined_frame()
                frames.append(frame)

            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = self.envs.step(action)

            total_rewards += rewards
            for i in range(self.num_envs):
                if dones[i]:
                    obs[i] = self.envs.reset()[i]

        self.envs.close()
        print(f"Average Total Reward: {np.mean(total_rewards)}")
        if not self.human:
            self.save_video(frames)

    def get_combined_frame(self):
        frames = self.envs.get_images()  # load Rendered images of environments

        height, width, _ = frames[0].shape
        grid_frame = np.zeros((height * 2, width * 4, 3), dtype=np.uint8)

        for i in range(self.num_envs):
            row, col = divmod(i, 4)
            grid_frame[row * height:(row + 1) * height, col * width:(col + 1) * width] = frames[i]

        return grid_frame

    def save_video(self, frames):
        imageio.mimsave(self.video_filename, frames, fps=30)
        print(f"Video saved: {self.video_filename}")

if __name__ == "__main__":
    trainer = PPOTrainer(prev_model_path=f"ppo_humanoid.zip")
    trainer.train()

    evaluator = PPOEvaluator(
        model_path=f"ppo_humanoid.zip",
        video_filename=f"ppo_humanoid.mp4",
        vecnorm_path="ppo_humanoid_vecnormalize.pkl",
        num_envs=8,
        human=False,
    )
    evaluator.evaluate()