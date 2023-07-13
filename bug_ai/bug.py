import os
import bug_env
from bug_env import BugEnv
from tqdm.auto import tqdm

from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO

models_dir = "models/PPO-env-v4"
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir) 

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = BugEnv()

######## Training

# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

# START_TIME = 0
# TIMESTEPS = 10000

# max_reward = 0

# for i in range(1, 100000000000000000000):
#     obs, _ = env.reset()
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO-env-v4")
#     model.save(f"models/PPO-env-v4/{START_TIME+TIMESTEPS*i}")

#     obs, _ = env.reset()
#     while True:
#         random_action, _ = model.predict(obs)
#         obs, reward, done, _, _ = env.step(random_action)
#         if reward > max_reward:
#             bug_env.save_maze(env, "maze.txt")
#             max_reward = reward
#             print(reward)
#         if done:
#             break

######### Training with loading

START_TIME = 7860000
TIMESTEPS = 10000

model_path = f"models/PPO-env-v4/{START_TIME}.zip"
model = PPO.load(model_path, env=env)

max_reward = 0

for i in range(1, 100000000000000000000):
    obs, _ = env.reset()
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO-env-v4")
    model.save(f"models/PPO-env-v4/{START_TIME+TIMESTEPS*i}")

    obs, _ = env.reset()
    sum_reward = 0
    while True:
        random_action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(random_action)
        sum_reward += reward
        if done:
            break
    if sum_reward > max_reward:
        bug_env.save_maze(env, "maze.txt")
        max_reward = sum_reward
        print(sum_reward)

######## Testing

# model_path = f"models/PPO-v4/860000.zip"
# model = PPO.load(model_path, env=env)

# # episodes = 30
# max_reward = 0

# for episode in range(1, 10000000):
#     done = False
#     obs, _ = env.reset()
#     while not done:
#         random_action, _ = model.predict(obs)
#         obs, reward, done, _, _ = env.step(random_action)
#         if reward > max_reward:
#             bug_env.save_maze(env, "maze.txt")
#             max_reward = reward
#             print(reward)