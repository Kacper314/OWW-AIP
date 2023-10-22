# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V_-2BPPDaDYNM3naAP4Pwh5ZaI3NWgzD
"""

# Instalowanie bibliotek, "!" ozacza wykonanie komendy w terminalu a nie w Pythonie:

!apt update && apt install xvfb
!pip install gym-notebook-wrapper
!pip install gym[classic_control]
!pip install gym[box2d]
!pip install opencv-python-headless
!pip install stable-baselines3
!pip install pyglet
!pip install 'shimmy>=0.2.1'

import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env4 = make_vec_env("LunarLander-v2", n_envs=1)

model = A2C("MlpPolicy", env4, verbose=1)
model.learn(total_timesteps=40000)
model.save("a2c_lunar")

import gnwrapper

env4 = gnwrapper.Monitor(gym.make("LunarLander-v2"),directory="./p/")

observation = env4.reset()
total_reward = 0
while True:
    action, _states = model.predict(observation)
    observation, reward, done, info = env4.step(action)
    total_reward += reward
    if done:
        env4.reset()
        break

env4.display()

print(f"Total rewad: {total_reward}")