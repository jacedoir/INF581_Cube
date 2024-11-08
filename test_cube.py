import numpy as np
import pytest
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import random
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Callable, Tuple
import collections
import gymnasium as gym
from gymnasium import spaces
import pygame
from cube import CubeEnv

@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
@pytest.mark.parametrize("n", [1,2,100])
def test_cube_env_n_steps(n,device):
    env = CubeEnv(device)
    env.reset()
    for _ in range(20):
        l=[np.random.randint(6) for _ in range(n)]
        for a in l:
            env.step(a)
        for a in reversed(l):
            opposite_action=a+1 if a%2==0 else a-1
            env.step(opposite_action)
        assert env._is_solved(env.state)