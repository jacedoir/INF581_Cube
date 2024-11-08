from typing import Callable, List
import gymnasium as gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pygame
from gymnasium import spaces
from tqdm.notebook import tqdm
import torch

# - https://gymnasium.farama.org/api/env/
# - https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py

class CubeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, device: torch.device, render_mode=None):
        self.size = 2  # size of rubik's cube
        self.state = np.chararray(
            (6, self.size, self.size), unicode=True
        )  # initialize cube config
        self.device = device
        # a chaque fois qu'on rajoute une dim, on rajoute 6 coups possible (2 par nouveau plan, x3 car nb d'axes de l'espace)
        self.action_space = spaces.Discrete(2 * 3 * (self.size - 1))

        # render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._position_dict = {
            0: "left",
            1: "top",
            2: "front",
            3: "bottom",
            4: "back",
            5: "right",
        }
        self._face_position_dict = {  # when rendering the cube
            0: (0, 1),  #  T
            1: (1, 0),  # LFR
            2: (1, 1),  #  Bo
            3: (1, 2),  #  Ba
            4: (1, 3),
            5: (2, 1),
        }
        self._color_dict = {
            "r": (255, 0, 0),
            "g": (0, 255, 0),
            "b": (0, 0, 255),
            "o": (255, 120, 0),
            "y": (255, 255, 0),
            "w": (220, 220, 220),
        }
        self.colors= ["r", "g", "b", "o", "y", "w"]

        self.color_encoding = {
            "r": np.array([1, 0, 0, 0, 0, 0]),
            "g": np.array([0, 1, 0, 0, 0, 0]),
            "b": np.array([0, 0, 1, 0, 0, 0]),
            "o": np.array([0, 0, 0, 1, 0, 0]),
            "y": np.array([0, 0, 0, 0, 1, 0]),
            "w": np.array([0, 0, 0, 0, 0, 1]),
        }
        self._action_map = {  # !!!!!! WARNING !!!!!! CHANGE WITH SIZE
            0: lambda: self._horizontale_rotation(0, 1),
            1: lambda: self._horizontale_rotation(0, -1),
            2: lambda: self._verticale_rotation(0, 1),
            3: lambda: self._verticale_rotation(0, -1),
            4: lambda: self._face_rotation(0, 1),
            5: lambda: self._face_rotation(0, -1),
        }

        self._window_dims = (384, 512)  # The size of the PyGame window
        self._face_size = self._window_dims[1] // 4  # size of face in pixels
        self._square_dims = self._tuple_mul(
            (1, 1), self._face_size / self.size
        )  # size of one square in pixels
        self.window = None
        self.clock = None

    def _get_obs(self):
        res = np.array(
            [
                [
                    [self.color_encoding[self.state[i][j][k]] for k in range(self.size)]
                    for j in range(self.size)
                ]
                for i in range(6)
            ]
        )
        res = torch.tensor(res, dtype=torch.float32, device=self.device).flatten()
        return res

    def _get_info(self):
        """Fully known environment"""
        return {}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def reset(self, n_moves=0, seed=None, options=None):
        """Puts Rubik's cube back to fully solved if n_moves is set to 0, otherwise solve and perform an n-shuffle"""
        self.state[0] = np.array([["o"] * self.size] * self.size)
        self.state[1] = np.array([["w"] * self.size] * self.size)
        self.state[2] = np.array([["g"] * self.size] * self.size)
        self.state[3] = np.array([["y"] * self.size] * self.size)
        self.state[4] = np.array([["r"] * self.size] * self.size)
        self.state[5] = np.array([["b"] * self.size] * self.size)
        
        # Shuffle if asked
        if n_moves > 0:
            for _ in range(n_moves):
                action = self.action_space.sample()
                observation, _, _, _, _ = self.step(action)

        observation = self._get_obs()
        info = self._get_info()

        # render if necessary
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        # rotate cube according to action map
        self._action_map[action]()

        # get info and everything
        terminated = self._is_solved(self.state)
        reward = self.reward(self.state)
        observation = self._get_obs()
        info = self._get_info()

        # render if necessary
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def step(self, action):

        # rotate cube according to action map
        self._action_map[action]()

        # get info and everything
        terminated = self._is_solved(self.state)
        reward = self.reward(self.state)
        observation = self._get_obs()
        info = self._get_info()

        # render if necessary
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def simulate_step(self, state, action):
        """Simulate a step without changing the state of the environment."""
        copy_state = self.state.copy()
        if torch.is_tensor(state):
            state = self.from_tensor_to_state(state)
        self.state = state
        # rotate cube according to action map
        self._action_map[action]()

        # get info and everything
        terminated = self._is_solved(self.state)
        reward = self.reward(self.state)
        observation = self._get_obs()
        info = self._get_info()

        # render if necessary
        if self.render_mode == "human":
            self._render_frame()

        self.state = copy_state
        return observation, reward, terminated, False, info

    def reward(self, state):
        """Reward function, containing minimal human knowledge"""
        return -1 + .3 * self._count_solved_faces(state)

    def render(self):
        """Renders one frame"""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _tuple_mul(self, tuple, coef):
        return (tuple[0] * coef, tuple[1] * coef)

    def _tuple_add(self, tuple1, tuple2):
        return (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1])

    def _tuple_had(self, tuple1, tuple2):
        return (tuple1[0] * tuple2[0], tuple1[1] * tuple2[1])

    def _render_face(self, canvas, face, face_position):
        for i in range(self.size):
            for j in range(self.size):
                color = self._color_dict[face[i, j]]
                pos = self._tuple_mul(face_position, self._face_size)
                pos = self._tuple_add(pos, self._tuple_had(self._square_dims, (j, i)))
                pygame.draw.rect(canvas, color, pygame.Rect(pos, self._square_dims))
                pygame.draw.rect(
                    canvas, (0, 0, 0), pygame.Rect(pos, self._square_dims), width=1
                )

    def animate_frames(self, states):
        """Short animation of the cube solving process."""

        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure( figsize=(8,8) )
        frames=[]
        for state in states:
            frames.append(self._render_frame(state,display=False))
        a = frames[0]
        im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)

        def animate_func(i):
            if i % 1 == 0:
                print( '.', end ='' )

            im.set_array(frames[i])
            return [im]

        anim = animation.FuncAnimation(
                                    fig, 
                                    animate_func, 
                                    frames = len(states),
                                    interval = 1000, # in ms
                                    )

        anim.save('anim.gif', fps=1)

    def _render_frame(self,state=None,display=True):
        if state is None :
            state=self.state

        # init window
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self._window_dims)

            # TODO
            # run = True
            # while run:
            #    for event in pygame.event.get():
            #        if event.type == pygame.QUIT:
            #            run = False
            #

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self._window_dims)
        canvas.fill((255, 255, 255))

        # draw face by face
        for face in self._face_position_dict.keys():
            self._render_face(canvas, state[face], self._face_position_dict[face])

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:

            # show using plt
            pixels = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
            plt.imshow(pixels)

    def _count_solved_faces(self, state):
        solved = 0
        for i in range(6):
            if len(np.unique(state[i])) == 1:
                solved += 1
        return solved

    def _is_solved(self, state):
        return self._count_solved_faces(state) == 6

    def _horizontale_rotation(self, row, direction):
        new_state = np.copy(self.state)
        if row > self.size - 1:
            raise ValueError("Row number must be between 0 and ", self.size - 1)
        if direction == 1:  # To the right
            new_state[2, row, :] = self.state[0, row, :]
            new_state[5, row, :] = self.state[2, row, :]
            new_state[4, row, :] = self.state[5, row, :]
            new_state[0, row, :] = self.state[4, row, :]
            if row == 0:
                new_state[1] = np.rot90(self.state[1])
            elif row == self.size - 1:
                new_state[3] = np.rot90(self.state[3], 3)
        elif direction == -1:  # To the left
            new_state[0, row, :] = self.state[2, row, :]
            new_state[2, row, :] = self.state[5, row, :]
            new_state[5, row, :] = self.state[4, row, :]
            new_state[4, row, :] = self.state[0, row, :]
            if row == 0:
                new_state[1] = np.rot90(self.state[1], 3)
            elif row == self.size - 1:
                new_state[3] = np.rot90(self.state[3])
        self.state = new_state

    def _verticale_rotation(self, column, direction):
        new_state = np.copy(self.state)
        if column > self.size - 1:
            raise ValueError("Column number must be between 0 and ", self.size - 1)
        if direction == 1:  # to the top
            new_state[2, :, column] = self.state[3, :, column]
            new_state[3, :, column] = self.state[4, :, column]
            new_state[4, :, column] = self.state[1, :, column]
            new_state[1, :, column] = self.state[2, :, column]
            if column == 0:
                new_state[0] = np.rot90(self.state[0])
            elif column == self.size - 1:
                new_state[5] = np.rot90(self.state[5], 3)
        elif direction == -1:  # to the bottom
            new_state[3, :, column] = self.state[2, :, column]
            new_state[4, :, column] = self.state[3, :, column]
            new_state[1, :, column] = self.state[4, :, column]
            new_state[2, :, column] = self.state[1, :, column]
            if column == 0:
                new_state[0] = np.rot90(self.state[0], 3)
            elif column == self.size - 1:
                new_state[5] = np.rot90(self.state[5])
        self.state = new_state

    def _face_rotation(self, face, direction):
        new_state = np.copy(self.state)
        if face > self.size - 1:
            raise ValueError("Face number must be between 0 and ", self.size - 1)
        if direction == 1:  # clockwise
            new_state[1, self.size - 1 - face, :] = self.state[
                0, :, self.size - 1 - face
            ]
            new_state[5, :, face] = self.state[1, self.size - 1 - face, :]
            new_state[3, face, :] = self.state[5, :, face]
            new_state[0, :, self.size - 1 - face] = self.state[3, face, :]
            if face == 0:
                new_state[2] = np.rot90(self.state[2], 3)
            elif face == self.size - 1:
                new_state[4] = np.rot90(self.state[4])
                
        elif direction == -1:  # counterclockwise
            new_state[0, :, self.size - 1 - face] = self.state[
                1, self.size - 1 - face, :
            ]
            new_state[1, self.size - 1 - face, :] = self.state[
                5, :, face
            ]
            new_state[5, :, face] = self.state[3, face, :]
            new_state[3, face, :] = self.state[0, :, self.size - 1 - face]
            if face == 0:
                new_state[2] = np.rot90(self.state[2])
            elif face == self.size - 1:
                new_state[4] = np.rot90(self.state[4], 3)
        self.state = new_state

    
    def from_tensor_to_state(self, tensor):
        tensor= tensor.view((6, self.size, self.size,6))
        state = np.chararray((6, self.size, self.size), unicode=True)
        for i in range(6):
            for j in range(self.size):
                for k in range(self.size):
                    state[i][j][k] = self.colors[torch.argmax(tensor[i][j][k])]
        return state
    
    def batch_shuffle(self, n_moves, batch_size=1):
        # TODO (not important) able to use seed
        batched_obs = torch.zeros(
            (n_moves,batch_size,6 * 6 * self.size * self.size), device=self.device
        )
        for i in range(batch_size):
            self.reset()
            observations = []
            for j in range(n_moves):
                action = self.action_space.sample()
                observation, _, _, _, _ = self.step(action)
                batched_obs[j][i]=observation
        return torch.tensor(batched_obs, device=self.device)