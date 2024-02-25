from typing import Callable, List
import gymnasium as gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pygame
from gymnasium import spaces
from tqdm.notebook import tqdm

# - https://gymnasium.farama.org/api/env/
# - https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py


class CubeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.size = 2  # size of rubik's cube
        self.state = np.chararray(
            (6, self.size, self.size), unicode=True
        )  # initialize cube config

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
        self._color_dict = {
            "r": "red",
            "g": "green",
            "b": "blue",
            "o": "orange",
            "y": "yellow",
            "w": "black",
            "": "none",
        }
        self._action_map = {  # !!!!!! WARNING !!!!!! CHANGE WITH SIZE
            0: lambda: self._horizontale_rotation(0, 1),
            1: lambda: self._horizontale_rotation(0, -1),
            2: lambda: self._verticale_rotation(0, 1),
            3: lambda: self._verticale_rotation(0, -1),
            4: lambda: self._face_rotation(0, 1),
            5: lambda: self._face_rotation(0, -1),
        }

        self.window_size = 512  # The size of the PyGame window
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.state

    def _get_info(self):
        """Fully known environment"""
        return {}

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def reset(self, n_move, seed=None, options=None):
        """Puts Rubik's cube back to fully solved"""
        self.state[0] = np.array([["o"] * self.size] * self.size)
        self.state[1] = np.array([["w"] * self.size] * self.size)
        self.state[2] = np.array([["g"] * self.size] * self.size)
        self.state[3] = np.array([["y"] * self.size] * self.size)
        self.state[4] = np.array([["r"] * self.size] * self.size)
        self.state[5] = np.array([["b"] * self.size] * self.size)

        self.shuffle(n_move)
        observation = self._get_obs()
        info = self._get_info()

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

    def reward(self, state):
        # TODO
        return int(self._is_solved(state))

    def render(self):
        """Renders one frame"""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):

        # init window
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _is_solved(self, state):
        for i in range(6):
            if len(np.unique(state[i])) != 1:  # check for all face uniqueness
                return False
        return True

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
            new_state[3, face, :] = self.state[5, :, self.size - 1 - face]
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
                5, :, self.size - 1 - face
            ]
            new_state[5, :, face] = self.state[3, face, :]
            new_state[3, face, :] = self.state[0, :, self.size - 1 - face]
            if face == 0:
                new_state[2] = np.rot90(self.state[2])
            elif face == self.size - 1:
                new_state[4] = np.rot90(self.state[4], 3)
        self.state = new_state

    def shuffle(self, n_moves):
        # TODO (not important) able to use seed
        observations = []
        infos = []
        for i in range(n_moves):
            face = np.random.randint(self.size)
            direction = np.random.choice([-1, 1])
            operation = np.random.randint(3)
            if operation == 0:
                self.horizontale_rotation(face, direction)
            elif operation == 1:
                self.verticale_rotation(face, direction)
            else:
                self.face_rotation(face, direction)
