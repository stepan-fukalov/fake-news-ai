from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import time
import logging

import numpy as np 

import pygame
import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding

from bug_utils import Maze

logging.basicConfig(filename="logging.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()
 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

HEIGHT = 19
WIDTH = 29

class BugEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.action_space = spaces.Discrete(HEIGHT*WIDTH)
        self.observation_space = spaces.MultiBinary(HEIGHT*WIDTH)
        self.maze = Maze()
        
        self.render_mode = render_mode
        
        # pygame utils
        self.window_size = (64*HEIGHT, 64*WIDTH)
        self.cell_size = (
            self.window_size[0] // HEIGHT,
            self.window_size[1] // WIDTH,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

        if(self.render_mode == "human"):
            if self.window_surface is None:
                pygame.init()
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)

            if self.clock is None:
                self.clock = pygame.time.Clock()
            if self.hole_img is None:
                file_name = path.join(path.dirname(__file__), "img/hole.png")
                self.hole_img = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.cracked_hole_img is None:
                file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
                self.cracked_hole_img = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.ice_img is None:
                file_name = path.join(path.dirname(__file__), "img/ice.png")
                self.ice_img = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.goal_img is None:
                file_name = path.join(path.dirname(__file__), "img/goal.png")
                self.goal_img = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.start_img is None:
                file_name = path.join(path.dirname(__file__), "img/stool.png")
                self.start_img = pygame.transform.scale(
                    pygame.image.load(file_name), self.cell_size
                )
            if self.elf_images is None:
                elfs = [
                    path.join(path.dirname(__file__), "img/elf_left.png"),
                    path.join(path.dirname(__file__), "img/elf_down.png"),
                    path.join(path.dirname(__file__), "img/elf_right.png"),
                    path.join(path.dirname(__file__), "img/elf_up.png"),
                ]
                self.elf_images = [
                    pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                    for f_name in elfs
                ]

    def step(self, action):
        x, y = action // WIDTH, action % WIDTH
        try:
            self.grid[x][y] = 1
        except Exception as e:
            logger.error(f"{self.cur} {x} {y} {action} {e}")
            return self.s, 0, True, False, {}

        self.s[self.cur] = 1
        self.cur += 1
        self.render()

        self.maze.update(x, y, 1)
        self.prev_reward = self.reward
        self.reward = self.maze.get_reward()

        self.terminated = (self.reward < self.prev_reward or self.cur == HEIGHT*WIDTH or self.reward <= 0)

        return self.s, self.reward-self.prev_reward, self.terminated, False, {}

    def reset(self, seed=None, options=None):
        # self.grid = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        self.grid = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        # self.grid = [[2]*WIDTH for _ in range(HEIGHT)]
        self.cur = 0
        self.s = np.zeros(HEIGHT*WIDTH, dtype=np.int8)
        self.reward = 0
        self.terminated = False
        self.maze.reset()
        self.render()

        return self.s, {}

    def render(self):
        if self.render_mode == "human":
            self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        # self.grid = self.grid.tolist()
        # assert isinstance(self.grid, list), f"self.grid should be a list or an array, got {self.grid}"
        for y in range(HEIGHT):
            for x in range(WIDTH):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if self.grid[y][x] == 2:
                    self.window_surface.blit(self.ice_img, pos)
                elif self.grid[y][x] == 0:
                    self.window_surface.blit(self.hole_img, pos)
                else:
                    self.window_surface.blit(self.goal_img, pos)
                # elif self.grid[y][x] == b"S":
                    # self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        # bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        # cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        # last_action = self.lastaction if self.lastaction is not None else 1
        # elf_img = self.elf_images[last_action]

        # if self.grid[bot_row][bot_col] == b"H":
        #     self.window_surface.blit(self.cracked_hole_img, cell_rect)
        # else:
        #     self.window_surface.blit(elf_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    # def _render_text(self):
    #     # self.grid = self.self.grid.tolist()
    #     outfile = StringIO()

    #     # row, col = self.s // self.ncol, self.s % self.ncol
    #     # self.grid[0] = [[c.decode("utf-8") for c in line] for line in self.grid[0]]
    #     # self.grid[0][row][col] = utils.colorize(self.grid[0][row][col], "red", highlight=True)
    #     if self.lastaction is not None:
    #         outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
    #     else:
    #         outfile.write("\n")
    #     outfile.write("\n".join("".join(line) for line in self.grid) + "\n")

    #     with closing(outfile):
    #         return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

def save_maze(env, filename):
    map = np.ones((HEIGHT+2, WIDTH+2))
    for x in range(HEIGHT):
        for y in range(WIDTH):
            map[1+x][1+y] = env.grid[x][y]
    with open(filename, "w") as txt_file:
        for line in map:
            txt_file.write(''.join([('#' if c == 1 else '.') for c in line]) + "\n") # works with any number of elements in a line