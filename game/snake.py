import random
import pygame
import numpy as np
from enum import Enum
from typing import Optional


class Direction(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class Fruit:
    block_size = None
    window_size = None
    pos = None
    color = None

    def __init__(self, block_size: int, window_size: int, color: tuple):
        """
        :param block_size: type (int): the size of each block in the window

        :param window_size: type (int): the size of the game window

        :param color: a 3-tuple representing rgb values
        """
        self.block_size = block_size
        self.window_size = window_size
        self.color = color

        self.reset()

    def render(self, surface, instance=pygame):
        """
        Renders the fruit on the input surface

        :param surface: the surface to render on

        :param instance: instance of the window
        """
        instance.draw.rect(
            surface,
            self.color,
            (self.pos[0], self.pos[1], self.block_size, self.block_size)
        )

    def reset(self, body: Optional[np.array] = None):
        """
        Resets the fruit to a random position

        :param body: the snake's body
        """
        num_blocks = self.window_size // self.block_size

        self.pos = np.random.randint(1, num_blocks-1, 2, dtype='int') * self.block_size
        if body is None:
            return

        while True:
            self.pos = np.random.randint(1, num_blocks-1, 2, dtype='int') * self.block_size

            if (self.pos == body).all(axis=1).any():
                break


class Snake:
    length = None
    direction = None
    body = []
    block_size = None
    window_size = None
    color = None

    def __init__(self, block_size: int, window_size: int, color: tuple):
        """
        :param block_size: type (int): the size of each block in the window

        :param window_size: type (int): the size of the game window

        :param color: a 3-tuple representing rgb values
        """
        self.block_size = block_size
        self.window_size = window_size
        self.color = color

        # a dictionary to convert direction to appropriate step size in the x and y direction
        self.direction_to_step = {
            Direction.RIGHT: np.array((self.block_size, 0)),
            Direction.UP: np.array((0, self.block_size)),
            Direction.LEFT: np.array((-self.block_size, 0)),
            Direction.DOWN: np.array((0, -self.block_size))
        }

        self.reset()

    def move(self):
        """
        A method which moves the snake according to it's current direction
        """
        head = self.body[-1]
        step = self.direction_to_step[self.direction]
        next_head = head + step

        self.body = np.concatenate((self.body, next_head), axis=0)
        if self.length < len(self.body):
            self.body.pop(0)

    def direction_change(self, inp_direction: Direction):
        """
        Method to change the direction of the snake.
        Only change if the inp_direction is not in the opposite direction.

        :param inp_direction: input direction
        """
        if (self.direction.value - inp_direction.value) % 2:
            self.direction = inp_direction

    def eat_check(self, fruit: Fruit):
        """
        Checks if the snake ate the fruit.
        If the fruit is eaten, increases the snake's length and returns True. Otherwise returns false.

        :param fruit: the fruit object

        :return: boolean value: True or False
        """
        head = self.body[-1]

        if (head == fruit.pos).all():
            self.length += 1
            fruit.reset(body=self.body)
            return True
        return False

    def self_collide_check(self):
        """
        Checks if the snake collided with itself (bit itself).

        :return: boolean value: True or False
        """
        head = self.body[-1]

        if (head == self.body[:-1]).all(axis=1).any():
            return True
        return False

    def border_collide_check(self):
        """
        Checks if the snake collided with the outlines of the window.

        :return: boolean value: True or False
        """
        head = self.body[-1]

        if head[0] >= self.window_size or head[0] < 0:
            return True
        if head[1] >= self.window_size or head[0] < 0:
            return True
        return False

    def wall_collide_check(self):
        pass

    def dead_check(self):
        """
        Checks if the snake died; either by biting itself or hitting a wall or hitting the window outline

        :return: boolean value: True or False
        """
        return self.self_collide_check() or self.wall_collide_check() or self.border_collide_check()

    def render(self, surface, instance=pygame):
        """
        Renders the fruit on the input surface

        :param surface: the surface to render on

        :param instance: instance of the window

        :return: None
        """
        for piece in self.body:
            instance.draw.rect(
                surface,
                self.color,
                (piece[0], piece[1], self.block_size, self.block_size)
            )

    def reset(self):
        """
        A method to reset the snake back to it's default position.
        """
        assert self.window_size >= 40

        self.length = 2
        num_blocks = self.window_size // self.block_size

        x_pos = (num_blocks // 2) * self.block_size
        self.body = np.array((x_pos, 20), (x_pos, 40))
        self.direction = Direction.DOWN


class Wall:
    segments = []

    def __init__(self, block_size: int, window_size: int, color: tuple, segment_length: int):
        self.block_size = block_size
        self.window_size = window_size
        self.color = color
        self.length = 5
        self.segment_length = segment_length

        self.reset()

    def render(self, surface, instance=pygame):
        for segment in self.segments:
            instance.draw.rect(
                surface,
                self.color,
                (segment[0], segment[1], self.block_size, self.block_size)
            )

    def reset(self):
        pass

    def add_segment(self):
        pass
