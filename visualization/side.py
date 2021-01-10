import pyglet
import numpy as np
from visualization.constants import CUBELET_SIZE, CUBE_SIZE, BORDER_COLOR, COLOR_DICT

class RubickSide():
    def __init__(self, x, y, batch):
        self._x = x
        self._y = y
        self._batch = batch
        self._elements = []

    def update(self, state):
        self._elements.clear()
        self._elements.extend(self(state))

    def __call__(self, state):
        shapes = pyglet.shapes

        def get_color_value(i):
            return np.argmax(state[i // CUBE_SIZE, i % CUBE_SIZE])

        rects = [shapes.Rectangle(x=self._x + CUBELET_SIZE * (i % CUBE_SIZE),
                                  y=self._y - CUBELET_SIZE * (1 + i // CUBE_SIZE),
                                  width=CUBELET_SIZE,
                                  height=CUBELET_SIZE,
                                  color=COLOR_DICT[get_color_value(i)],
                                  batch=self._batch)
                 for i in range(9)]

        horizontal_lines = [shapes.Line(x=self._x,
                                        x2=self._x + CUBELET_SIZE * CUBE_SIZE,
                                        y=self._y - i * CUBELET_SIZE,
                                        y2=self._y - i * CUBELET_SIZE,
                                        color=BORDER_COLOR,
                                        batch=self._batch)
                            for i in range(4)]

        vertical_lines = [shapes.Line(x=self._x + i * CUBELET_SIZE,
                                      x2=self._x + i * CUBELET_SIZE,
                                      y=self._y,
                                      y2=self._y - CUBELET_SIZE * CUBE_SIZE,
                                      color=BORDER_COLOR,
                                      batch=self._batch)
                          for i in range(4)]
        return [rects, horizontal_lines, vertical_lines]
