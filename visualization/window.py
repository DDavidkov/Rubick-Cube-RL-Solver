import pyglet
import pyglet.window.key as key
import pyglet.shapes as shapes

from common.move import RubickMove

from visualization.constants import WINDOW_WIDTH, WINDOW_HEIGHT, CAPTION, BACKGROUND
from visualization.text import ScreenText
from visualization.cube import CubeVisualization

class RubickCubeWindow(pyglet.window.Window):
    def __init__(self, env, bottom_labels, top_labels):
        super(RubickCubeWindow, self).__init__(width=WINDOW_WIDTH, height=WINDOW_HEIGHT, caption=CAPTION)

        self._env = env
        self._bottom_labels = bottom_labels
        self._top_labels = top_labels

        self._background_batch = shapes.Batch()
        self._text_batch = shapes.Batch()
        self._cube_batch = shapes.Batch()

        self._background = shapes.Rectangle(x=0,
                                            y=0,
                                            width=WINDOW_WIDTH,
                                            height=WINDOW_HEIGHT,
                                            color=BACKGROUND,
                                            batch=self._background_batch)
        self._text = ScreenText(self._text_batch, top_labels, bottom_labels)
        self._cube = CubeVisualization(self._cube_batch, self._env.get_pretty_state())

    def on_draw(self):
        self.clear()
        self._background_batch.draw()
        self._text_batch.draw()
        self._cube_batch.draw()

    def on_key_press(self, symbol, modifiers):
        action = None
        has_shift = modifiers & key.MOD_SHIFT
        has_ctrl = modifiers & key.MOD_CTRL
        if symbol == key.W:
            action = RubickMove.U_PRIM if has_shift else RubickMove.U
        elif symbol == key.S:
            action = RubickMove.D_PRIM if has_shift else RubickMove.D
        elif symbol == key.Q:
            action = RubickMove.F_PRIM if has_shift else RubickMove.F
        elif symbol == key.E:
            action = RubickMove.B_PRIM if has_shift else RubickMove.B
        elif symbol == key.A:
            action = RubickMove.L_PRIM if has_shift else RubickMove.L
        elif symbol == key.D:
            action = RubickMove.R_PRIM if has_shift else RubickMove.R
        elif symbol == key.SPACE:
            action = RubickMove.RESET
        elif symbol == key.ENTER:
            action = RubickMove.SCRAMBLE
        elif symbol == key.Z:
            action = RubickMove.UNDO if has_ctrl and self._env.can_undo() else None
        elif symbol == key.Y:
            action = RubickMove.REDO if has_ctrl and self._env.can_redo() else None

        if action is not None:
            self._env.step(action)
            self._cube.update(self._env.get_pretty_state())

