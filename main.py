import pyglet

from cubelet_cube.rubick_env import RubickEnv
from visualization.window import RubickCubeWindow

top_labels = [
    "Q - F Move, Shift + Q - F' Move",
    "E - B Move, Shift + E - B' Move",
    "D - R Move, Shift + D - R' Move",
    "A - L Move, Shift + A - L' Move",
    "W - U Move, Shift + W - U' Move",
    "S - D Move, Shift + S - D' Move"
]

bottom_labels = [
    "Enter - Scramble the cube",
    "Space - Reset the cube",
    "Ctrl + Z - Undo",
    "Ctrl + Y - Redo"
]

env = RubickEnv()
window = RubickCubeWindow(env, bottom_labels, top_labels)
pyglet.app.run()
