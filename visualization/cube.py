from visualization.side import RubickSide
from visualization.constants import CENTER_X, CENTER_Y, SIDE_SIZE, HALF_SIDE_SIZE

class CubeVisualization():
    def __init__(self, batch, initial_state):
        self._sides = [
            RubickSide(CENTER_X - SIDE_SIZE, CENTER_Y + HALF_SIDE_SIZE, batch),
            RubickSide(CENTER_X, CENTER_Y + HALF_SIDE_SIZE, batch),
            RubickSide(CENTER_X + SIDE_SIZE, CENTER_Y + HALF_SIDE_SIZE, batch),
            RubickSide(CENTER_X - SIDE_SIZE * 2, CENTER_Y + HALF_SIDE_SIZE, batch),
            RubickSide(CENTER_X - SIDE_SIZE, CENTER_Y + HALF_SIDE_SIZE + SIDE_SIZE, batch),
            RubickSide(CENTER_X - SIDE_SIZE, CENTER_Y + HALF_SIDE_SIZE - SIDE_SIZE, batch)
        ]
        self.update(initial_state)

    def update(self, state):
        for i, side in enumerate(self._sides):
            side.update(state[i])
