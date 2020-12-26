import pyglet.text as text

from visualization.constants import WINDOW_WIDTH, WINDOW_HEIGHT, FONT_SIZE, FONT_NAME, TEXT_COLOR

class ScreenText:
    def __init__(self, batch, top_labels, bottom_labels):
        self._labels = []
        self._batch = batch

        self.draw_labels(top_labels, bottom_labels)

    def draw_labels(self, top_labels, bottom_labels):
        self._labels.clear()

        for i, label in enumerate(top_labels):
            self._labels.append(text.Label(label,
                                           font_name=FONT_NAME,
                                           font_size=FONT_SIZE,
                                           x=WINDOW_WIDTH - FONT_SIZE - 4,
                                           y=WINDOW_HEIGHT - (FONT_SIZE + 4) * (i + 1),
                                           anchor_x='right', anchor_y='top',
                                           color=TEXT_COLOR,
                                           batch=self._batch))

        bottom_labels_length = len(bottom_labels)
        for i, label in enumerate(bottom_labels):
            self._labels.append(text.Label(label,
                                           font_name=FONT_NAME,
                                           font_size=FONT_SIZE,
                                           x=WINDOW_WIDTH - FONT_SIZE - 4,
                                           y=(FONT_SIZE + 4) * (bottom_labels_length - i + 1),
                                           anchor_x='right', anchor_y='top',
                                           color=TEXT_COLOR,
                                           batch=self._batch))


