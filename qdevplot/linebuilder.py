import numpy as np
from functools import partial


class LineBuilder:
    def __init__(self, line, ax, color, action=None, use_single_click=False):
        self.line = line
        self.ax = ax
        self.color = color
        self.xs = []
        self.ys = []
        self.ramp = []
        self.cid = line.figure.canvas.mpl_connect("button_press_event", self)
        self.counter = 0
        self.precision = 0.0005
        # self.action = partial(action, self)
        self.action = action
        self.markers = None
        self.line_segment = None
        self.use_single_click = use_single_click

    def __call__(self, event):
        if event.inaxes != self.line.axes:
            return
        if self.double_or_single_click(event):
            if self.counter == 0:
                self.xs = [event.xdata]
                self.ys = [event.ydata]
                self.ramp = []
                self.right_or_left(str(event.button))
                self.counter += 1
            elif self.click_same_point(event):
                self.xs.append(self.xs[self.counter - 1])
                self.ys.append(self.ys[self.counter - 1])
                self.right_or_left(str(event.button))
                self.plot_line()
                self.counter = self.counter + 1
            else:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.right_or_left(str(event.button))
                if self.markers:
                    self.markers.remove()
                self.plot_line()
                self.counter = self.counter + 1
            if self.action:
                self.action()

    def double_or_single_click(self, event) -> bool:
        return any((event.dblclick, self.use_single_click))

    def click_same_point(self, event) -> bool:
        return (
            np.abs(event.xdata - self.xs[self.counter - 1]) <= self.precision
            and np.abs(event.ydata - self.ys[self.counter - 1]) <= self.precision
        )

    def right_or_left(self, rightleftmouse: str) -> None:
        if rightleftmouse == "MouseButton.RIGHT":
            self.ramp.append(1)
        else:
            self.ramp.append(0)

    def plot_line(self) -> None:
        plateau = list(
            zip(
                *[
                    (self.xs[i], self.ys[i])
                    for i in range(len(self.ramp))
                    if self.ramp[i] == 0
                ]
            )
        )
        ramp = list(
            zip(
                *[
                    (self.xs[i], self.ys[i])
                    for i in range(len(self.ramp))
                    if self.ramp[i] == 1
                ]
            )
        )
        if plateau != []:
            self.markers = self.ax.scatter(
                plateau[0], plateau[1], s=120, color=self.color, marker="o"
            )
        if ramp != []:
            self.markers = self.ax.scatter(
                ramp[0], ramp[1], s=120, color="black", marker="x"
            )
        self.plot_line_segment()
        self.line.figure.canvas.draw()

    def plot_line_segment(self):
        if self.line_segment:
            self.ax.lines.remove(self.line_segment.pop(0))
        for i in range(len(self.xs)):
            if i > 0:
                if self.ramp[i] == 1:
                    linestyle = "-"
                else:
                    linestyle = "--"
                self.line_segment = self.ax.plot(
                    self.xs[i - 1 : i + 1],
                    self.ys[i - 1 : i + 1],
                    color=self.color,
                    linestyle=linestyle,
                )
