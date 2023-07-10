import numpy as np
from matplotlib.lines import Line2D
from matplotlib.artist import Artist


class LineMover:
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, action=None, **kwargs):
        self.ax = ax
        self.action = action
        canvas = ax.figure.canvas
        self.counter = 0

        self.xs, self.ys = [], []
        self.line = Line2D(
            self.xs,
            self.ys,
            color="r",
            linestyle="--",
            marker="1",
            markerfacecolor="r",
            animated=True,
        )
        self.ax.add_line(self.line)

        self.cid = self.line.add_callback(self.line_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect("draw_event", self.on_draw)
        canvas.mpl_connect("button_press_event", self.on_button_press)
        canvas.mpl_connect("key_press_event", self.on_key_press)
        canvas.mpl_connect("button_release_event", self.on_button_release)
        canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas = canvas

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        # self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        # do not need to blit here, this will fire before the screen is
        # updated

    def line_changed(self):
        """This method is called whenever the pathpatch object is called."""
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        xy = np.asarray(self.line.get_data())
        xyt = self.line.get_transform().transform(xy.T)

        xt, yt = xyt.T[0, :], xyt.T[1, :]

        d = np.hypot(xt - event.x, yt - event.y)
        (indseq,) = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if event.dblclick:
            if self.counter == 0:
                self.xs = [event.xdata, event.xdata]
                self.ys = [event.ydata, event.ydata]
                self.draw_line()
                # self.line.set_data(self.xs, self.ys)
                # self.canvas.restore_region(self.background)
                # self.ax.draw_artist(self.line)
                # self.canvas.blit(self.ax.bbox)

                self.counter += 1
            else:
                self.xs[1] = event.xdata
                self.ys[1] = event.ydata
                self.counter = 0
                self.draw_line()

        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def on_key_press(self, event):
        """Callback for key presses."""
        if not event.inaxes:
            return
        if event.key == "t":
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == "d":
            ind = self.get_ind_under_point(event)
            # if ind is not None:
            # self.line.set_data(zip(*self.poly.xy))
            # elif event.key == 'i':
            """xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.insert(
                        self.poly.xy, i+1,
                        [event.xdata, event.ydata],
                        axis=0)
                    self.line.set_data(zip(*self.poly.xy))
                    break
                    """

        if self.line.stale:
            self.canvas.draw_idle()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.xs[self._ind] = x
        self.ys[self._ind] = y

        # print(f"blax {self.xs}")
        # print(f"blay {self.ys}")
        self.draw_line()

    def draw_line(self):
        if self.xs:
            self.line.set_data(self.xs, self.ys)
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.action()
            self.canvas.blit(self.ax.bbox)
            # self.canvas.draw()
