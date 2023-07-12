import numpy as np
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
import matplotlib.pyplot as plt


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

    def __init__(self, two_d_plotter, linescoop, **kwargs):
        _, (self.ax, self.ax_line) = plt.subplots(
            1, 2, sharex=False, sharey=False, constrained_layout=True
        )

        self.linescoop = linescoop
        if two_d_plotter is None:
            two_d_plotter = self.df_to_scatter
        _, self.cbaxes = two_d_plotter(axes=self.ax)

        self.ax.set_aspect(self.linescoop.normalized_aspect)

        canvas = self.ax.figure.canvas
        self.counter = 0
        self.projektion_plot = None
        self.contributing_points_plot = None
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
        self.linescoop = linescoop
        self.cid = self.line.add_callback(self.line_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect("draw_event", self.on_draw)
        canvas.mpl_connect("button_press_event", self.on_button_press)
        canvas.mpl_connect("key_press_event", self.on_key_press)
        canvas.mpl_connect("button_release_event", self.on_button_release)
        canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas = canvas

    def action(self):
        self.reset_counter_ax_plot_line_scoop()
        self.plot_contributing_points()
        self.plot_projection()
        self.ax.figure.canvas.draw()

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
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

    def plot_projection(self):
        remove_plot(self.projektion_plot)
        x_one, y_one, x_two, y_two = self.linescoop.projection()
        self.projektion_plot = self.ax.quiver(
            x_one,
            y_one,
            x_two,
            y_two,
            scale=1,
            angles="xy",
            scale_units="xy",
            width=0.001,
            headwidth=1,
            headlength=1,
        )

    def plot_contributing_points(self):
        remove_plot(self.contributing_points_plot)
        x_cord, y_cord = self.linescoop.contributing_points()
        self.contributing_points_plot = self.ax.scatter(
            x_cord, y_cord, marker=".", color="red"
        )

    def reset_counter_ax_plot_line_scoop(self):
        self.ax_line.cla()

        self.plot_line_scoop_from_df_points()

    def plot_line_scoop_from_df_points(self):
        self.linescoop.p1, self.linescoop.p2 = (self.xs[0], self.ys[0]), (
            self.xs[1],
            self.ys[1],
        )
        x_values, z_values, a, b, col_names = self.linescoop.line_scoop_from_points()
        self.ax_line.plot(
            x_values,
            z_values,
            linestyle="--",
            marker="o",
            label="data",
        )

        # self.ax_line.figure.canvas.draw()
        self.ax_line.figure.canvas.blit(self.ax_line.bbox)

        self.ax_line.legend(
            loc="best",
            fontsize=7,
        )
        self.ax_line.set_title(
            f"t({col_names[0]} + {a:.2f}{col_names[1]}) + {b:.2f}{col_names[1]}"
        )

        self.ax_line.set_xlabel("t")
        return self.ax_line


    # TODO: fix df_to_scatter
    def df_to_scatter(
        self, axes=None, x_label=None, y_label=None, cb_label=None, **kwargs
    ):
        x = self.df[self.X_original].values
        y = self.df[self.Y_original].values
        z = self.df[self.Z].values

        axes = if_not_ax_make_ax(axes)

        if x_label is None:
            x_label = self.X_original
        if y_label is None:
            y_label = self.Y_original
        if cb_label is None:
            cb_label = self.Z

        mappable = axes.scatter(x=x, y=y, c=z, cmap="viridis", **kwargs)
        cb = axes.figure.colorbar(mappable, ax=axes)
        cb.set_label(cb_label)
        axes.set_ylabel(y_label)
        axes.set_xlabel(x_label)

        return axes, cb


def remove_plot(the_plot) -> None:
    if the_plot is not None:
        the_plot.remove()
    return None
