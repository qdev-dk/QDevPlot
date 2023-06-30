from turtle import width
import matplotlib.pyplot as plt
import numpy as np
from qcodes.dataset.plotting import plot_dataset
from qcodes.dataset.data_set import load_by_id
from qdevplot.linebuilder import LineBuilder
from qdevplot.plotfunctions import if_not_ax_make_ax


class LineScoop:
    def __init__(
        self,
        data,
        delta: float = 0.01,
        normalize_axes=True,
        do_plot_projection=True,
        do_mark_contributing_points=True,
        aspect=1,
    ):
        self.delta = delta
        self.do_plot_projection = do_plot_projection
        self.do_mark_contributing_points = do_mark_contributing_points
        self.aspect = aspect**0.5
        _, (self.ax_2d, self.ax_line) = plt.subplots(
            1, 2, sharex=False, sharey=False, constrained_layout=True
        )

        self.ax_2d, self.cbaxes = plot_dataset(data, axes=self.ax_2d)
        self.df = data.to_pandas_dataframe().reset_index()

        self.normalize_axes = normalize_axes
        col_names = list(self.df.columns)
        self.X = col_names[0]
        self.Y = col_names[1]
        self.Z = col_names[2]
        self.X_original = col_names[0]
        self.Y_original = col_names[1]

        self.min_X_original = self.df[self.X].min()
        self.max_X_original = self.df[self.X].max()
        self.min_Y_original = self.df[self.Y].min()
        self.max_Y_original = self.df[self.Y].max()

        self.df[f"{self.X}_normalized"] = (
            (1 / self.aspect)
            * (self.df[self.X] - self.df[self.X].min())
            / (self.df[self.X].max() - self.df[self.X].min())
        )

        self.df[f"{self.Y}_normalized"] = (
            self.aspect
            * (self.df[self.Y] - self.df[self.Y].min())
            / (self.df[self.Y].max() - self.df[self.Y].min())
        )

        if self.normalize_axes:
            self.X = f"{self.X}_normalized"
            self.Y = f"{self.Y}_normalized"

        self.normalized_aspect = (self.max_X_original - self.min_X_original) / (
            self.max_Y_original - self.min_Y_original
        )
        self.ax_2d.set_aspect(self.aspect**2 * self.normalized_aspect)

        self.contributing_points_plot = None
        self.projection_plot = None
        self.df_plot = None

        self.line = LineBuilder(
            self.ax_2d,
            "red",
            action=self.reset_counter_ax_plot_line_scoop,
            use_single_click=True,
        )

    @classmethod
    def from_run_id(cls, run_id: int, delta: float = 0.01) -> "LineScoop":
        data = load_by_id(run_id)
        return cls(data, delta)

    def reset_counter_ax_plot_line_scoop(self):
        if self.line.counter <= 1:
            return

        self.line.counter = 0
        self.ax_line.cla()
        self.contributing_points_plot = remove_plot(self.contributing_points_plot)
        self.projection_plot = remove_plot(self.projection_plot)
        self.plot_line_scoop_from_df_points()

    def plot_line_scoop_from_df_points(self, labels=None):
        print(f"p1 {self.p1}")
        print(f"p2 {self.p2}")
        a, b = get_line_from_two_points(self.p1, self.p2)
        self.ax_line = self.plot_line_scoop_from_df_line(
            a,
            b,
        )
        if self.do_plot_projection:
            self.plot_projection()
        if self.do_mark_contributing_points:
            self.mark_contributing_points()
        return self.ax_line

    def plot_line_scoop_from_df_line(
        self,
        a,
        b,
        labels=None,
    ):
        self.ax_line = if_not_ax_make_ax(self.ax_line)
        self.df_plot = self.get_scoop_line(
            a,
            b,
            self.delta,
            xlim=self.xlim,
            ylim=self.ylim,
        )
        col_names = list(self.df_plot.columns)
        self.ax_line.plot(
            self.df_plot["crossing_x"].tolist(),
            self.df_plot[self.Z].tolist(),
            linestyle="--",
            marker="o",
            label="data",
        )

        self.ax_line.figure.canvas.draw()

        self.ax_line.legend(
            loc="best",
            fontsize=7,
            # f"t({col_names[0]} + {a:.2f}{col_names[1]}) + {b:.2f}{col_names[1]}"
        )
        self.ax_line.set_title(
            f"t({col_names[0]} + {a:.2f}{col_names[1]}) + {b:.2f}{col_names[1]}"
        )
        if labels:
            self.ax_line.set_xlabel(
                f"t({labels[0]} + {a:.2f}{labels[1]}) + {b:.2f}{labels[1]}"
            )
        else:
            self.ax_line.set_xlabel("t")
        return self.ax_line

    def get_scoop_line(
        self, a, b, delta, xlim=(-np.inf, np.inf), ylim=(-np.inf, np.inf), scale=None
    ):
        self.df = self.ad_dist_and_crossing(a, b)
        return self.df[
            (self.df["dist"] < delta)
            * self.df["crossing_x"].between(*xlim, "both")
            * self.df["crossing_y"].between(*ylim, "both")
        ].sort_values(["crossing_x", "crossing_y"])[
            [
                self.X_original,
                self.Y_original,
                self.X,
                self.Y,
                self.Z,
                "crossing_x",
                "crossing_y",
                "dist",
            ]
        ]

    def ad_dist_and_crossing(self, a, b):
        temp = np.array(
            [
                *self.df.apply(
                    lambda x: get_dist_between_line_and_point(
                        a, b, x[self.X], x[self.Y]
                    ),
                    axis=1,
                )
            ]
        )
        self.df["crossing_x"] = temp[:, 0]
        self.df["crossing_y"] = temp[:, 1]
        self.df["dist"] = temp[:, 2]
        return self.df

    @property
    def p1(self):
        return self.normalize_or_not((self.line.xs[0], self.line.ys[0]))

    @property
    def p2(self):
        return self.normalize_or_not((self.line.xs[1], self.line.ys[1]))

    @property
    def xlim(self):
        return (min(self.p1[0], self.p2[0]), max(self.p1[0], self.p2[0]))

    @property
    def ylim(self):
        return (min(self.p1[1], self.p2[1]), max(self.p1[1], self.p2[1]))

    def df_to_scatter(
        self, x_label=None, y_label=None, cb_label=None, ax=None, **kwargs
    ):
        x = self.df[self.X].values
        y = self.df[self.Y].values
        z = self.df[self.Z].values

        ax = if_not_ax_make_ax(ax)
        # ax = style_jose(ax)

        if x_label is None:
            x_label = self.X
        if y_label is None:
            y_label = self.Y
        if cb_label is None:
            cb_label = self.Z

        mappable = ax.scatter(
            x=x, y=y, c=z, cmap="viridis", **kwargs  # vmin=0.00, vmax=4.0,
        )
        cb = ax.figure.colorbar(mappable, ax=ax)
        cb.set_label(cb_label)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

        return ax, cb

    def mark_contributing_points(self):
        x_cord = self.df_plot[self.X_original].tolist()
        y_cord = self.df_plot[self.Y_original].tolist()
        self.contributing_points_plot = self.ax_2d.scatter(
            x_cord, y_cord, marker=".", color="red"
        )
        self.ax_2d.figure.canvas.draw()

    def plot_projection(self):
        self.df_plot["diff_y"] = self.df_plot["crossing_y"] - self.df_plot[self.Y]
        self.df_plot["diff_x"] = self.df_plot["crossing_x"] - self.df_plot[self.X]

        self.projection_plot = self.ax_2d.quiver(
            self.df_plot[self.X_original].tolist(),
            self.df_plot[self.Y_original].tolist(),
            [
                self.from_normalized_to_original_X(x)
                for x in self.df_plot["diff_x"].tolist()
            ],
            [
                self.from_normalized_to_original_Y(y)
                for y in self.df_plot["diff_y"].tolist()
            ],
            scale=1,
            angles="xy",
            scale_units="xy",
            width=0.001,
            headwidth=1,
            headlength=1,
        )
        self.ax_2d.figure.canvas.draw()

    def normalize_or_not(self, p):
        if self.normalize_axes:
            return self.from_original_to_normalized_point(p)
        return p

    def from_original_to_normalized_point(self, p):
        return self.from_original_to_normalized_X(
            p[0]
        ), self.from_original_to_normalized_Y(p[1])

    def from_normalized_to_original_point(self, p):
        return self.from_normalized_to_original_X(
            p[0]
        ), self.from_normalized_to_original_Y(p[1])

    def from_original_to_normalized_X(self, x):
        return (1 / self.aspect) * normalize(
            x, self.min_X_original, self.max_X_original
        )

    def from_original_to_normalized_Y(self, y):
        return self.aspect * normalize(y, self.min_Y_original, self.max_Y_original)

    def from_normalized_to_original_X(self, x):
        return self.aspect * inverse_normalize(
            x, self.min_X_original, self.max_X_original
        )

    def from_normalized_to_original_Y(self, y):
        return (1 / self.aspect) * inverse_normalize(
            y, self.min_Y_original, self.max_Y_original
        )


def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def inverse_normalize(x, x_min, x_max):
    return x * (x_max - x_min) + x_min


def get_line_from_two_points(point_1: tuple, point_2: tuple):
    a = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
    b = point_1[1] - a * point_1[0]
    return a, b


def range_of_column(col):
    return col.max() - col.min()


def get_dist_between_line_and_point(a: float, b: float, px: float, py: float):
    crossing_x, crossing_y = get_line_crossing(a, b, px, py)
    return (
        crossing_x,
        crossing_y,
        ((crossing_x - px) ** 2 + (crossing_y - py) ** 2) ** 0.5,
    )


def get_line_crossing(a: float, b: float, px: float, py: float):
    crossing_x = (px + a * py - a * b) / (1 + a**2)
    crossing_y = a * crossing_x + b

    return crossing_x, crossing_y


def remove_plot(the_plot) -> None:
    if the_plot is not None:
        the_plot.remove()
    return None
