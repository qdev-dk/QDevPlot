from turtle import width
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from qcodes.dataset.plotting import plot_dataset
from qcodes.dataset.data_set import load_by_id
from qdevplot.linebuilder import LineBuilder
from qdevplot.linemover import LineMover
from qdevplot.plotfunctions import if_not_ax_make_ax
import polars as pl


class LineScoop:
    def __init__(
        self,
        df,
        two_d_plotter=None,
        delta: float = 0.01,
        normalize_axes=True,
        do_plot_projection=True,
        do_mark_contributing_points=True,
        aspect=1,
    ):
        self.delta = delta
        # self.do_plot_projection = do_plot_projection
        #self.do_mark_contributing_points = do_mark_contributing_points
        self.aspect = aspect**0.5
        _, (self.ax_2d, self.ax_line) = plt.subplots(
            1, 2, sharex=False, sharey=False, constrained_layout=True
        )

        self.df = df

        self.normalize_axes = normalize_axes
        col_names = list(self.df.columns)
        self.X = col_names[0]
        self.Y = col_names[1]
        self.Z = col_names[2]
        self.X_original = col_names[0]
        self.Y_original = col_names[1]

        self.x_nat = NormalAspectTransform(self.df[self.X], self.aspect)
        self.y_nat = NormalAspectTransform(self.df[self.Y], 1 / self.aspect)

        self.df[f"{self.X}_normalized"] = self.df[self.X_original].apply(
            self.x_nat.from_original_to_normalized
        )

        self.df[f"{self.Y}_normalized"] = self.df[self.Y_original].apply(
            self.y_nat.from_original_to_normalized
        )

        if self.normalize_axes:
            self.X = f"{self.X}_normalized"
            self.Y = f"{self.Y}_normalized"

        self.normalized_aspect = self.x_nat.span / self.y_nat.span
        self.ax_2d.set_aspect(self.aspect**2 * self.normalized_aspect)

        # self.contributing_points_plot = None
        # self.projection_plot = None
        self.df_plot = None

        if two_d_plotter is None:
            two_d_plotter = self.df_to_scatter
        _, self.cbaxes = two_d_plotter(axes=self.ax_2d)

        self.line = LineMover(  # LineBuilder(
            self.ax_2d,
            linescoop=self,
            # "red",
            action=self.reset_counter_ax_plot_line_scoop,
            use_single_click=True,
        )

    @classmethod
    def from_qcodes_data(cls, data, **kwargs) -> "LineScoop":
        df = data.to_pandas_dataframe().reset_index()
        two_d_plotter = partial(plot_dataset, data)
        return cls(df=df, two_d_plotter=two_d_plotter, **kwargs)

    @classmethod
    def from_qcodes_run_id(cls, run_id, **kwargs) -> "LineScoop":
        data = load_by_id(run_id)
        return cls.from_qcodes_data(data, **kwargs)

    def reset_counter_ax_plot_line_scoop(self):
        # if self.line.counter <= 1:
        #   return

        # self.line.counter = 0
        self.ax_line.cla()
        print("action")
        #self.contributing_points_plot = remove_plot(self.contributing_points_plot)
        # self.projection_plot = remove_plot(self.projection_plot)
        self.plot_line_scoop_from_df_points()
        print("action2")

    def plot_line_scoop_from_df_points(self, labels=None):
        print(f"p1 {self.p1}")
        print(f"p2 {self.p2}")
        a, b = get_line_from_two_points(self.p1, self.p2)
        self.ax_line = self.plot_line_scoop_from_df_line(
            a,
            b,
        )
        # if self.do_plot_projection:
        #    self.plot_projection()
        # if self.do_mark_contributing_points:
        #   self.mark_contributing_points()
        return self.ax_line

    def plot_line_scoop_from_df_line(
        self,
        a,
        b,
        labels=None,
    ):
        self.ax_line = if_not_ax_make_ax(self.ax_line)
        self.df_plot = self.get_scoop_line_pl(
            a,
            b,
            # self.delta,
            # xlim=self.xlim,
            # ylim=self.ylim,
        )
        col_names = list(self.df_plot.columns)
        self.ax_line.plot(
            [
                self.x_nat.from_normalized_to_original(x)
                for x in self.df_plot["crossing_x"].tolist()
            ],
            self.df_plot[self.Z].tolist(),
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
        self, axes=None, x_label=None, y_label=None, cb_label=None, **kwargs
    ):
        x = self.df[self.X_original].values
        y = self.df[self.Y_original].values
        z = self.df[self.Z].values

        axes = if_not_ax_make_ax(axes)
        # ax = style_jose(ax)

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

    def contributing_points(self):
        return self.df_plot[self.X_original].tolist(), self.df_plot[self.Y_original].tolist()


    def projection(self):
        self.df_plot["diff_y"] = self.df_plot["crossing_y"] - self.df_plot[self.Y]
        self.df_plot["diff_x"] = self.df_plot["crossing_x"] - self.df_plot[self.X]

        return (
            self.df_plot[self.X_original].tolist(),
            self.df_plot[self.Y_original].tolist(),
            [
                self.x_nat.from_normalized_to_original(x) - self.x_nat.min
                for x in self.df_plot["diff_x"].tolist()
            ],
            [
                self.y_nat.from_normalized_to_original(y) - self.y_nat.min
                for y in self.df_plot["diff_y"].tolist()
            ],
        )

    def normalize_or_not(self, p):
        if self.normalize_axes:
            return self.from_original_to_normalized_point(p)
        return p

    def from_original_to_normalized_point(self, p):
        return self.x_nat.from_original_to_normalized(
            p[0]
        ), self.y_nat.from_original_to_normalized(p[1])

    def from_normalized_to_original_point(self, p):
        return self.x_nat.from_normalized_to_original(
            p[0]
        ), self.y_nat.from_normalized_to_original(p[1])

    def get_scoop_line_pl(
        self,
        a,
        b,
    ):
        df_pl = pl.from_pandas(self.df)
        df_pl = df_pl.with_columns(
            ((pl.col(self.X) + a * pl.col(self.Y) - a * b) / (1 + a**2)).alias(
                "crossing_x"
            )
        )

        df_pl = df_pl.with_columns(((a * pl.col("crossing_x") + b)).alias("crossing_y"))
        df_pl = df_pl.with_columns(
            (
                (
                    (pl.col("crossing_x") - pl.col(self.X)) ** 2
                    + (pl.col("crossing_y") - pl.col(self.Y)) ** 2
                )
                ** 0.5
            ).alias("dist")
        )
        return (
            df_pl.filter(
                (pl.col("dist") < self.delta)
                & (pl.col("crossing_x").is_between(*self.xlim))
                & (pl.col("crossing_y").is_between(*self.ylim))
            )
            .sort([pl.col("crossing_x"), pl.col("crossing_y")])
            .to_pandas()
        )


class NormalAspectTransform:
    def __init__(self, col, aspect):
        self.aspect = aspect
        self.min = col.min()
        self.max = col.max()
        self.span = self.max - self.min

    def from_original_to_normalized(self, x):
        return (1 / self.aspect) * self.normalize(x)

    def from_normalized_to_original(self, x):
        return self.aspect * self.inverse_normalize(x)

    def normalize(self, x):
        return (x - self.min) / self.span

    def inverse_normalize(self, x):
        return x * self.span + self.min


def get_line_from_two_points(point_1: tuple, point_2: tuple):
    a = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
    b = point_1[1] - a * point_1[0]
    return a, b


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
