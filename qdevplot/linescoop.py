import matplotlib.pyplot as plt
import numpy as np
from qcodes.dataset.plotting import plot_dataset
from qcodes.dataset.data_set import load_by_id
from qdevplot.linebuilder import LineBuilder
from qdevplot.plotfunctions import if_not_ax_make_ax


class LineScoop:
    def __init__(self, data, delta: float = 0.01):
        self.delta = delta
        _, (self.ax_2d, self.ax_line) = plt.subplots(
            1, 2, sharex=False, sharey=False, constrained_layout=True
        )
        axes, self.cbaxes = plot_dataset(data, axes=self.ax_2d)
        self.df = data.to_pandas_dataframe().reset_index()
        self.line = LineBuilder(
            self.ax_2d,
            "red",
            action=self.reset_counter_ax_plot_line_scoop,
            use_single_click=True,
        )
        self.df_plot = None

    def reset_counter_ax_plot_line_scoop(self):
        print(self.line.counter)
        if self.line.counter > 1:
            self.line.counter = 0
            self.ax_line.cla()
            self.plot_line_scoop_from_df_points()

    def plot_line_scoop_from_df_points(self, labels=None):
        print(f"p1 {self.p1}")
        print(f"p2 {self.p2}")
        a, b = get_line_from_two_points(self.p1, self.p2)
        self.ax_line = self.plot_line_scoop_from_df_line(
            a,
            b,
        )
        return self.ax_line

    @classmethod
    def from_run_id(cls, run_id: int, delta: float = 0.01) -> "LineScoop":
        data = load_by_id(run_id)
        return cls(data, delta)

    @property
    def p1(self):
        return (self.line.xs[0], self.line.ys[0])

    @property
    def p2(self):
        return (self.line.xs[1], self.line.ys[1])

    @property
    def xlim(self):
        return (min(self.p1[0], self.p2[0]), max(self.p1[0], self.p2[0]))

    @property
    def ylim(self):
        return (min(self.p1[1], self.p2[1]), max(self.p1[1], self.p2[1]))

    def range_col(self, col):
        return col.max() - col.min()

    def plot_line_scoop_from_df_line(
        self,
        a,
        b,
        labels=None,
    ):
        self.ax_line = if_not_ax_make_ax(self.ax_line)
        a_norm, self.df_plot = get_scoop_line(
            self.df, a, b, self.delta, xlim=self.xlim, ylim=self.ylim
        )
        col_names = list(self.df_plot.columns)
        print(type(self.df_plot))
        # dist_to_point = np.array(self.df_plot["distsign"].values)
        # dist_to_point = np.array(
        #     [
        #         [abs(x * int(x > 0)) * scale, abs(x * int(x <= 0) * scale)]
        #         for x in self.df_plot["distsign"].values
        #     ]
        # ).T
        # self.df_plot[col_names[0]]
        self.ax_line.plot(
            self.df_plot["crossing_x"].tolist(),
            self.df_plot[col_names[2]].tolist(),
            linestyle="--",
            marker="o",
            label="data",
        )
        # self.df_plot.plot(
        #    col_names[0],
        #    col_names[2],
        #    linestyle="--",
        #    marker="o",
        #    ax=self.ax_line,
        #    label="data",
        # )
        self.ax_line.figure.canvas.draw()
        print("Points in line-scoop")
        print("x-coordinates: ", self.df_plot[col_names[0]].tolist())
        print("y-coordinates: ", self.df_plot[col_names[1]].tolist())

        # self.ax_line.quiver(
        #    self.df_plot[col_names[0]].tolist(),
        #    self.df_plot[col_names[2]].tolist(),
        #    dist_to_point
        #    * np.sin(-np.arctan(a_norm))
        #    * range_of_column(self.df_plot[col_names[0]]),
        #    dist_to_point
        #    * np.cos(np.arctan(a_norm))
        #    * range_of_column(self.df_plot[col_names[2]]),
        #    color=["r"],
        #    label="reference",
        #    angles="xy",
        #    scale_units="xy",
        #    scale=1,
        # )

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


def get_line_from_two_points(point_1: tuple, point_2: tuple):
    a = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
    b = point_1[1] - a * point_1[0]
    return a, b


def range_of_column(col):
    return col.max() - col.min()


def get_scoop_line_old(
    df, a, b, delta, xlim=(-np.inf, np.inf), ylim=(-np.inf, np.inf), scale=None
):
    col_names = list(df.columns)
    X = col_names[0]
    Y = col_names[1]
    Z = col_names[2]

    if scale == None:
        a_norm = (
            a * range_of_column(df[col_names[0]]) / range_of_column(df[col_names[1]])
        )

    df["distsign"] = (
        (df[X] * a + b - df[Y])
        / range_of_column(df[col_names[1]])
        * a_norm
        / (a_norm**2 + 1) ** 0.5
    )
    df["dist"] = df["distsign"].abs()
    df["line"] = df[X] * a + b
    print(f"xlim {xlim}")
    print(f"ylim {ylim}")

    return (
        a_norm,
        df[
            (df["dist"] < delta)
            * df[X].between(*xlim, "both")
            * df[Y].between(*ylim, "both")
        ].sort_values([X, Y])[[X, Y, Z, "distsign"]],
    )


def get_scoop_line(
    df, a, b, delta, xlim=(-np.inf, np.inf), ylim=(-np.inf, np.inf), scale=None
):
    col_names = list(df.columns)
    X = col_names[0]
    Y = col_names[1]
    Z = col_names[2]

    df = ad_dist_and_crossing(df, a, b)
    return (
        None,
        df[
            (df["dist"] < delta)
            * df["crossing_x"].between(*xlim, "both")
            * df["crossing_y"].between(*ylim, "both")
        ].sort_values(["crossing_x", "crossing_y"])[
            [X, Y, Z, "crossing_x", "crossing_y", "dist"]
        ],
    )


def ad_dist_and_crossing(df, a, b):
    col_names = list(df.columns)
    X_name = col_names[0]
    Y_name = col_names[1]
    temp = np.array(
        [
            *df.apply(
                lambda x: get_dist_between_line_and_point(a, b, x[X_name], x[Y_name]),
                axis=1,
            )
        ]
    )
    df["crossing_x"] = temp[:, 0]
    df["crossing_y"] = temp[:, 1]
    df["dist"] = temp[:, 2]
    return df


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
