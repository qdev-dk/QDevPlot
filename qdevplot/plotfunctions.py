import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pylab as pylab
from scipy import interpolate
from typing import Optional
from functools import partial
from matplotlib.ticker import FuncFormatter
from qcodes.dataset.plotting import plot_dataset
from qdevplot.linebuilder import LineBuilder


params = {
    "legend.fontsize": 16,
    "figure.figsize": (4, 4),
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
}
pylab.rcParams.update(params)


def plot_scatter_linescoop(
    x,
    y,
    z,
    xmeas=None,
    ymeas=None,
    ivallist=None,
    x_label=None,
    y_label=None,
    cb_label=None,
):
    _, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, sharey=False, constrained_layout=True
    )

    df = data_array_to_sorted_cordinate_df(
        x,
        y,
        z,
        xnew=xmeas,
        ynew=ymeas,
        x_label=x_label,
        y_label=y_label,
        z_label=cb_label,
    )
    ax1, cb = df_to_scatter(df, ax=ax1)
    xmin = df.iloc[:, 0].min()
    xmax = df.iloc[:, 0].max()
    legend_list = []
    for ival in ivallist:
        legend_list.append(str(ival))
        ax1 = add_line_to_plot(ax1, (xmin, xmax), ival[0])
        ax1 = add_line_to_plot(ax1, (xmin, xmax), ival[1])
        ax2 = plot_line_scoop_from_df(df, ival, ax=ax2)
    ax2.legend(legend_list)
    ax1.set_xlabel("")
    return ax1, cb, ax2


def plot_scatter_linecut(
    x,
    y,
    z,
    xmeas=None,
    ymeas=None,
    ylist=None,
    x_label=None,
    y_label=None,
    cb_label=None,
):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, sharey=False, constrained_layout=True
    )

    df = data_array_to_sorted_cordinate_df(
        x,
        y,
        z,
        xnew=xmeas,
        ynew=ymeas,
        x_label=x_label,
        y_label=y_label,
        z_label=cb_label,
    )
    ax1, cb = df_to_scatter(df, ax=ax1)
    xmin = df.iloc[:, 0].min()
    xmax = df.iloc[:, 0].max()
    for yval in ylist:
        ax1 = add_line_to_plot(ax1, (xmin, xmax), yval)

    ax2 = linecuts(df, ylist, ax=ax2)
    ax1.set_xlabel("")
    return ax1, cb, ax2


def add_line_to_plot(ax, x, y):
    ax.plot(x, (y, y), "--", color="r")
    return ax


def plot_scatter(
    x, y, z, xmeas=None, ymeas=None, x_label=None, y_label=None, cb_label=None
):
    df = data_array_to_sorted_cordinate_df(
        x,
        y,
        z,
        xnew=xmeas,
        ynew=ymeas,
        x_label=x_label,
        y_label=y_label,
        z_label=cb_label,
    )
    ax, cb = df_to_scatter(df)
    return ax, cb


def df_to_scatter(df, x_label=None, y_label=None, cb_label=None, ax=None, **kwargs):

    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    z = df.iloc[:, 2].values

    ax = if_not_ax_make_ax(ax)
    ax = style_jose(ax)

    col_names = list(df.columns)

    if x_label is None:
        x_label = col_names[0]
    if y_label is None:
        y_label = col_names[1]
    if cb_label is None:
        cb_label = col_names[2]

    mappable = ax.scatter(
        x=x, y=y, c=z, cmap="RdBu_r", **kwargs  # vmin=0.00, vmax=4.0,
    )
    cb = ax.figure.colorbar(mappable, ax=ax)
    cb.set_label(cb_label)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    return ax, cb


def plot_line_cuts(
    x,
    y,
    z,
    xmeas=None,
    ymeas=None,
    ylist=None,
    x_label=None,
    y_label=None,
    cb_label=None,
    ax=None,
):
    df = data_array_to_sorted_cordinate_df(
        x,
        y,
        z,
        xnew=xmeas,
        ynew=ymeas,
        x_label=x_label,
        y_label=y_label,
        z_label=cb_label,
    )
    ax = linecuts(df, ylist, ax=ax)
    return ax


def linecuts(df, ylist, x_label=None, y_label=None, cb_label=None, ax=None):
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    z = df.iloc[:, 2].values

    col_names = list(df.columns)

    if x_label is None:
        x_label = col_names[0]
    if y_label is None:
        y_label = col_names[1]
    if cb_label is None:
        cb_label = col_names[2]
    interpN = interpolate.NearestNDInterpolator(list(zip(x, y)), z)

    ax = if_not_ax_make_ax(ax)
    for y_val in ylist:
        interp_z = interpN(x, y_val)
        (line,) = ax.plot(x, interp_z, linestyle="--", marker="o")
        line.set_label(f"{y_label}={y_val}")

    ax.legend()
    plt.ylabel(cb_label)
    plt.xlabel(x_label)
    plt.grid()
    # plt.ylim(0,4.5)
    return ax


def plot_line_scoop(
    x,
    y,
    z,
    xmeas=None,
    ymeas=None,
    ival=None,
    x_label=None,
    y_label=None,
    cb_label=None,
    ax=None,
):
    df = data_array_to_sorted_cordinate_df(
        x,
        y,
        z,
        xnew=xmeas,
        ynew=ymeas,
        x_label=x_label,
        y_label=y_label,
        z_label=cb_label,
    )
    ax = plot_line_scoop_from_df(df, ival, ax=ax)
    return ax


def plot_line_scoop_from_df(df, ival, ax=None):
    ax = if_not_ax_make_ax(ax)
    df_plot = line_scoop(df, ival)
    col_names = list(df_plot.columns)
    df_plot.plot(col_names[0], col_names[1], linestyle="--", marker="o", ax=ax)
    return ax


def line_scoop(df, ival):
    col_names = list(df.columns)
    df_plot = df[
        (df[col_names[1]] >= ival[0]) & (df[col_names[1]] <= ival[1])
    ].sort_values(col_names[0])
    return df_plot[[col_names[0], col_names[2]]]


def data_array_to_sorted_cordinate_df(
    x,
    y,
    z,
    xnew=None,
    ynew=None,
    x_label: str = "X",
    y_label: str = "Y",
    z_label: str = "Z",
):
    if x_label is None:
        x_label = "X"
    if y_label is None:
        y_label = "Y"
    if z_label is None:
        z_label = "Z"

    z_data = []
    for ix, vx in enumerate(x):
        for iy, vy in enumerate(y):
            if xnew is None:
                xij = vx
            else:
                xij = xnew[iy][ix]
            if ynew is None:
                yij = vy
            else:
                yij = ynew[iy][ix]
            z_data.append([xij, yij, z[iy][ix]])
    df = pd.DataFrame(z_data, columns=[x_label, y_label, z_label])
    df = df.dropna()
    df_s = df.sort_values([x_label, y_label])
    return df_s


def style_jose(ax):
    ax.set_axisbelow(True)
    ax.grid(which="major", color="black", linewidth=1.2)
    ax.grid(which="minor", color="black", linewidth=0.6)
    ax.set_facecolor("#708099")
    return ax


def if_not_ax_make_ax(ax):
    if ax is None:
        ax = plt.axes()
    return ax


def set_real_aspect(ax):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    ax.set_aspect(aspect)
    return ax


def change_unit_axis(axes, axis, scale: float, newlabel: Optional[str] = None):

    formatter = FuncFormatter(partial(unit_scale, scale=scale))

    if axis in ["x", "X"]:
        axes.xaxis.set_major_formatter(formatter)
        if newlabel is None:
            label = axes.get_xlabel()
            axes.set_xlabel(f"{label}{scale}")
        else:
            axes.set_xlabel(newlabel)

    elif axis in ["y", "Y"]:
        axes.yaxis.set_major_formatter(formatter)
        if newlabel is None:
            label = axes.get_ylabel()
            axes.set_ylabel(f"{label}{scale}")
        else:
            axes.set_ylabel(newlabel)

    elif axis in ["cb", "colorbar", "z", "Z"]:
        axes.formatter = formatter
        axes.update_ticks()
        if newlabel is None:
            label = axes.get_label()
            axes.set_label(f"{label}{scale}")
        else:
            axes.set_label(newlabel)

    return axes


def unit_scale(x, pos, scale: float):

    return "%1.1f" % (x / scale)


def line_cut_qcodes_data(data, p1, p2, delta):
    _, (ax1, ax2) = plt.subplots(
        1, 2, sharex=False, sharey=False, constrained_layout=True
    )
    df = data.to_pandas_dataframe().reset_index()
    plot_dataset(data, axes=ax1)
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color="r")
    col_names = list(df.columns)
    paramspecs = data.paramspecs
    labels = [
        f"{paramspecs[col_name].label} ({paramspecs[col_name].unit})"
        for col_name in col_names
    ]
    plot_line_scoop_from_df_points(df, p1, p2, delta, ax=ax2, labels=labels)


def plot_line_scoop_from_df_points(df, p1, p2, delta, ax=None, labels=None):
    a, b = get_line_from_two_points(p1, p2)
    ax = plot_line_scoop_from_df_line(
        df,
        a,
        b,
        delta,
        ax=ax,
        labels=labels,
        xlim=(min(p1[0], p2[0]), max(p1[0], p2[0])),
        ylim=(min(p1[1], p2[1]), max(p1[1], p2[1])),
    )
    return ax


def plot_line_scoop_from_df_line(
    df,
    a,
    b,
    delta,
    ax=None,
    labels=None,
    xlim=(-np.inf, np.inf),
    ylim=(-np.inf, np.inf),
):
    ax = if_not_ax_make_ax(ax)
    df_plot = get_scoop_line(df, a, b, delta, xlim=xlim, ylim=ylim)
    print(df_plot.head())
    dist_to_point = np.array(
        [
            [abs(x * int(x > 0)), abs(x * int(x <= 0))]
            for x in df_plot["distsign"].values
        ]
    ).T
    col_names = list(df_plot.columns)
    df_plot.plot(
        col_names[0],
        col_names[2],
        linestyle="--",
        marker="o",
        ax=ax,
        yerr=dist_to_point,
    )
    ax.legend(f"t({col_names[0]} + {a:.2f}{col_names[1]}) + {b:.2f}{col_names[1]}")
    ax.set_title(f"t({col_names[0]} + {a:.2f}{col_names[1]}) + {b:.2f}{col_names[1]}")
    if labels:
        ax.set_xlabel(f"t({labels[0]} + {a:.2f}{labels[1]}) + {b:.2f}{labels[1]}")
    else:
        ax.set_xlabel("t")
    return ax


def get_scoop_points(df, p1, p2, delta):
    a, b = get_line_from_two_points(p1, p2)
    return get_scoop_line(
        df,
        a,
        b,
        delta,
        xlim=(min(p1[0], p2[0]), max(p1[0], p2[0])),
        ylim=(min(p1[1], p2[1]), max(p1[1], p2[1])),
    )


def get_scoop_line(df, a, b, delta, xlim=(-np.inf, np.inf), ylim=(-np.inf, np.inf)):
    col_names = list(df.columns)
    X = col_names[0]
    Y = col_names[1]
    Z = col_names[2]

    df["distsign"] = (df[X] * a + b - df[Y]) / (a**2 + 1) ** 0.5
    df["dist"] = df["distsign"].abs()
    df["line"] = df[X] * a + b
    print(f"xlim {xlim}")
    print(f"ylim {ylim}")
    return df[
        (df["dist"] < delta)
        * df[X].between(*xlim, "both")
        * df[Y].between(*ylim, "both")
    ].sort_values([X, Y])[[X, Y, Z, "distsign"]]


def get_line_from_two_points(point_1: tuple, point_2: tuple):
    a = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
    b = point_1[1] - a * point_1[0]
    return a, b


def line_on_plot(my_data, delta=0.01) -> tuple:
    _, (ax, ax2) = plt.subplots(
        1, 2, sharex=False, sharey=False, constrained_layout=True
    )
    axes, cbaxes = plot_dataset(my_data, axes=ax)
    df = my_data.to_pandas_dataframe().reset_index()

    line = LineBuilder(
        axes[0],
        ax,
        "red",
        action=partial(reset_counter_ax_plot_line_scoop, df, ax2, delta),
        use_single_click=True,
    )
    plt.show()
    return (line.xs, line.ys, line.ramp)


def reset_counter_ax_plot_line_scoop(df, ax, delta, self):
    if self.counter > 1:
        self.counter = 0
        ax.cla()
        plot_line_scoop_from_df_points(
            df, (self.xs[0], self.ys[0]), (self.xs[1], self.ys[1]), delta, ax
        )


class LineScoop:
    def __init__(self, data, delta=0.01):
        self.delta = delta
        _, (self.ax_2d, self.ax_line) = plt.subplots(
            1, 2, sharex=False, sharey=False, constrained_layout=True
        )
        axes, self.cbaxes = plot_dataset(data, axes=self.ax_2d)
        self.df = data.to_pandas_dataframe().reset_index()
        self.line = LineBuilder(
            axes[0],
            self.ax_2d,
            "red",
            action=self.reset_counter_ax_plot_line_scoop,
            use_single_click=True,
        )
        self.df_plot = None

    def reset_counter_ax_plot_line_scoop(self):
        if self.line.counter > 1:
            self.line.counter = 0
            self.ax_line.cla()
            self.plot_line_scoop_from_df_points()

    def plot_line_scoop_from_df_points(self, labels=None):
        a, b = get_line_from_two_points(self.p1, self.p2)
        ax = self.plot_line_scoop_from_df_line(
            a,
            b,
        )
        return ax

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

    def plot_line_scoop_from_df_line(
        self,
        a,
        b,
        labels=None,
    ):
        self.ax_line = if_not_ax_make_ax(self.ax_line)
        self.df_plot = get_scoop_line(
            self.df, a, b, self.delta, xlim=self.xlim, ylim=self.ylim
        )
        dist_to_point = np.array(
            [
                [abs(x * int(x > 0)), abs(x * int(x <= 0))]
                for x in self.df_plot["distsign"].values
            ]
        ).T
        col_names = list(self.df_plot.columns)
        self.df_plot.plot(
            col_names[0],
            col_names[2],
            linestyle="--",
            marker="o",
            ax=self.ax_line,
            yerr=dist_to_point,
        )
        self.ax_line.legend(
            f"t({col_names[0]} + {a:.2f}{col_names[1]}) + {b:.2f}{col_names[1]}"
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
