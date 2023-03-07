import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import matplotlib.pylab as pylab
from typing import Optional
from functools import partial
from matplotlib.ticker import FuncFormatter

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
