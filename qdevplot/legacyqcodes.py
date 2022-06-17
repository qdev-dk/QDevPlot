import numpy as np
import pandas as pd
import qdevplot.plotfunctions as qp
from matplotlib import pyplot as plt


def legacy_vna_plot(filepath, **kwargs):
    ax, cb = legacy_qcodes_plot(filepath, **kwargs)
    qp.change_unit_axis(axes=ax, axis='Y', scale=1e9, newlabel='SPEC frequncy[GHz]')
    qp.change_unit_axis(axes=ax, axis='X', scale=1, newlabel='Power[dBm]')
    qp.change_unit_axis(axes=cb, axis='Z', scale=1, newlabel='SPEC Magnitude[dBm]')

    return ax, cb

def plot_legacy_qcodes_linescoop(filepath, ivallist=None, x_label=None, y_label=None,
                           cb_label=None, invertaxes=False):
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False, constrained_layout=True)
    
    df, dim = read_legacy_qcodes(filepath, invertaxes=invertaxes)

    ax1, cb = df_to_pcolor(df, dims=dim, ax=ax1, invertaxes=invertaxes) # qp.df_to_scatter(df, ax=ax1)
    xmin = df.iloc[:, 0].min()
    xmax = df.iloc[:, 0].max()
    legend_list=[]
    for ival in ivallist:
        legend_list.append(str(ival))
        ax1 = qp.add_line_to_plot(ax1,(xmin,xmax), ival[0])
        ax1 = qp.add_line_to_plot(ax1,(xmin,xmax), ival[1])
        ax2 = qp.plot_line_scoop_from_df(df, ival, ax=ax2)
    ax2.legend(legend_list)
    ax1.set_xlabel('')
    return ax1, cb, ax2


def legacy_qcodes_plot(filepath, **kwargs):
    df, dims = read_legacy_qcodes(filepath)
    ax, cb = df_to_pcolor(df, dims, **kwargs)
    return ax, cb


def read_legacy_qcodes(filepath, invertaxes: bool = False):
    df = pd.read_csv(filepath, delimiter='\t', header=1)
    df.rename(columns={df.columns[0]: df.columns[0].split('"')[1::2][0]},
              inplace=True)
    datashape = (np.int64(df.iloc[0][0][1:]), df.iloc[0][1])
    df.drop(index=df.index[0], axis=0, inplace=True)
    if invertaxes:
        column_names = [df.columns[1], df.columns[0], df.columns[2]]
        df = df.reindex(columns=column_names)
    return df.astype('float64'), datashape


def df_to_pcolor(df, dims,
                 x_label=None, y_label=None, cb_label=None,
                 ax=None,
                 invertaxes: bool = False,
                 **kwargs):
    ax = qp.if_not_ax_make_ax(ax)
    ax = qp.style_jose(ax)

    col_names = list(df.columns)

    if x_label is None:
        x_label = col_names[0]
    if y_label is None:
        y_label = col_names[1]
    if cb_label is None:
        cb_label = col_names[2]

    X, Y, Z = shaped_data_from_df(df, dims, invertaxes=invertaxes)
    mappable = ax.pcolor(X, Y, Z, **kwargs)
    cb = ax.figure.colorbar(mappable, ax=ax)
    cb.set_label(cb_label)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    return ax, cb


def shaped_data_from_df(df, dims, invertaxes: bool = False):
    data = df.to_numpy()
    if invertaxes:
        X, Y = np.meshgrid(data[:, 0][:dims[1]], data[:, 1][::dims[1]])
        Z = data[:, 2].reshape(dims)
    else:
        X, Y = np.meshgrid(data[:, 0][::dims[1]], data[:, 1][:dims[1]])
        Z = data[:, 2].reshape(dims).T

    return X.astype(np.float64), Y.astype(np.float64), Z.astype(np.float64)
