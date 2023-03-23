
import matplotlib.pyplot as plt
from qcodes.dataset.plotting import plot_dataset
from qdevplot.linebuilder import LineBuilder

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
