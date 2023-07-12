from qdevplot.linescoop import LineScoop
from qdevplot.linemover import LineMover
from functools import partial
from qcodes.dataset.plotting import plot_dataset
from qcodes.dataset.data_set import load_by_id


class LinePlot:
    def __init__(
        self,
        df,
        two_d_plotter=None,
        delta: float = 0.01,
        normalize_axes=True,
        aspect=1,
    ):
        self.data = LineScoop(df, delta, normalize_axes=normalize_axes, aspect=aspect)

        self.linemover = LineMover(  # LineBuilder(
            two_d_plotter=two_d_plotter,
            linescoop=self.data,
            use_single_click=True,
        )

    @classmethod
    def from_qcodes_data(cls, data, **kwargs) -> "LinePlot":
        df = data.to_pandas_dataframe().reset_index()
        two_d_plotter = partial(plot_dataset, data)
        return cls(df=df, two_d_plotter=two_d_plotter, **kwargs)

    @classmethod
    def from_qcodes_run_id(cls, run_id, **kwargs) -> "LinePlot":
        data = load_by_id(run_id)
        return cls.from_qcodes_data(data, **kwargs)
