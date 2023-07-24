import polars as pl
import pandas as pd
import numpy as np


class LineScoop:
    def __init__(
        self,
        df,
        delta: float = 0.01,
        normalize_axes=True,
        aspect=1,
    ):
        self.delta = delta
        self.normalize_axes = normalize_axes
        self.aspect = aspect**0.5

        self.df = df

        col_names = list(self.df.columns)
        self.X = col_names[0]
        self.Y = col_names[1]
        self.Z = col_names[2]
        self.X_original = col_names[0]
        self.Y_original = col_names[1]

        self.x_nat = NormalAspectTransform(self.df[self.X], self.aspect)
        self.y_nat = NormalAspectTransform(self.df[self.Y], 1 / self.aspect)
        self.normalized_aspect = (self.x_nat.span / self.y_nat.span) * self.aspect**2

        self.p1 = (0.0, 0.0)
        self.p2 = (0.0, 0.0)

        self.df[f"{self.X}_normalized"] = self.df[self.X_original].apply(
            self.x_nat.from_original_to_normalized
        )

        self.df[f"{self.Y}_normalized"] = self.df[self.Y_original].apply(
            self.y_nat.from_original_to_normalized
        )

        if self.normalize_axes:
            self.X = f"{self.X}_normalized"
            self.Y = f"{self.Y}_normalized"

        self.df_plot = pd.DataFrame({self.X: []})

    def line_scoop_from_points(self):
        a, b = get_line_from_two_points(self.p1, self.p2)
        self.df_plot = self.get_scoop_line(
            a,
            b,
        )
        return (
            [
                self.x_nat.from_normalized_to_original(x)
                for x in self.df_plot["crossing_x"].tolist()
            ],
            self.df_plot[self.Z].tolist(),
            a,
            b,
            list(self.df_plot.columns),
        )

    def get_scoop_line(
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

    @property
    def p1(self):
        return self._p1

    @p1.setter
    def p1(self, value: tuple[float, float]):
        self._p1 = self.normalize_or_not(value)

    @property
    def p2(self):
        return self._p2

    @p2.setter
    def p2(self, value: tuple[float, float]):
        self._p2 = self.normalize_or_not(value)

    @property
    def xlim(self):
        return (min(self.p1[0], self.p2[0]), max(self.p1[0], self.p2[0]))

    @property
    def ylim(self):
        return (min(self.p1[1], self.p2[1]), max(self.p1[1], self.p2[1]))

    def contributing_points(self):
        return (
            self.df_plot[self.X_original].tolist(),
            self.df_plot[self.Y_original].tolist(),
        )

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
        return self.from_original_to_normalized_point(p) if self.normalize_axes else p

    def from_original_to_normalized_point(self, p):
        return self.x_nat.from_original_to_normalized(
            p[0]
        ), self.y_nat.from_original_to_normalized(p[1])

    def from_normalized_to_original_point(self, p):
        return self.x_nat.from_normalized_to_original(
            p[0]
        ), self.y_nat.from_normalized_to_original(p[1])


class NormalAspectTransform:
    def __init__(self, col, aspect):
        col = np.array(col)
        self.aspect = aspect
        self.min = col.min()
        self.max = col.max()
        self.span = self.max - self.min

    def from_original_to_normalized(self, x):
        return (1 / self.aspect) * self.normalize(x)

    def from_normalized_to_original(self, x):
        return  self.inverse_normalize(x*self.aspect)

    def normalize(self, x):
        return (x - self.min) / self.span

    def inverse_normalize(self, x):
        return x * self.span + self.min


def get_line_from_two_points(point_1: tuple, point_2: tuple):
    a = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
    b = point_1[1] - a * point_1[0]
    return a, b
