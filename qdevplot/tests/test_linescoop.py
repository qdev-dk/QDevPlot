import numpy as np
import pandas as pd
import pytest
from qdevplot.linescoop import (
    get_line_from_two_points,
    NormalAspectTransform,
    LineScoop,
)


def test_get_line_from_two_points():
    point_1 = (0, 1)
    point_2 = (1, 3)
    expected_result = (2, 1)

    result = get_line_from_two_points(point_1, point_2)

    assert result == expected_result


class TestNormalAspectTransform:
    def test_from_original_to_normalized(self):
        col = np.array([1, 2, 3, 4, 5])
        aspect = 2
        transform = NormalAspectTransform(col, aspect)
        expected_result = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * 0.5

        result = transform.from_original_to_normalized(col)

        np.testing.assert_array_almost_equal(result, expected_result)

    def test_from_normalized_to_original(self):
        col_not_nomalized = [1, 2, 3, 4, 5]
        col = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * 0.5
        aspect = 2
        transform = NormalAspectTransform(col_not_nomalized, aspect)
        expected_result = np.array([1, 2, 3, 4, 5])

        result = transform.from_normalized_to_original(col)

        np.testing.assert_array_almost_equal(result, expected_result)

    def test_normalize(self):
        col = np.array([1, 2, 3, 4, 5])
        aspect = 2
        transform = NormalAspectTransform(col, aspect)
        expected_result = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        result = transform.normalize(col)

        np.testing.assert_array_almost_equal(result, expected_result)

    def test_inverse_normalize(self):
        col_not_nomalized = [1, 2, 3, 4, 5]
        col = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        aspect = 2
        transform = NormalAspectTransform(col_not_nomalized, aspect)
        expected_result = np.array([1, 2, 3, 4, 5])

        result = transform.inverse_normalize(col)

        np.testing.assert_array_almost_equal(result, expected_result)


# Replace "your_module" with the actual module/file where the LineScoop class is defined.


@pytest.fixture
def sample_dataframe():
    # Create a sample DataFrame for testing
    data = {"X": [1, 2, 3, 4, 5], "Y": [2, 4, 1, 6, 3], "Z": [10, 20, 30, 40, 50]}
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframedist():
    # Create a sample DataFrame for testing
    data = {
        "X": [0, 2, 4, 6, 0, 2, 4, 6, 9, 10],
        "Y": [5, 6, 7, 8, 5, 4, 3, 1, 0, 10],
        "Z": [10, 20, 30, 40, 50, 10, 20, 30, 40, 45],
    }
    return pd.DataFrame(data)


def test_from_original_to_normalized_point(sample_dataframe):
    # Test the from_original_to_normalized_point method

    # Create an instance of LineScoop
    line_scoop = LineScoop(sample_dataframe, delta=0.1)

    # Normalize the axes
    line_scoop.normalize_axes = True

    # Set a sample point in the original coordinate system
    original_point = (3, 3.5)

    # Get the corresponding normalized point
    normalized_point = line_scoop.from_original_to_normalized_point(original_point)

    # Expected normalized point based on normalization transformation
    expected_normalized_point = (0.5, 0.5)

    # Perform assertions to check if the results are as expected
    assert normalized_point == pytest.approx(expected_normalized_point)


def test_from_normalized_to_original_point(sample_dataframe):
    # Test the from_normalized_to_original_point method

    # Create an instance of LineScoop
    line_scoop = LineScoop(sample_dataframe, delta=0.1)

    # Normalize the axes
    line_scoop.normalize_axes = True

    # Set a sample normalized point
    normalized_point = (0.5, 0.5)

    # Get the corresponding point in the original coordinate system
    original_point = line_scoop.from_normalized_to_original_point(normalized_point)

    # Expected original point based on the reverse normalization transformation
    expected_original_point = (3, 3.5)

    # Perform assertions to check if the results are as expected
    assert original_point == pytest.approx(expected_original_point)


def test_line_scoop_from_points_dist(sample_dataframedist):
    line_scoop = LineScoop(sample_dataframedist, delta=0.099)

    line_scoop.p1 = (0.0, 5)
    line_scoop.p2 = (10, 5.0)

    # Get the scoop line
    x_values, z_values, a, b, columns = line_scoop.line_scoop_from_points()

    assert len(x_values) == 2
    assert len(z_values) == 2

    line_scoop.delta = 0.101
    x_values, z_values, a, b, columns = line_scoop.line_scoop_from_points()

    assert len(x_values) == 4
    assert len(z_values) == 4

    line_scoop.delta = 0.19
    x_values, z_values, a, b, columns = line_scoop.line_scoop_from_points()

    assert len(x_values) == 4
    assert len(z_values) == 4

    line_scoop.delta = 0.201
    x_values, z_values, a, b, columns = line_scoop.line_scoop_from_points()

    assert len(x_values) == 6
    assert len(z_values) == 6

    line_scoop.delta = 0.301
    x_values, z_values, a, b, columns = line_scoop.line_scoop_from_points()

    assert len(x_values) == 7
    assert len(z_values) == 7


def test_line_scoop_from_points(sample_dataframe):
    line_scoop = LineScoop(sample_dataframe, delta=0.1)
    line_scoop.p1 = (1.0, 3.5)
    line_scoop.p2 = (3.0, 6.0)

    x_values, z_values, a, b, columns = line_scoop.line_scoop_from_points()

    assert a == pytest.approx(1)
    assert b == pytest.approx(0.5)
    assert columns == [
        "X",
        "Y",
        "Z",
        "X_normalized",
        "Y_normalized",
        "crossing_x",
        "crossing_y",
        "dist",
    ]  # Expected column names in the DataFrame


def test_contributing_points(sample_dataframedist):
    # Test the contributing_points method

    # Create an instance of LineScoop
    line_scoop = LineScoop(sample_dataframedist, delta=0.11)

    # Set two points to create a line segment
    line_scoop.p1 = (0.0, 5)
    line_scoop.p2 = (10, 5.0)

    # Get the contributing points
    x_values, z_values, a, b, columns = line_scoop.line_scoop_from_points()
    x_values, y_values = line_scoop.contributing_points()

    # Perform assertions to check if the results are as expected
    assert len(x_values) == 4  # Expected number of contributing points along the X-axis
    assert len(y_values) == 4  # Expected number of contributing points along the Y-axis

    assert x_values == [0, 0, 2, 2]
    assert y_values == [5, 5, 6, 4]
