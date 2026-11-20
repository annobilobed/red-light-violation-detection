import pytest

from app.pipeline.geometry import (bottom_centre, iou, point_in_polygon, projection_param,
                                   signed_distance, zero_crossing_fraction, clamp_rect)


def test_bottom_centre():
    assert bottom_centre((10, 20, 30, 60)) == (20.0, 60.0)


def test_signed_distance_sides():
    a, b = (0, 0), (10, 0)
    assert signed_distance((5, 3), a, b) == pytest.approx(3)
    assert signed_distance((5, -3), a, b) == pytest.approx(-3)
    assert signed_distance((5, 0), a, b) == 0


def test_signed_distance_zero_length():
    with pytest.raises(ValueError):
        signed_distance((1, 1), (2, 2), (2, 2))


def test_projection_param():
    assert projection_param((5, 7), (0, 0), (10, 0)) == pytest.approx(0.5)
    assert projection_param((-5, 0), (0, 0), (10, 0)) == pytest.approx(-0.5)


def test_point_in_polygon():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq)
    assert not point_in_polygon((15, 5), sq)
    assert not point_in_polygon((5, 5), sq[:2])


def test_zero_crossing_fraction():
    assert zero_crossing_fraction(3, -1) == pytest.approx(0.75)
    assert zero_crossing_fraction(2, 0) == pytest.approx(1.0)


def test_iou_and_clamp():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert clamp_rect(-5, -5, 20, 20, 10, 10) == (0, 0, 10, 10)
