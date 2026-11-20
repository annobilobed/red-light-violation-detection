"""Plane geometry helpers for the stop-line crossing test.

All coordinates are image pixels (x to the right, y downwards).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def bottom_centre(box: Box) -> Point:
    """Ground-contact anchor of a vehicle box: the middle of its bottom edge."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def line_length(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def signed_distance(p: Point, a: Point, b: Point) -> float:
    """Signed perpendicular distance from p to the infinite line a->b.

    Positive on the left of a->b in image coordinates (cross product > 0).
    """
    length = line_length(a, b)
    if length == 0:
        raise ValueError("stop line has zero length")
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return cross / length


def projection_param(p: Point, a: Point, b: Point) -> float:
    """Position of p's projection along segment a->b: 0 at a, 1 at b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    if denom == 0:
        raise ValueError("stop line has zero length")
    return ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom


def point_in_polygon(p: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting test. Points on the boundary may fall either way."""
    x, y = p
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def zero_crossing_fraction(d0: float, d1: float) -> float:
    """Fraction in [0, 1] between two samples where a linear signal hits zero."""
    if d0 == d1:
        return 1.0
    f = d0 / (d0 - d1)
    return min(1.0, max(0.0, f))


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def clamp_rect(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Clip an (x, y, w, h) rectangle to the frame. May return zero size."""
    x0 = max(0, min(int(x), width))
    y0 = max(0, min(int(y), height))
    x1 = max(0, min(int(x + w), width))
    y1 = max(0, min(int(y + h), height))
    return x0, y0, x1 - x0, y1 - y0


def as_points(values: Iterable[Iterable[float]]) -> list[Point]:
    return [(float(v[0]), float(v[1])) for v in values]
