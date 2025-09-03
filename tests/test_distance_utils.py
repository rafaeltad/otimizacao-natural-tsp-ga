import pytest
from tsp_genetic.utils.distance import get_distance

def test_get_distance_basic():
    coords1 = (0.0, 0.0)
    coords2 = (0.0, 1.0)
    dist = get_distance(coords1, coords2)
    assert dist > 0
    assert isinstance(dist, float)


def test_get_distance_same_point():
    coords = (10.0, 10.0)
    dist = get_distance(coords, coords)
    assert dist == 0


def test_get_distance_negative_coords():
    coords1 = (-10.0, -10.0)
    coords2 = (10.0, 10.0)
    dist = get_distance(coords1, coords2)
    assert dist > 0
