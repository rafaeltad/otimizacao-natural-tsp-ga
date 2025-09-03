import pytest
import geopandas as gpd
import pandas as pd
from tsp_genetic.models.tsp import TSPGenetic

# Minimal city data for testing
CITY_DATA = [
    {"latitude": 0.0, "longitude": 0.0},
    {"latitude": 0.0, "longitude": 1.0},
    {"latitude": 1.0, "longitude": 0.0},
]
CITIES_GDF = gpd.GeoDataFrame(CITY_DATA)


def test_tspgenetic_init():
    tsp = TSPGenetic(CITIES_GDF, pop_size=2)
    assert tsp.num_cities == 3
    assert len(tsp.initial_population) == 2
    assert tsp.distance_matrix.shape == (3, 3)


def test_total_distance_and_fitness():
    tsp = TSPGenetic(CITIES_GDF, pop_size=1)
    individual = tsp.initial_population[0][0]
    dist = tsp.total_distance(individual)
    assert dist > 0
    fit = tsp.fitness(individual)
    assert fit > 0


def test_swap_and_inversion():
    tsp = TSPGenetic(CITIES_GDF, pop_size=1)
    individual = tsp.initial_population[0][0]
    swapped = tsp.swap(individual)
    inverted = tsp.inversion(individual)
    assert len(swapped) == len(individual)
    assert len(inverted) == len(individual)


def test_mutate():
    tsp = TSPGenetic(CITIES_GDF, pop_size=1)
    individual = tsp.initial_population[0][0]
    swapped = tsp.mutate("swap", individual)
    inverted = tsp.mutate("inversion", individual)
    assert len(swapped) == len(individual)
    assert len(inverted) == len(individual)
    with pytest.raises(ValueError):
        tsp.mutate("unknown", individual)


def test_parent_selection():
    tsp = TSPGenetic(CITIES_GDF, pop_size=3)
    selected = tsp.parent_selection(2)
    assert len(selected) == 2
    assert selected[0][1] >= selected[1][1]


def test_crossover():
    tsp = TSPGenetic(CITIES_GDF, pop_size=2)
    parent1 = tsp.initial_population[0]
    parent2 = tsp.initial_population[1]
    child1, child2 = tsp.crossover(parent1, parent2)
    assert len(child1) == tsp.num_cities
    assert len(child2) == tsp.num_cities


def test_run():
    tsp = TSPGenetic(CITIES_GDF, pop_size=4)
    history, best = tsp.run(num_generations=5, pop_size=4)
    assert len(history) == 5
    assert best[2] > 0
