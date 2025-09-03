import os
import random
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from tsp_genetic.utils import get_distance

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TSPGenetic:
    """Class representing the Traveling Salesman Problem (TSP) for a set of cities.
    Attributes:
        cities (GeoDataFrame): GeoDataFrame containing city information.
        num_cities (int): Number of cities in the problem.
        initial_population (List of GeoDataFrames): Initial population for the TSP.
    """

    def __init__(
        self,
        cities_gdf: gpd.GeoDataFrame,
        pop_size=10,
    ):
        self.cities = cities_gdf
        self.num_cities = len(cities_gdf)
        self.distance_matrix = self._compute_distance_matrix()
        self.coord_to_index = self._coord_to_index(cities_gdf)
        self.initial_population = self._initial_population(pop_size)
        self.current_population = self.initial_population

    def _coord_to_index(self, cities_gdf):
        coord_to_index = {}
        for idx, row in cities_gdf.iterrows():
            coord_key = (row["latitude"], row["longitude"])
            coord_to_index[coord_key] = idx
        return coord_to_index

    def _initial_population(self, size):
        """Generate a random initial population."""
        population = []
        for _ in range(size):  # Generate random solutions
            individual = self.cities.sample(n=self.num_cities).reset_index(
                drop=True
            )
            population.append((individual, self.fitness(individual)))
        return population

    def _compute_distance_matrix(self):
        """Pre-compute all pairwise distances between cities for performance."""
        n = len(self.cities)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                coords_a = (
                    self.cities.iloc[i]["latitude"],
                    self.cities.iloc[i]["longitude"],
                )
                coords_b = (
                    self.cities.iloc[j]["latitude"],
                    self.cities.iloc[j]["longitude"],
                )
                dist = get_distance(coords_a, coords_b)
                distances[i][j] = distances[j][i] = dist
        return distances

    def total_distance(self, individual):
        """Calculate the total distance of the TSP individual using pre-computed distance matrix."""
        if individual is None or len(individual) == 0:
            return float("inf")

        total_distance = 0.0
        # Get the original city indices in the order they appear in the individual
        individual_indices = []
        for i in range(len(individual)):
            city_row = individual.iloc[i]
            coord_key = (city_row["latitude"], city_row["longitude"])
            original_idx = self.coord_to_index[coord_key]
            individual_indices.append(original_idx)

        for i in range(len(individual_indices)):
            idx_a = individual_indices[i]
            idx_b = individual_indices[(i + 1) % len(individual_indices)]
            total_distance += float(self.distance_matrix[idx_a][idx_b])
        return total_distance

    def fitness(self, individual):
        """Calculate the fitness of the TSP individual."""
        # Fitness is inversely related to the total distance
        total_dist = self.total_distance(individual)
        if total_dist == 0:
            return float("inf")
        else:
            return 1 / total_dist

    def swap(self, individual):
        """Optimized simple swap with reduced DataFrame operations."""
        new_individual = individual.copy()
        n = len(individual)
        idx1, idx2 = random.sample(range(n), 2)
        # Use more efficient iloc swapping
        temp = new_individual.iloc[idx1].copy()
        new_individual.iloc[idx1] = new_individual.iloc[idx2]
        new_individual.iloc[idx2] = temp
        return new_individual

    def inversion(self, individual):
        """Optimized inversion with reduced operations."""
        new_individual = individual.copy()
        n = len(individual)
        idx1, idx2 = sorted(random.sample(range(n), 2))
        # Reverse the slice more efficiently
        segment = new_individual.iloc[idx1:idx2].iloc[::-1]
        new_individual.iloc[idx1:idx2] = segment.values
        return new_individual

    def mutate(self, type, individual):
        if type == "swap":
            return self.swap(individual)
        elif type == "inversion":
            return self.inversion(individual)
        else:
            raise ValueError("Unknown disturbance type")

    def parent_selection(self, tournament_size):
        # TOURNAMENT SELECTION (sample without replacement, sort by fitness)
        tournament = random.sample(self.current_population, k=min(tournament_size, len(self.current_population)))
        # Each individual is a tuple (individual, fitness)
        tournament_sorted = sorted(
            tournament, key=lambda x: x[1], reverse=True
        )
        return tournament_sorted

    def crossover(self, parent1, parent2):
        # Order 1 Crossover for DataFrames
        # parent1 and parent2 are tuples: (individual_df, fitness)
        p1_df = parent1[0].copy().reset_index(drop=True)
        p2_df = parent2[0].copy().reset_index(drop=True)
        point = random.randint(1, self.num_cities - 1)

        def df_contains(df, row):
            return any((df['latitude'] == row['latitude']) & (df['longitude'] == row['longitude']))

        child1 = p1_df.iloc[:point].copy()
        for idx, row in p2_df.iterrows():
            if not df_contains(child1, row):
                child1 = pd.concat(
                    [child1, row.to_frame().T], ignore_index=True
                )

        child2 = p2_df.iloc[:point].copy()
        for idx, row in p1_df.iterrows():
            if not df_contains(child2, row):
                child2 = pd.concat(
                    [child2, row.to_frame().T], ignore_index=True
                )

        return child1, child2

    def run(
        self,
        num_generations=200,
        mutation_type="inversion",
        pop_size=50,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elitism_percentage=0.1,
    ):
        history = []
        for generation in range(num_generations):
            # Elitism
            num_elites = int(elitism_percentage * len(self.current_population))
            elites = sorted(
                self.current_population, key=lambda x: x[1], reverse=True
            )[:num_elites]
            new_population = elites

            # Crossover and mutation
            offspring_list = []
            for i in range(len(self.current_population) - 1):
                # Crossover check
                if random.random() < crossover_rate:
                    # Crossover
                    offspring1, offspring2 = self.crossover(
                        self.current_population[i],
                        self.current_population[i + 1],
                    )
                    if random.random() < mutation_rate:
                        offspring1 = self.mutate(mutation_type, offspring1)
                    if random.random() < mutation_rate:
                        offspring2 = self.mutate(mutation_type, offspring2)
                    offspring_list.append(
                        (offspring1, self.fitness(offspring1))
                    )
                    offspring_list.append(
                        (offspring2, self.fitness(offspring2))
                    )
                else:
                    # No crossover, just copy parents
                    offspring1 = self.current_population[i]
                    offspring2 = self.current_population[i + 1]
                    offspring_list.append(offspring1)
                    offspring_list.append(offspring2)

            new_population.extend(offspring_list)

            # Update population
            self.current_population = new_population.copy()[:pop_size]

            # Save best for each generation
            generation_best = max(self.current_population, key=lambda x: x[1])
            history.append(
                (generation, generation_best[0], generation_best[1])
            )
        history_best = max(history, key=lambda x: x[2])
        return history, history_best
