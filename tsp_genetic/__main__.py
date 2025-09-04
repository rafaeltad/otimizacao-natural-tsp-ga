import os
import typer
import mlflow
import logging
import geopandas as gpd

from geobr import read_state
from typing_extensions import Annotated
from tsp_genetic.models.tsp import TSPGenetic
from tsp_genetic.utils.plot import plot_tsp_solution
from tsp_genetic.utils.load import load_config, load_cities, load_brazil_map
from tsp_genetic.utils.gridsearch import generate_gridsearch


# Configure logging with custom format for better readability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger(__name__)


def main(config_path: Annotated[str, typer.Option()] = ".data/config.yaml"):

    # Load configuration
    config = load_config(config_path)
    params = config["params"]

    # Set up MLflow experiment
    mlflow.set_experiment(config.get("mlflow_experiment", "Default"))

    # Load Brazil map
    brazil = load_brazil_map()

    # Load cities from geobr
    cities_gdf = load_cities(
        config.get("cities_data_path", "data/cities.csv"),
        endpoint=config.get(
            "ibge_endpoint",
            "https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/2024/variaveis/9324?localidades=N6[all]",
        ),
    )
    # Make grid search
    param_combinations = generate_gridsearch(params)

    for i, param_run in enumerate(param_combinations):
        n_cities = param_run.get("n_cities", 10)
        if n_cities > len(cities_gdf):
            raise ValueError(
                f"Requested {n_cities} cities, but only {len(cities_gdf)} available."
            )
        cities_gdf = cities_gdf.sample(
            n=n_cities, random_state=42
        ).reset_index(drop=True)
        # Start MLflow run
        with mlflow.start_run():
            plot_path = plot_tsp_solution(
                cities_gdf.nome.tolist(),
                cities_gdf,
                brazil,
                title=f"All Cities (n={n_cities}). Initial State",
            )

            LOGGER.info(
                f"Running combination {i+1}/{len(param_combinations)}: {param_run}"
            )

            # Create TSP instance and run with current parameters
            tsp = TSPGenetic(
                cities_gdf, pop_size=param_run.get("pop_size", 10)
            )
            history, best = tsp.run(
                num_generations=param_run.get("num_generations", 20),
                pop_size=param_run.get("pop_size", 10),
                mutation_rate=param_run.get("mutation_rate", 0.2),
                crossover_rate=param_run.get("crossover_rate", 0.8),
                mutation_type=param_run.get("mutation_type", "inversion"),
            )
            LOGGER.info(
                f"Best fitness: {best[2]:.6f}, Distance: {tsp.total_distance(best[1]):.2f}"
            )

            # Log parameters run for this combination
            mlflow.log_param("param_run", i)

            # Log parameters for this combination
            for param_name, param_value in param_run.items():
                mlflow.log_param(f"{param_name}", param_value)

            # Log metrics
            mlflow.log_metric(f"best_fitness", best[2])
            mlflow.log_metric(f"best_distance", tsp.total_distance(best[1]))
            mlflow.log_metric(f"num_generations", len(history))

            # Plot and log best solution for this combination
            plot_path = plot_tsp_solution(
                best[1].nome.tolist(),
                cities_gdf,
                brazil,
                title=f"Distance: {tsp.total_distance(best[1]):.2f}",
            )


if __name__ == "__main__":
    typer.run(main)
