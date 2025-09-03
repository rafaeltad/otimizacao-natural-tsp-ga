# TSP Genetic Algorithm

A minimal Python package for solving the Traveling Salesman Problem (TSP) using a genetic algorithm. Cities are represented as latitude/longitude pairs and distance is calculated geodesically.

## Features
- TSP solver using genetic algorithm
- Geodesic distance calculation
- Easily extensible for new mutation/crossover strategies
- Includes unit tests for models and utilities

## Installation

```bash
uv pip install -r requirements.txt
```
Or use `pyproject.toml` with your preferred Python environment manager.

## Usage

Run the main module:
```bash
uv run tsp_genetic
```
Or import and use the `TSPGenetic` class in your own scripts.

## Testing

Run all tests:
```bash
uv run -m pytest tests/
```
---

For more details, see the source code and tests in the `tsp_genetic` and `tests` directories.
