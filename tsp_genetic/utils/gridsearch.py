import itertools
from typing import Dict, List, Any


def generate_gridsearch(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all combinations from parameter grid.

    Parameters:
        params (dict): Dictionary where keys are parameter names and values are lists of values to try

    Returns:
        list: List of dictionaries, each containing one parameter combination

    Example:
        params = {
            'pop_size': [10, 20],
            'mutation_rate': [0.1, 0.2],
            'num_generations': [50, 100]
        }
        # Returns 8 combinations (2 * 2 * 2)
    """
    if not params:
        return [{}]

    # Get parameter names and their possible values
    param_names = list(params.keys())
    param_values = list(params.values())

    # Generate all combinations using itertools.product
    combinations = []
    for combination in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combination))
        combinations.append(param_dict)

    return combinations
