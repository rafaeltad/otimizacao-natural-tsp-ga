import mlflow
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

def plot_tsp_solution(
    individual,
    gdf,
    brazil,
    title="TSP Best Individual for Brazilian Cities",
    plot_path=None,
    solution_name="tsp_solution",
):
    """
    Plots the TSP solution on the Brazil map.

    Parameters:
        individual (list): List of indices or city names in the visiting order.
        gdf (GeoDataFrame): GeoDataFrame with city info (must include 'longitude' and 'latitude').
        brazil (GeoDataFrame): GeoDataFrame with Brazil state boundaries.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new one.
        title (str): Title for the plot.
        show_names_only (bool): If True, only show city names in visiting order.
        show_plot (bool): If True, displays the plot with plt.show().
    """

    # Prepare coordinates in visiting order
    if isinstance(individual[0], str):
        visited_gdf = gdf[gdf["nome"].isin(individual)]
        coords = (
            visited_gdf.set_index("nome")
            .loc[individual][["longitude", "latitude"]]
            .values
        )
        names = individual
    else:
        visited_gdf = gdf.iloc[individual]
        coords = visited_gdf[["longitude", "latitude"]].values
        names = visited_gdf["nome"].values

    # Optionally close the loop
    coords = np.vstack([coords, coords[0]])
    names = list(names) + [names[0]]

    fig, ax = plt.subplots(figsize=(12, 12))
    brazil.plot(ax=ax, color="white", edgecolor="black")
    visited_gdf.plot(ax=ax, color="red", markersize=30, zorder=2)

    # Plot the TSP path
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color="black",
        linewidth=2,
        zorder=3,
        label="TSP Path",
    )

    # Annotate cities
    for idx, (lon, lat) in enumerate(coords[:-1]):
        ax.text(lon, lat, names[idx], fontsize=9, ha="right", va="bottom")

    ax.set_title(title)
    ax.legend()
    # Save to file
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = os.path.join(
            tmpdir, f"{solution_name}_plot.png"
        )
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(plot_path, f"plots/")
    return plot_path
