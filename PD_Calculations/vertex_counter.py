import pandas as pd
import ast
from collections import Counter

def compute_death_simplex_vertex_counter(csv_path, dimension=None):
    """
    Reads a persistence‐diagram CSV and counts how often each vertex
    appears in the Death Simplex column.

    Parameters
    ----------
    csv_path : str
        Path to your CSV file.
    dimension : int or None
        If specified, only rows with Dimension == dimension are used.
        Otherwise, all rows are included.

    Returns
    -------
    Counter
        A Counter mapping vertex → frequency.
    """
    # 1. Load the CSV
    df = pd.read_csv(csv_path)

    # 2. (Optional) filter by homology dimension
    if dimension is not None:
        df = df[df['Dimension'] == dimension]

    # 3. Initialize counter
    vertex_counter = Counter()

    # 4. Iterate death‐simplex strings, parse them, and update the counter
    for simplex_str in df['Death Simplex']:
        # parse string like "[40653, 10327]" into a Python list
        vertices = ast.literal_eval(simplex_str)
        vertex_counter.update(vertices)

    return vertex_counter

# Example usage:
if __name__ == "__main__":
    vc = compute_death_simplex_vertex_counter("/vast/s219110279/PD_RAWDATA/tinyimagenet_pickle_pairs_RAWDATA.csv", dimension=0)
    # Print the 10 most common vertices
    # Save to .npy if you like:
    import numpy as np
    np.save("TinyImageNet_vertex_counter.npy", vc)
