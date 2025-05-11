import pandas as pd
import os
import matplotlib.pyplot as plt

def extract_persistence_pairs(persistence_pairs,intervals, max_dim=2):
    """
    Extract persistence pairs and return them as a pandas DataFrame.

    Parameters:
    -----------
    simplex_tree : gudhi.SimplexTree
        A GUDHI simplex tree with computed persistence.
    max_dim : int, optional
        Maximum homology dimension to consider (default: 2).

    Returns:
    --------
    persistence_df : pd.DataFrame
        DataFrame containing persistence pairs with the following columns:
        - Dimension
        - Birth
        - Death
        - Birth Simplex
        - Death Simplex (if applicable)
    """
    # Ensure persistence has been computed

    persistence_data = []
    for dim in range(max_dim + 1):
         
        for i, (birth, death) in enumerate(intervals):
            # Retrieve associated simplices if possible
            
            birth_simplex, death_simplex = persistence_pairs[i]
            death = death if death != float('inf') else None  # Use None for infinite death times
            persistence_data.append({
                "Dimension": dim,
                "Birth": birth,
                "Death": death,
                "Birth Simplex": list(birth_simplex),
                "Death Simplex": list(death_simplex) if death_simplex else None
            })

    # Create DataFrame from the collected data
    persistence_df = pd.DataFrame(persistence_data)
    return persistence_df


import os
import matplotlib.pyplot as plt

def save_persistence_diagram(persistence_df, lSet, output_image_dir, episode):
    """
    Generate and save a persistence diagram for a single episode, highlighting lSet points.

    Parameters:
    -----------
    persistence_df : pd.DataFrame
        DataFrame containing persistence pairs with 'Birth', 'Death', and 'Birth Simplex' columns.
    lSet : np.ndarray
        Array of indices representing the lSet for the episode.
    output_image_dir : str
        Directory where the output image will be saved.
    episode : str
        Name of the episode.

    Returns:
    --------
    None
    """
    print(episode)

    # Match lSet with 'Birth Simplex'
    persistence_df['Selected'] = persistence_df['Birth Simplex'].apply(
        lambda x: any(elem in lSet for elem in eval(x))
    )

    # Extract birth and death values
    birth = persistence_df['Birth']
    death = persistence_df['Death']

    # Create a boolean array for lSet points (indices from lSet matched with Birth Simplex)
    lSet_highlight = persistence_df['Selected']

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(birth[~lSet_highlight], death[~lSet_highlight], label='Not in lSet', alpha=0.7)
    plt.scatter(birth[lSet_highlight], death[lSet_highlight], color='orange', label='lSet Points', alpha=0.9)

    # Add the diagonal line for reference
    diagonal = [min(0, 0), max(1, 1)]
    plt.plot(diagonal, diagonal, 'k--', label='Diagonal')

    # Add labels and legend
    plt.title(f"Persistence Diagram for {episode} (Highlighting lSet Points)")
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.legend()
    plt.grid(alpha=0.5)
    
    # Set axis limits to remove negative values
    plt.xlim(-0.1, birth.max() + 0.1)  # Ensure the range starts at 0 and includes all points
    plt.ylim(-0.1, death.max() + 0.1)  # Ensure the range starts at 0 and includes all points

    # Construct output image path with episode name
    output_image_path = os.path.join(output_image_dir, f"persistence_diagram_{episode}.png")

    # Save the plot as an image
    plt.savefig(output_image_path)
    plt.close()

    print(f"Image saved to {output_image_path}")

# Example usage:
# save_persistence_diagram(persistence_df, lSet, "./output_images", "episode1")


