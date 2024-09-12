import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Loads data from the pickle file.

    Args:
      file_path: Path to the pickle file.

    Returns:
      The loaded data from the pickle file.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def prepare_tsne_data(data):
    """Prepares data for t-SNE visualization.

    Args:
      data: Dictionary containing syndrome IDs as keys, with nested subject and image embeddings.

    Returns:
      A tuple containing:
        - embeddings: NumPy array of image embeddings.
        - syndrome_ids: NumPy array of corresponding syndrome IDs.
    """
    embeddings = []
    syndrome_ids = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                embeddings.append(embedding)
                syndrome_ids.append(syndrome_id)
    return np.array(embeddings), np.array(syndrome_ids)

def plot_tsne(embeddings_tsne, syndrome_ids, output_path='tsne_plot.png'):
    """Plots the t-SNE and saves the image.

    Args:
      embeddings_tsne: 2D t-SNE embeddings.
      syndrome_ids: Syndrome labels.
      output_path: File path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1], hue=syndrome_ids, palette="tab10")
    plt.title('t-SNE Visualization of Embeddings')
    plt.legend(loc='best')
    plt.savefig(output_path)

if __name__ == '__main__':
    data = load_data('mini_gm_public_v0.1.p')
    embeddings, syndrome_ids = prepare_tsne_data(data)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    plot_tsne(embeddings_tsne, syndrome_ids)