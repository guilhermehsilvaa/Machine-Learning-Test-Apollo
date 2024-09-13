import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

def prepare_data(data):
    """Prepares data for KNN and evaluation.

    Args:
      data: The data loaded from the pickle file.

    Returns:
      embeddings: Image embeddings.
      syndrome_ids: Syndrome IDs.
      label_encoder: LabelEncoder used to encode syndrome IDs.
    """
    embeddings = []
    syndrome_ids = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                embeddings.append(embedding)
                syndrome_ids.append(syndrome_id)
    embeddings = np.array(embeddings)
    syndrome_ids = np.array(syndrome_ids)

    # Encode syndrome IDs: transforms IDs from strings to labels
    le = LabelEncoder()
    encoded_syndrome_ids = le.fit_transform(syndrome_ids)

    # Normalize embeddings
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    return embeddings, encoded_syndrome_ids, le

def knn_classification(train_embeddings, train_labels, test_embeddings, n_neighbors=5, metric='cosine'):
    """Classifies test embeddings using KNN.

    Args:
      train_embeddings: Training embeddings.
      train_labels: Training labels.
      test_embeddings: Test embeddings.
      n_neighbors: Number of neighbors.
      metric: Distance metric ('cosine' or 'euclidean').

    Returns:
      KNN predictions.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    knn.fit(train_embeddings, train_labels)
    return knn.predict_proba(test_embeddings)

def evaluate_model(train_embeddings, train_labels, test_embeddings, test_labels, metric='cosine', n_neighbors=5):
    """Evaluates the KNN model using AUC.

    Args:
      train_embeddings: Training embeddings.
      train_labels: Training labels.
      test_embeddings: Test embeddings.
      test_labels: Test labels.
      metric: Distance metric ('cosine' or 'euclidean').
      n_neighbors: Number of neighbors.

    Returns:
      AUC score.
    """
    knn_predictions = knn_classification(train_embeddings, train_labels, test_embeddings, n_neighbors, metric)
    roc_auc = roc_auc_score(test_labels, knn_predictions, multi_class='ovr')
    return roc_auc

def cross_validation(embeddings, syndrome_ids, metric='cosine', n_neighbors=5, n_splits=10):
    """Performs cross-validation to evaluate the model.

    Args:
      embeddings: Image embeddings.
      syndrome_ids: Syndrome IDs.
      metric: Distance metric ('cosine' or 'euclidean').
      n_neighbors: Number of neighbors.
      n_splits: Number of folds for cross-validation.

    Returns:
      Mean AUC scores.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []

    for train_index, test_index in kf.split(embeddings):
        train_embeddings, test_embeddings = embeddings[train_index], embeddings[test_index]
        train_labels, test_labels = syndrome_ids[train_index], syndrome_ids[test_index]

        roc_auc = evaluate_model(train_embeddings, train_labels, test_embeddings, test_labels, metric, n_neighbors)
        auc_scores.append(roc_auc)

    return np.mean(auc_scores)

def plot_roc_auc(embeddings, syndrome_ids, n_splits=10, output_path='roc_auc_plot.png'):
    """Saves the ROC AUC plots for cosine and euclidean metrics, averaged across all folds.

    Args:
      embeddings: Image embeddings.
      syndrome_ids: Syndrome IDs.
      output_path: File path to save the plot.
      n_splits: Number of folds for cross-validation.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    tprs_cosine, tprs_euclidean = [], []
    mean_fpr = np.linspace(0, 1, 100)
    cosine_auc_values = []
    euclidean_auc_values = []

    plt.figure(figsize=(10, 8))

    for train_index, test_index in kf.split(embeddings):
        train_embeddings, test_embeddings = embeddings[train_index], embeddings[test_index]
        train_labels, test_labels = syndrome_ids[train_index], syndrome_ids[test_index]

        # Cosine ROC AUC
        cosine_knn_preds = knn_classification(train_embeddings, train_labels, test_embeddings, metric='cosine')
        fpr_cos, tpr_cos, _ = roc_curve(test_labels, cosine_knn_preds[:, 1], pos_label=1)
        cosine_auc_values.append(auc(fpr_cos, tpr_cos))
        tprs_cosine.append(np.interp(mean_fpr, fpr_cos, tpr_cos))  # Interpolate TPR for the common FPR scale
        tprs_cosine[-1][0] = 0.0

        # Euclidean ROC AUC
        euclidean_knn_preds = knn_classification(train_embeddings, train_labels, test_embeddings, metric='euclidean')
        fpr_euc, tpr_euc, _ = roc_curve(test_labels, euclidean_knn_preds[:, 1], pos_label=1)
        euclidean_auc_values.append(auc(fpr_euc, tpr_euc))
        tprs_euclidean.append(np.interp(mean_fpr, fpr_euc, tpr_euc))
        tprs_euclidean[-1][0] = 0.0

    # Compute the mean TPR for each algorithm
    mean_tpr_cosine = np.mean(tprs_cosine, axis=0)
    mean_tpr_cosine[-1] = 1.0  # Ensure TPR ends at 1
    mean_auc_cosine = auc(mean_fpr, mean_tpr_cosine)

    mean_tpr_euclidean = np.mean(tprs_euclidean, axis=0)
    mean_tpr_euclidean[-1] = 1.0
    mean_auc_euclidean = auc(mean_fpr, mean_tpr_euclidean)

    # Plot the averaged ROC AUC curves
    plt.plot(mean_fpr, mean_tpr_cosine, color='blue', label=f'Cosine (Mean AUC = {mean_auc_cosine:.2f})')
    plt.plot(mean_fpr, mean_tpr_euclidean, color='green', label=f'Euclidean (Mean AUC = {mean_auc_euclidean:.2f})')

    plt.title('ROC AUC Comparison - Cosine vs Euclidean (Averaged)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f'ROC AUC plot saved as {output_path}')

    return cosine_auc_values, euclidean_auc_values

if __name__ == '__main__':
    data = load_data('mini_gm_public_v0.1.p')
    embeddings, encoded_syndrome_ids, le = prepare_data(data)

    cosine_auc = cross_validation(embeddings, encoded_syndrome_ids, metric='cosine')
    euclidean_auc = cross_validation(embeddings, encoded_syndrome_ids, metric='euclidean')

    print(f'Mean AUC for Cosine Distance: {cosine_auc}')
    print(f'Mean AUC for Euclidean Distance: {euclidean_auc}')

    cosine_auc_values, euclidean_auc_values = plot_roc_auc(embeddings, encoded_syndrome_ids)

    # Write to a text file
    with open('performance_comparison.txt', 'w') as f:
        f.write('Performance Comparison\n')
        f.write('----------------------\n')
        f.write('\nCosine Distance\n')
        f.write('Fold\tAUC\n')
        for i, score in enumerate(cosine_auc_values, start=1):
            f.write(f'Fold {i}\t{score:.2f}\n')
        f.write(f'\nAverage AUC: {cosine_auc:.2f}\n')
        
        f.write('\nEuclidean Distance\n')
        f.write('Fold\tAUC\n')
        for i, score in enumerate(euclidean_auc_values, start=1):
            f.write(f'Fold {i}\t{score:.2f}\n')
        f.write(f'\nAverage AUC: {euclidean_auc:.2f}\n')