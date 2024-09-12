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
    """Saves the ROC AUC plots for cosine and euclidean metrics.

    Args:
      embeddings: Image embeddings.
      syndrome_ids: Syndrome IDs.
      output_path: File path to save the plot.
      n_splits: Number of folds for cross-validation.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
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
        plt.plot(fpr_cos, tpr_cos, label=f'Cosine Fold {len(cosine_auc_values)} (AUC={auc(fpr_cos, tpr_cos):.2f})')
        print(f'Cosine Fold {len(cosine_auc_values)} (AUC={auc(fpr_cos, tpr_cos):.2f})')

        # Euclidean ROC AUC
        euclidean_knn_preds = knn_classification(train_embeddings, train_labels, test_embeddings, metric='euclidean')
        fpr_euc, tpr_euc, _ = roc_curve(test_labels, euclidean_knn_preds[:, 1], pos_label=1)
        euclidean_auc_values.append(auc(fpr_euc, tpr_euc))
        plt.plot(fpr_euc, tpr_euc, label=f'Euclidean Fold {len(euclidean_auc_values)} (AUC={auc(fpr_euc, tpr_euc):.2f})')
        print(f'Euclidean Fold {len(euclidean_auc_values)} (AUC={auc(fpr_euc, tpr_euc):.2f})')
        
    plt.title('ROC AUC Comparison - Cosine vs Euclidean')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(output_path)
    print(f'ROC AUC plot save as {output_path}')

# Função principal
if __name__ == '__main__':
    data = load_data('mini_gm_public_v0.1.p')
    embeddings, encoded_syndrome_ids, le = prepare_data(data)

    cosine_auc = cross_validation(embeddings, encoded_syndrome_ids, metric='cosine')
    euclidean_auc = cross_validation(embeddings, encoded_syndrome_ids, metric='euclidean')

    print(f'Mean AUC for Cosine Distance: {cosine_auc}')
    print(f'Mean AUC for Euclidean Distance: {euclidean_auc}')

    plot_roc_auc(embeddings, encoded_syndrome_ids)