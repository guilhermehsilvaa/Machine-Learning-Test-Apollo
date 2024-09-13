import pytest
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from knn_evaluation import load_data, prepare_data

@pytest.fixture
def mock_data():
    """Creates mock data for testing.

    Returns:
        A dictionary representing syndrome IDs, subject IDs, and embeddings.
    """
    return {
        "syndrome_1": {
            "subject_1": {
                "image_1": np.array([0.1, 0.2, 0.3]),
                "image_2": np.array([0.4, 0.5, 0.6]),
            },
            "subject_2": {
                "image_1": np.array([0.7, 0.8, 0.9]),
            },
        },
        "syndrome_2": {
            "subject_3": {
                "image_1": np.array([1.0, 1.1, 1.2]),
            },
        },
    }

def test_load_data(tmp_path, mock_data):
    """Tests the load_data function to ensure it correctly loads data from a pickle file.

    Args:
        tmp_path: Temporary path to store the test pickle file.
        mock_data: Mock data to be written and compared after loading.

    Raises:
        AssertionError: If the loaded data doesn't match the original mock data.
    """
    # Create a temporary pickle file for testing
    file_path = tmp_path / "test_data.p"
    with open(file_path, 'wb') as f:
        pickle.dump(mock_data, f)

    loaded_data = load_data(file_path)

    # Manually compare dictionaries to handle NumPy arrays
    for syndrome_id in mock_data:
        assert syndrome_id in loaded_data, f"Syndrome ID '{syndrome_id}' not found in loaded data"
        for subject_id in mock_data[syndrome_id]:
            assert subject_id in loaded_data[syndrome_id], f"Subject ID '{subject_id}' not found in loaded data"
            for image_id in mock_data[syndrome_id][subject_id]:
                np.testing.assert_array_equal(
                    loaded_data[syndrome_id][subject_id][image_id],
                    mock_data[syndrome_id][subject_id][image_id],
                    err_msg=f"Image ID '{image_id}' does not match the original data"
                )

def test_prepare_data(mock_data):
    """Tests the prepare_data function to ensure it correctly processes and normalizes the data.

    Args:
        mock_data: Mock data to be processed and compared after preparation.

    Raises:
        AssertionError: If the processed data does not match the expected results.
    """
    embeddings, encoded_syndrome_ids, le = prepare_data(mock_data)

    # Verify the arrays and encoded labels types
    assert isinstance(embeddings, np.ndarray), "Embeddings should be a NumPy array"
    assert isinstance(encoded_syndrome_ids, np.ndarray), "Syndrome IDs should be a NumPy array"
    assert isinstance(le, LabelEncoder), "LabelEncoder should be returned"

    # Verify the embeddings shape
    assert embeddings.shape == (4, 3), "Embeddings shape should be (4, 3)"
    assert encoded_syndrome_ids.shape == (4,), "Syndrome IDs shape should be (4,)"

    # Flatten the mock data for comparison
    mock_embeddings = []
    mock_syndrome_ids = []
    for syndrome_id, subjects in mock_data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                mock_embeddings.append(embedding)
                mock_syndrome_ids.append(syndrome_id)

    mock_embeddings = np.array(mock_embeddings)
    mock_syndrome_ids = np.array(mock_syndrome_ids)

    # Check if the embeddings have been scaled (mean should be ~0 and std should be ~1)
    assert np.allclose(np.mean(embeddings, axis=0), 0, atol=1e-1), "Mean of embeddings should be close to 0 after scaling"
    assert np.allclose(np.std(embeddings, axis=0), 1, atol=1e-1), "Standard deviation of embeddings should be close to 1 after scaling"

    # Check if the syndrome IDs are encoded correctly
    expected_encoded_syndrome_ids = LabelEncoder().fit(mock_syndrome_ids).transform(mock_syndrome_ids)
    np.testing.assert_array_equal(encoded_syndrome_ids, expected_encoded_syndrome_ids, err_msg="Encoded syndrome IDs do not match expected values")