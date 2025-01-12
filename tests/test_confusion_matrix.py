import pytest
import numpy as np
from utility.confusion_matrix import ConfusionMatrix
from sklearn.metrics import confusion_matrix

@pytest.fixture(scope="module")
def generate_test_masks():
    test_masks = []
    actual_mask = np.array([[0, 0, 0],
                            [255, 255, 255],
                            [255, 255, 255]]).astype(np.uint8)
    predicted_mask = np.array([[0, 255, 0],
                               [255, 0, 255],
                               [0, 255, 255]]).astype(np.uint8)
    test_masks.append((actual_mask, predicted_mask))

    actual_mask = np.array([[0, 0, 0, 255, 0],
                            [255, 255, 255, 0, 0],
                            [255, 255, 255, 255, 0]]).astype(np.uint8)
    predicted_mask = np.array([[0, 255, 0, 255, 0],
                               [255, 0, 255, 0, 0],
                               [0, 255, 255, 255, 255]]).astype(np.uint8)
    test_masks.append((actual_mask, predicted_mask))

    actual_mask = np.array([[0, 0, 0, 255, 0, 255],
                            [255, 255, 255, 0, 0, 255],
                            [255, 255, 255, 255, 0, 255]]).astype(np.uint8)
    predicted_mask = np.array([[0, 255, 0, 255, 0, 0],
                               [255, 0, 255, 0, 0, 255],
                               [0, 255, 255, 255, 255, 255]]).astype(np.uint8)
    test_masks.append((actual_mask, predicted_mask))

    return test_masks

def compute_on_single_sample(actual_mask, predicted_mask):
    my_cm = ConfusionMatrix()
    my_cm.compute_on_single_sample(actual_mask, predicted_mask)
    my_cm_res = my_cm.get_confusion_matrix()

    actual_mask_skl = actual_mask.flatten()
    predicted_mask_skl = predicted_mask.flatten()
    skl_cm_res = confusion_matrix(actual_mask_skl, predicted_mask_skl).ravel()

    tn, fp, fn, tp = skl_cm_res
    assert my_cm_res["tp"] == tp
    assert my_cm_res["fp"] == fp
    assert my_cm_res["tn"] == tn
    assert my_cm_res["fn"] == fn

def test_all_single_samples(generate_test_masks):
    test_masks = generate_test_masks

    for actual_mask, predicted_mask in test_masks:
        compute_on_single_sample(actual_mask, predicted_mask)