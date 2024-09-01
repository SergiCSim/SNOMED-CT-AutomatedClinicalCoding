import numpy as np
import pandas as pd
from typing import List
import scipy.sparse as sp
from sklearn.metrics import precision_recall_fscore_support


def miou(user_annotations: pd.DataFrame, target_annotations: pd.DataFrame) -> List[float]:
    """
    Calculate the IoU metric for each class in a set of annotations.
    """
    # Get mapping from note_id to index in array
    docs = np.unique(np.concatenate([user_annotations.note_id, target_annotations.note_id]))
    doc_index_mapping = dict(zip(docs, range(len(docs))))

    # Identify union of categories in GT and PRED
    cats = np.unique(np.concatenate([user_annotations.concept_id, target_annotations.concept_id]))

    # Find max character index in GT or PRED
    max_end = np.max(np.concatenate([user_annotations.end, target_annotations.end]))

    # Populate matrices for keeping track of character class categorization
    def populate_char_mtx(n_rows, n_cols, annot_df):
        mtx = sp.lil_array((n_rows, n_cols), dtype=np.uint64)
        for row in annot_df.itertuples():
            doc_index = doc_index_mapping[row.note_id]
            mtx[doc_index, row.start : row.end] = row.concept_id  # noqa: E203
        return mtx.tocsr()

    gt_mtx = populate_char_mtx(docs.shape[0], max_end, target_annotations)
    pred_mtx = populate_char_mtx(docs.shape[0], max_end, user_annotations)

    # Calculate IoU per category
    ious = []
    for cat in cats:
        gt_cat = gt_mtx == cat
        pred_cat = pred_mtx == cat
        # sparse matrices don't support bitwise operators, but the _cat matrices
        # have bool dtypes so when we multiply/add them we end up with only T/F values
        intersection = gt_cat * pred_cat
        union = gt_cat + pred_cat
        iou = intersection.sum() / union.sum()
        ious.append(iou)

    return np.mean(ious)


def f1_score(prediction_df, ground_truth_df):
    """
    Calculate the F1 score between prediction and ground truth dataframes.
    """

    # Merge to find true positives (exact matches)
    merged = pd.merge(prediction_df, ground_truth_df, 
                      on=['note_id', 'start', 'end', 'concept_id'],
                      how='inner')

    # True Positives (TP)
    tp = len(merged)

    # False Positives (FP) - Predictions not in Ground Truth
    fp = len(prediction_df) - tp

    # False Negatives (FN) - Ground Truth not in Predictions
    fn = len(ground_truth_df) - tp

    # Calculating precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def macro_f1_score(df_pred, df_truth):
    """
    Calculate the Macro-F1 score for entity recognition and linking.

    Parameters:
    df_pred (pd.DataFrame): DataFrame containing the predictions with columns ['note_id', 'start', 'end', 'concept_id']
    df_truth (pd.DataFrame): DataFrame containing the ground truth with columns ['note_id', 'start', 'end', 'concept_id']

    Returns:
    float: Macro-F1 score
    """
    # Create a set of unique concepts in the predictions and ground truth
    unique_concepts = set(df_pred['concept_id']).union(set(df_truth['concept_id']))

    # Initialize lists to store precision, recall, and F1 scores for each concept
    f1_scores = []

    for concept in unique_concepts:
        # Filter predictions and ground truth for the current concept
        pred_concept = df_pred[df_pred['concept_id'] == concept]
        truth_concept = df_truth[df_truth['concept_id'] == concept]

        # Merge on note_id, start, and end to find true positives
        tp = pd.merge(pred_concept, truth_concept, on=['note_id', 'start', 'end', 'concept_id']).shape[0]

        # Calculate false positives and false negatives
        fp = pred_concept.shape[0] - tp
        fn = truth_concept.shape[0] - tp

        # Handle the case where tp, fp, fn are all zero to avoid division by zero
        if tp == 0 and fp == 0 and fn == 0:
            f1 = 1.0  # Perfect F1 for a concept with no instances in both prediction and truth
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores.append(f1)

    # Calculate Macro-F1 as the average of all F1 scores
    macro_f1 = sum(f1_scores) / len(f1_scores)
    return macro_f1