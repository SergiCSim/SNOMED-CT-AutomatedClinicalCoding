from pathlib import Path
import pandas as pd

from mimic_dev_main import make_kiri_dicts


if __name__ == '__main__':
    texts = pd.read_csv('../data/raw/mimic-iv_notes_training_set.csv').set_index(
        "note_id"
    )["text"]
    annotations = pd.read_csv('../data/interim/train_annotations_cln.csv')
    submission_path = Path('./')
    
    make_kiri_dicts(texts, annotations, submission_path)
