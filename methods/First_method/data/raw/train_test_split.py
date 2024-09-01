import numpy as np
import pandas as pd


def files_train_test_split(annotations, notes, discharge, test_ratio=0.2):    
    test_ratio = 0.2

    assert set(list(annotations.note_id)) == set(list(notes.note_id))
    note_ids = set(list(annotations.note_id))

    test_size = int(len(note_ids) * test_ratio)
    test_ids = set(np.random.choice(np.array(list(note_ids)), test_size, replace=False))
    
    train_ids = note_ids - test_ids
    
    annotations_train = annotations.loc[annotations.note_id.isin(train_ids)]
    annotations_test = annotations.loc[annotations.note_id.isin(test_ids)]
    
    notes_train = notes.loc[notes.note_id.isin(train_ids)]
    notes_test = notes.loc[notes.note_id.isin(test_ids)]
    
    train_ids_discharge = set(list(discharge.note_id)) - test_ids
    discharge_train = discharge.loc[discharge.note_id.isin(train_ids_discharge)]
    discharge_test = discharge.loc[discharge.note_id.isin(test_ids)]
    
    annotations_train.to_csv('train_annotations.csv')
    annotations_test.to_csv('test_annotations.csv')
    
    notes_train.to_csv('mimic-iv_notes_training_set.csv')
    notes_test.to_csv('mimic-iv_notes_testing_set.csv')
    
    discharge_train.to_csv('discharge.csv.gz', compression='gzip')
    discharge_test.to_csv('discharge_test.csv.gz', compression='gzip')
    

if __name__ == '__main__':
    annotations = pd.read_csv('complete_annotations.csv')
    notes = pd.read_csv('mimic-iv_notes_complete_set.csv')
    discharge = pd.read_csv('complete_discharge.csv.gz', compression='gzip')
    
    files_train_test_split(annotations, notes, discharge, test_ratio=0.2)
