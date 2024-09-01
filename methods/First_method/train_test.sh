#!/bin/bash

echo "TRAIN-TEST SPLITTING..."
cd data/raw
python train_test_split.py

echo "\nPREPROCESSING..."
cd ../..
make interim

cp data/raw/mimic-iv_notes_testing_set.csv submission/data/test_notes.csv
cp data/raw/test_annotations.csv submission/data/test_annotations.csv

echo "\nMAKING KIRI DICTIONARIES..."
cd src
python make_kiri_dicts.py

cd ..
mv src/kiri_dicts.pkl submission/assets/kiri_dicts.pkl
cp data/interim/abbr_dict.pkl submission/assets/abbr_dict.pkl
cp data/interim/term_extension.csv submission/assets/term_extension.csv

echo "GENERATING THE SUBMISSION FILE..."
cd submission
python main.py

echo "\nEVALUATING..."
python evaluate.py
