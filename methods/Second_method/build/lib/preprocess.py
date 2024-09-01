import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from cut_headers import cut_headers
from embeds import SepBERTEmbedder, get_embeds
from loguru import logger
from sklearn.model_selection import KFold
from snomed_graph import SnomedGraph
from static_dict import get_most_common_concept
from transformers import AutoModel, AutoTokenizer

root = r"D:\2nd Place"
BodyId = 123037004
ProcId = 71388002
FindId = 404684003


def get_syn(class_id, SG: SnomedGraph):
    allc = SG.get_descendants(class_id)
    res = {a.sctid: a.synonyms for a in list(allc)}
    if class_id == FindId:
        res[298430001] = ["Increased active range of cervical spine right lateral flexion"]
        res[298577009] = ["Observation of sensation of musculoskeletal structure of thoracic spine"]
        res[312087002] = ["Disorder following clinical procedure"]
    return res


def convert_snomed_rf2_to_serialized(src_path: str, dst_path: str):
    print('-----------------------------------------\n\n\n\n')
    print(src_path)
    print('\n\n\n\n-----------------------------------------')
    logger.info(f"Converting RF2 to serialized graph")
    SG = SnomedGraph.from_rf2(src_path)
    SG.save(dst_path)
    logger.info(f"serialized snomed graph saved to {dst_path}")


def get_checkpoint(name: str, path: str):
    model_path = os.path.join(path, "model")
    os.makedirs(model_path, exist_ok=True)
    model = AutoModel.from_pretrained(name)
    model.save_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Successfully saved {name} to {path}")
    embeds_path = os.path.join(path, "embeds")
    os.makedirs(embeds_path, exist_ok=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--val", action="store_true")
    args = args.parse_args()

    preprocess_data_path = os.path.join(root, "data", "preprocess_data")
    print('-----------------------------------------\n\n\n\n')
    print(root)
    print('\n\n\n\n-----------------------------------------')
    os.makedirs(preprocess_data_path, exist_ok=True)

    ROOT_DIR = os.path.join(root, "data")
    if args.val:
        TRAIN_NOTES_PATH = os.path.join(ROOT_DIR, "preprocess_data", "splits", "train_note_split_0.csv")
        TRAIN_ANNOTAIONS_PATH = os.path.join(ROOT_DIR, "preprocess_data", "splits", "train_ann_split_0.csv")
        STATIC_DICT_PATH = os.path.join(ROOT_DIR, "preprocess_data", "most_common_concept_val_0.pkl")
    else:
        RAW_TRAIN_NOTES_PATH = os.path.join(ROOT_DIR, "competition_data", "mimic-iv_notes_training_set.csv")
        RAW_TRAIN_ANNOTAIONS_PATH = os.path.join(ROOT_DIR, "competition_data", "train_annotations.csv")
        TRAIN_NOTES_PATH = os.path.join(ROOT_DIR, "competition_data", "cutmed_notes.csv")
        TRAIN_ANNOTAIONS_PATH = os.path.join(ROOT_DIR, "competition_data", "cutmed_fixed_train_annotations.csv")
        STATIC_DICT_PATH = os.path.join(ROOT_DIR, "preprocess_data", "most_common_concept.pkl")

    SPLIT_PATH = os.path.join(ROOT_DIR, "preprocess_data", "splits")
    SNOMED_GRAPH_RF2_DIR = os.path.join(ROOT_DIR, "competition_data",
                                        "SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition") #"sct2_Relationship_Snapshot_INT_20230531.txt")
    print('-----------------------------------------\n\n\n\n')
    print(SNOMED_GRAPH_RF2_DIR)
    print('\n\n\n\n-----------------------------------------')
    SNOMED_GRAPH_RF2_SERIALIZED = os.path.join(ROOT_DIR, "competition_data", "graph.gml")

    PROC_SCTID_SYN_PATH = os.path.join(ROOT_DIR, "preprocess_data", "proc_sctid_syn.json")
    FIND_SCTID_SYN_PATH = os.path.join(ROOT_DIR, "preprocess_data", "find_sctid_syn.json")
    BODY_SCTID_SYN_PATH = os.path.join(ROOT_DIR, "preprocess_data", "body_sctid_syn.json")
    SECOND_STAGE_PATH = os.path.join(ROOT_DIR, "second_stage")
    SECOND_STAGE_MODELS = {
        "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token",
    }

    if all([os.path.exists(TRAIN_NOTES_PATH), os.path.exists(TRAIN_ANNOTAIONS_PATH)]):
        logger.warning("cutmed_notes already exist, skipping")
    else:
        notes = pd.read_csv(RAW_TRAIN_NOTES_PATH)
        annotation = pd.read_csv(RAW_TRAIN_ANNOTAIONS_PATH)
        print(notes.shape, annotation.shape)
        notes, annotation = cut_headers(notes, annotation)
        print(notes.shape, annotation.shape)
        notes.to_csv(TRAIN_NOTES_PATH, index=False)

        annotation.to_csv(TRAIN_ANNOTAIONS_PATH, index=False)
        os.makedirs(SPLIT_PATH, exist_ok=True)
        Fold = KFold(n_splits=4, shuffle=True, random_state=42)

        X = np.array(range(len(notes)))
        for n, (_, val_index) in enumerate(Fold.split(X)):
            val_note_split = notes.copy().loc[val_index]
            print(val_note_split.head())
            print(val_note_split.shape, len(val_index))
            val_note_split.to_csv(os.path.join(SPLIT_PATH, f"val_note_split_{n}.csv"), index=False)
            train_note_split = notes.drop(val_index)
            train_note_split.to_csv(os.path.join(SPLIT_PATH, f"train_note_split_{n}.csv"), index=False)
            val_ann_split = annotation[annotation.note_id.isin(val_note_split.note_id)]
            val_ann_split.to_csv(os.path.join(SPLIT_PATH, f"val_ann_split_{n}.csv"), index=False)
            train_ann_split = annotation[annotation.note_id.isin(train_note_split.note_id)]
            train_ann_split.to_csv(os.path.join(SPLIT_PATH, f"train_ann_split_{n}.csv"), index=False)
    if all(
        [os.path.exists(PROC_SCTID_SYN_PATH), os.path.exists(FIND_SCTID_SYN_PATH), os.path.exists(BODY_SCTID_SYN_PATH)]
    ):
        logger.warning("sctid_syns already exist, skipping")
    else:
        if not os.path.exists(SNOMED_GRAPH_RF2_SERIALIZED):
            assert os.path.exists(SNOMED_GRAPH_RF2_DIR), f"{SNOMED_GRAPH_RF2_DIR} does not exist"
            print('-----------------------------------------\n\n\n\n')
            print(SNOMED_GRAPH_RF2_DIR)
            print('\n\n\n\n-----------------------------------------')
            print('-----------------------------------------\n\n\n\n')
            print(SNOMED_GRAPH_RF2_SERIALIZED)
            print('\n\n\n\n-----------------------------------------')
            convert_snomed_rf2_to_serialized(SNOMED_GRAPH_RF2_DIR, SNOMED_GRAPH_RF2_SERIALIZED)
        SG = SnomedGraph.from_serialized(SNOMED_GRAPH_RF2_SERIALIZED)
        for path, concept_id in [
            (PROC_SCTID_SYN_PATH, ProcId),
            (FIND_SCTID_SYN_PATH, FindId),
            (BODY_SCTID_SYN_PATH, BodyId),
        ]:
            syn = get_syn(concept_id, SG)
            with open(path, "w") as f:
                json.dump(syn, f)
            logger.info(f"sctid_syns ({concept_id}) saved to {path}")
    if os.path.exists(STATIC_DICT_PATH):
        logger.warning("static_dict already exists, skipping")
    else:
        notes = pd.read_csv(TRAIN_NOTES_PATH)
        annotation = pd.read_csv(TRAIN_ANNOTAIONS_PATH)
        get_most_common_concept(STATIC_DICT_PATH, notes, annotation)
        logger.info(f"static_dict saved to {STATIC_DICT_PATH}")
    for path, name in SECOND_STAGE_MODELS.items():
        if os.path.exists(os.path.join(SECOND_STAGE_PATH, path)):
            logger.warning(f"{name} already exist, skipping")
        else:
            get_checkpoint(name, os.path.join(SECOND_STAGE_PATH, path))

    cuda = True
    embedders = [
        (SepBERTEmbedder(cuda), "sapbert", "sapbertmean"),
    ]
    for embedder, dir_name, name in embedders:
        for sctid_syn_path, concept_type in [
            (BODY_SCTID_SYN_PATH, "body"),
            (FIND_SCTID_SYN_PATH, "find"),
            (PROC_SCTID_SYN_PATH, "proc"),
        ]:
            save_path = os.path.join(SECOND_STAGE_PATH, dir_name, "embeds", f"{name}_{concept_type}.pth")
            if os.path.exists(save_path):
                logger.warning(f"{name} {concept_type} embeds already exist, skipping")
                continue
            with open(sctid_syn_path) as f:
                sctid_syn = json.load(f)
            xb = get_embeds(embedder, sctid_syn, cuda)
            torch.save(xb, save_path)
            logger.info(f"embeds saved to {save_path}")
