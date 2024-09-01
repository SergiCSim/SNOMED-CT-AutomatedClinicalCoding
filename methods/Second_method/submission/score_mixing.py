import re
from collections import OrderedDict

import pandas as pd
from tqdm import tqdm

from pathlib import Path
import json

import ast



def get_true_header_indices(text, true_headers):
    text = re.sub("\n", " ", text.lower())
    true_header_indices = {}
    for true_header in true_headers:
        pos = text.find(true_header)
        if pos != -1:
            true_header_indices[true_header] = pos
    true_header_indices = dict(sorted(true_header_indices.items(), key=lambda item: item[1]))
    true_header_indices = OrderedDict(true_header_indices)
    return true_header_indices


def calc_header_span(df):
    true_headers = [
        "past medical history:",
        "allergies:",
        "history of present illness:",
        "physical exam:",
        "admission date:  discharge date:",
        "attending:",
        "major surgical or invasive procedure:",
        "family history:",
        "discharge disposition:",
        "discharge condition:",
        "discharge instructions:",
        "name:  unit no:",
        "social history:",
        "chief complaint:",
        "pertinent results:",
        "discharge medications:",
        "medications on admission:",
        "___ on admission:",
        "discharge diagnosis:",
        "followup instructions:",
        "brief hospital course:",
        "facility:",
        "impression:",
    ]
    res = {}
    for i, row in df.iterrows():
        text = row["text"]
        headers = get_true_header_indices(text.lower(), true_headers)
        headers_spans = {}
        for header, start in headers.items():
            i = list(headers).index(header)
            if i == len(headers) - 1:
                end = len(text)
            else:
                next_header = list(headers)[i + 1]
                end = headers[next_header]
            headers_spans[header] = (start, end)
        res[row.note_id] = headers_spans
    return res


def add_spans(dfa, dfn):
    notes_headers = calc_header_span(dfn)
    gg = dfa.groupby("note_id")
    res = []
    for note_id, group in tqdm(gg, desc="Adding spans.."):
        note_headers_spans = notes_headers[note_id]
        for i, row in group.iterrows():
            start, end = row.start, row.end
            for header, (hstart, hend) in note_headers_spans.items():
                if start >= hstart and start <= hend:
                    row["header"] = header
                    break
            res.append(row)
    dfa = pd.DataFrame(res)
    return dfa


def clean_by_header(dfa, dfn):
    cut_headers = [
        "medications on admission:",
        "___ on admission:",
        "discharge medications:",
    ]
    dfa = add_spans(dfa, dfn)
    dfa = dfa[~dfa.header.isin(cut_headers)]
    return dfa
    

def load_sctid_syn(sctid_syn_parh: Path):
    with open(sctid_syn_parh, "r") as f:
        data = json.load(f)
        return set(map(int, data.keys()))

def add_concept_class(ann_df, sctid_syn_dir: Path):
    p_cids = load_sctid_syn(Path(sctid_syn_dir) / "proc_sctid_syn.json")
    p_cids.add(71388002)
    f_cids = load_sctid_syn(Path(sctid_syn_dir) / "find_sctid_syn.json")
    b_cids = load_sctid_syn(Path(sctid_syn_dir) / "body_sctid_syn.json")
    snomed_class = []
    for i, r in ann_df.iterrows():
        cid = r.concept_id
        if cid in p_cids:
            label = "proc"
        elif cid in b_cids:
            label = "body"
        elif cid in f_cids:
            label = "find"
        else:
            label = "???"
            raise ValueError(f"unknown concept_id: {cid}")
        snomed_class.append(label)

    ann_df["cls"] = snomed_class
    return ann_df
    
    
def handle_nan(s):
    split = s.split(',')
    split[0] = split[0][1:]
    split[-1] = split[-1][:-1]
    split.insert(0, '[')
    split.insert(len(split), ']')
    for i, s in enumerate(split):
        if 'nan' in s:
            split[i] = '-1'
    lst = ','.join(split)
    lst = lst[0:1] + lst[2:-2] + lst[-1:]
    return lst

def process_row(row, ann, total, header_probs, class2int):
    scores = row['sap_scores']
    cids = row['sap_cids']

    if 'nan' in scores:
      scores = handle_nan(scores)

    scores = ast.literal_eval(scores)
    cids = ast.literal_eval(cids)

    header = row['header']
    cls = row['class']

    filtered_scores_cids = [(score, cid) for score, cid in zip(scores, cids)]

    combined_scores_cids = [(combine_probs(ann, score, cid, header, cls, total, header_probs, class2int), cid) for score, cid in filtered_scores_cids]

    max_combined_score, max_cid = max(combined_scores_cids, key=lambda x: x[0])
    row['sap_scores'] = max_combined_score
    row['sap_cids'] = max_cid

    return row

'''
def calculate_bayesian_prob(ann, cid, header, cls, total, header_probs, class2int):
    p_intersect = len(ann.loc[(ann.concept_id == cid) & (ann.header == header) & (ann.cls == cls)]) / total[class2int[cls]]
    if header not in header_probs[class2int[cls]]:
        return 0
    p_h = header_probs[class2int[cls]][header]
    #print(p_intersect, p_h)
    return p_intersect / p_h
'''

def calculate_bayesian_prob(ann, cid, header, cls, total, header_probs, class2int):
    num = len(ann.loc[(ann.header == header) & (ann.concept_id == cid)])
    den = len(ann.loc[ann.concept_id == cid])
    #return f"{num} / {den} = {num / den}"
    if den == 0:
        return 0
    return num / den

def combine_probs(ann, p, cid, header, cls, total, header_probs, class2int, weights=[0.8, 0.2]):
    bp = calculate_bayesian_prob(ann, cid, header, cls, total, header_probs, class2int)
    #print(p, bp)
    return p * 1 + bp * 0.5
    
    
def mix_scores(df, train_annotations_path, train_notes_path):
    #sctid_syn_dir = Path('/content/drive/MyDrive/2nd_Place/data/preprocess_data')
    sctid_syn_dir = Path(r'C:\Users\sergi\TFM\mixed\data\preprocess_data')
    
    train_ann = pd.read_csv(train_annotations_path)
    train_notes = pd.read_csv(train_notes_path)

    train_ann = clean_by_header(train_ann, train_notes)
    train_ann = add_concept_class(train_ann, sctid_syn_dir)

    header_counts = [dict(train_ann.loc[train_ann.cls == c].header.value_counts()) for c in train_ann.cls.unique()]
    total = [sum(hc.values()) for hc in header_counts]
    header_probs = [{k : v / total[i] for k, v in hc.items()} for i, hc in enumerate(header_counts)]
    class2int = {c : i for i, c in enumerate(train_ann.cls.unique())}
    
    # Mix the scores
    df = df.apply(process_row, axis=1, args=(train_ann, total, header_probs, class2int))
    
    return df