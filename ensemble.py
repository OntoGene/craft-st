#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


"""
Merge individual predictions into ensemble output.
"""


import csv
import sys
import logging
import itertools as it
from pathlib import Path
from collections import Counter
from typing import List, Dict, Iterable, Iterator

import numpy as np

import train


FOLDS = 6
TESTSET_SIZE = 30

# Hard-coded paths!
HERE = Path(__file__).parent

CORPUS = HERE / 'labeled.feat'
SPLITS = HERE / 'splits.subm.json'
VOCAB = HERE / 'vocab.res.txt'
ABBREVS = HERE / 'abbrevs.json'


def main():
    '''
    Run as script.
    '''
    etype = sys.argv[1]
    srcdir = Path(sys.argv[2])
    tgtdir = Path(sys.argv[3])
    train.setup_logging()
    merge(etype, srcdir, tgtdir)


def merge(etype: str, srcdir: Path, tgtdir: Path, pick_best: bool = True):
    """Pick the best models and merge their predictions."""
    splits = train.read_json(SPLITS)
    if pick_best:
        result_files = sorted(
            srcdir.glob('run*/{}_results.tsv'.format(etype.lower())))
        runs = pick_runs_per_fold(result_files, splits)
        folds = [srcdir/'subm/{}.fold-{}.run-{}'.format(etype, f, r)
                 for f, r in enumerate(runs)]
        assert len(folds) == FOLDS, 'missing folds, found only {}'.format(folds)
    else:
        folds = list(srcdir.glob('subm/{}.fold-*.run-*'.format(etype)))
    labels = [train.read_vocab(p/'labels', reserved=0) for p in folds]
    assert all(l == labels[0] for l in labels[1:]), 'differing labels'
    labels = labels[0]

    conll_files = (CORPUS/etype).glob('*')
    vocab = train.read_vocab(VOCAB)
    abbrevs = train.read_json(ABBREVS)
    data = train.Dataset.from_files(conll_files, vocab=vocab, concept_ids=labels,
                                    abbrevs=abbrevs)
    docs = splits[0]['test']
    assert len(docs) == TESTSET_SIZE, 'missing test documents'

    for docid in docs:
        scores = _merge_scores(p/'{}.npz'.format(docid) for p in folds)
        fix_disagreements(*scores)
        data.dump_conll(tgtdir/etype, [docid], scores)


def pick_runs_per_fold(result_files: Iterable[Path],
                       splits: Iterable[Dict[str, List[str]]]) -> Iterator[int]:
    """
    Pick the best run for each fold, based on the official scores.

    Args:
        result_files: one TSV per run
        splits: train/dev/test split for each fold

    Returns:
        a run number 1..n for each fold
    """
    docs_by_fold = {}
    for f, docs in enumerate(splits):
        docs_by_fold.update(dict.fromkeys(docs['dev'], f))
    results = [_parse_results(p, docs_by_fold) for p in result_files]
    for scores in zip(*results):
        yield _pick_run(scores)


def _parse_results(path, docs_by_fold):
    scorers = [Scorer() for _ in range(max(docs_by_fold.values())+1)]
    with path.open(encoding='utf8') as f:
        rows = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in rows:
            try:
                fold = docs_by_fold[row['#document-id']]
            except KeyError:
                continue  # skip the "TOTAL" line
            scorers[fold].update(row)
    return scorers


def _pick_run(results):
    winner_ser = min(range(len(results)), key=lambda i: results[i].ser())
    winner_f1 = max(range(len(results)), key=lambda i: results[i].f1())
    if winner_f1 != winner_ser:
        logging.warning('picking unclear')
    return winner_f1 + 1


class Scorer:
    """Aggregator for micro SER/F1."""
    def __init__(self):
        self.counts = Counter()

    def update(self, row):
        """Update counts with a csv.DictReader row."""
        self.counts.update((k, float(v)) for k, v in row.items())

    def f1(self):
        """F-Score."""
        c = self.counts
        try:
            return 2 * c['matches'] / (c['ref-count'] + c['prediction-count'])
        except ZeroDivisionError:  # |ref| == |pred| == 0 -> correct
            return 1.

    def ser(self):
        """Slot error rate."""
        c = self.counts
        numerator = c['substitutions'] + c['insertions'] + c['deletions']
        try:
            return numerator / c['ref-count']
        except ZeroDivisionError:  # correct (0) if |ins| == |subst| == 0
            return float(bool(numerator))


def _merge_scores(paths):
    ner, nen = [], []
    for p in paths:
        with np.load(str(p)) as f:
            ner.append(f['ner'])
            nen.append(f['nen'])
    ner = np.mean(ner, axis=0)
    nen = np.mean(nen, axis=0)
    return [ner, nen]


# TODO: this should probably be done in train.Dataset.dump_conll()

def fix_disagreements(ner, nen):
    """
    Fix cases where NER and NEN disagree.

    Disagreement means NER predicts O and NEN predicts a
    non-NIL label, or vice versa.
    In those cases, change either of them to O/NIL or to
    the second-best label, whichever gives the higher score
    in combination.
    """
    disagreements = (ner.argmax(-1)==0) != (nen.argmax(-1)==0)
    for s, t in it.product(*map(range, disagreements.shape)):
        if disagreements[s, t]:
            # What scores better? O * NIL or max(BIES) * max(non-NIL)?
            irrelevant = ner[s, t, 0] * nen[s, t, 0]
            relevant = ner[s, t, 1:].max() * nen[s, t, 1:].max()
            # Set the scores for the losing combination to zero,
            # so it won't get picked later.
            i = 0 if relevant > irrelevant else slice(1, None)
            ner[s, t, i] = 0
            nen[s, t, i] = 0


if __name__ == '__main__':
    main()
