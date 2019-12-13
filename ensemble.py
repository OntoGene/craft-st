#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


"""
Merge individual predictions into ensemble output.
"""


import io
import csv
import sys
import logging
from pathlib import Path
from typing import List, Dict, Iterable

import numpy as np

import train


FOLDS = 6
TESTSET_SIZE = 30

# Hard-coded paths!
HERE = Path(__file__).parent

CORPUS = HERE / 'labeled.feat'
SPLITS = HERE / 'splits.subm.json'
VOCAB = HERE / 'vocab.res.txt'
ALPHABET = HERE / 'alphabet.txt'
ABBREVS = HERE / 'abbrevs.json'
WVECTORS = HERE / 'w2v200.res.npy'


def main():
    '''
    Run as script.
    '''
    etype = sys.argv[1]
    srcdir = Path(sys.argv[2])
    tgtdir = Path(sys.argv[3])
    agreement = sys.argv[4] if len(sys.argv) > 4 else 'mutual'
    train.setup_logging()
    merge(etype, srcdir, tgtdir, agreement)


def merge(etype: str, srcdir: Path, tgtdir: Path, pick_best: bool = True,
          agreement: str = 'mutual'):
    """Pick the best models and merge their predictions."""
    splits = train.read_json(SPLITS)
    result_files = sorted(
        srcdir.glob('run*/{}_results.tsv'.format(etype.lower())))
    if pick_best and result_files:
        runs = pick_runs_per_fold(result_files, splits)
        logging.info('best runs: %s', runs)
        models = [srcdir/'models'/'{}.fold-{}.run-{}.h5'.format(etype, f, r)
                  for f, r in enumerate(runs)]
        assert len(models) == FOLDS, \
            'missing folds, found only {}'.format(len(models))
    else:
        models = list(srcdir.glob('models/{}.fold-*.run-*.h5'.format(etype)))
        if pick_best:
            logging.warning('no result files found for picking best run, '
                            'using all %d fold/runs', len(models))

    conll_files = (CORPUS/etype).glob('*')
    labels = get_labels(models)
    vocab = train.read_vocab(VOCAB)
    alphabet = train.read_vocab(ALPHABET)
    abbrevs = train.read_json(ABBREVS)
    data = train.Dataset.from_files(conll_files, vocab=vocab, concept_ids=labels,
                                    abbrevs=abbrevs, alphabet=alphabet)
    ensemble = Ensemble(models, emb=WVECTORS, n_concepts=len(labels),
                        n_spans=len(train.NER_TAGS), n_features=data.n_features)

    docs = splits[0]['test']
    assert len(docs) == TESTSET_SIZE, 'missing test documents'
    for docid in docs:
        x, _ = data.x_y([docid])
        scores = ensemble.predict(x)
        data.dump_conll(tgtdir/etype, [docid], scores, agreement)


def pick_runs_per_fold(result_files: Iterable[Path],
                       splits: Iterable[Dict[str, List[str]]]) -> List[int]:
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
    return [_pick_run(scores) for scores in zip(*results)]


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
        logging.warning('picking unclear: %d:%d (SER: %g:%g, F1: %g:%g)',
                        winner_ser+1, winner_f1+1,
                        results[winner_ser].ser(), results[winner_f1].ser(),
                        results[winner_ser].f1(), results[winner_f1].f1())
    return winner_f1 + 1


class Scorer:
    """Aggregator for micro SER/F1."""
    def __init__(self):
        self.docs = []
        self.counts = dict.fromkeys(
            ['substitutions', 'insertions', 'deletions', 'matches',
             'ref-count', 'prediction-count'],
            0.)

    def update(self, row):
        """Update counts with a csv.DictReader row."""
        self.docs.append(row['#document-id'])
        for k in self.counts:
            self.counts[k] += float(row[k])

    def f1(self):
        """F-Score."""
        if not self.docs:
            return float('-inf')

        c = self.counts
        try:
            return 2 * c['matches'] / (c['ref-count'] + c['prediction-count'])
        except ZeroDivisionError:  # |ref| == |pred| == 0 -> correct
            return 1.

    def ser(self):
        """Slot error rate."""
        if not self.docs:
            return float('inf')

        c = self.counts
        numerator = c['substitutions'] + c['insertions'] + c['deletions']
        try:
            return numerator / c['ref-count']
        except ZeroDivisionError:  # correct (0) if |ins| == |subst| == 0
            return float(bool(numerator))


class Ensemble:
    """Wrapper for using multiple models in the same graph."""

    def __init__(self, dumps, emb, **kwargs):
        pre_wemb = np.load(str(emb), mmap_mode='r')
        self.graph = train.build_network(pre_wemb, **kwargs)
        self.weights = list(self._read_weights(dumps))

    @staticmethod
    def _read_weights(paths):
        for p in paths:
            with Path(p).open('rb') as f:
                yield io.BytesIO(f.read())

    def predict(self, x, batch_size=train.BATCH):
        """Predict, average and fix disagreement."""
        logging.debug('Merge predictions')
        # Compute average scores, incrementally.
        ner, nen = None, None
        for t, c in self.iter_predict(x, batch_size):
            try:
                ner += t
                nen += c
            except TypeError:
                ner = t
                nen = c
        ner /= len(self.weights)
        nen /= len(self.weights)
        return ner, nen

    def iter_predict(self, x, batch_size=train.BATCH):
        """Iterate over predictions from all models."""
        for i, weights in enumerate(self.weights, start=1):
            logging.debug('Predicting (%d/%d)', i, len(self.weights))
            self.graph.load_weights(weights)
            yield self.graph.predict(x, batch_size=batch_size)


def get_labels(model_paths):
    """Get the labels for all models. They need to be identical."""
    labels = None
    for path in model_paths:
        label_path = path.resolve().with_suffix('.labels')
        tmp = train.read_vocab(label_path, reserved=0)
        if labels is None:
            labels = tmp
        else:
            assert tmp == labels, 'differing labels'
    return labels


if __name__ == '__main__':
    main()
