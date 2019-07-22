#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


"""
Create predictions for the dev and test set.

Predictions for the dev set (test set of the cross validation)
are written in CoNLL format.
Predictions for the test (30 reserved articles for submission)
are exported as raw softmax matrices in .npz format.

On collisions, an auto-incremented run number is created.
"""


import logging
import argparse
import itertools as it
from pathlib import Path

import numpy as np

import train


# Hard-coded paths!
HERE = Path(__file__).parent

TGTDIR = HERE / 'fpred'
CORPUS = HERE / 'labeled.feat'
SPLITS = HERE / 'splits.subm.json'
VOCAB = HERE / 'vocab.res.txt'
ABBREVS = HERE / 'abbrevs.json'
WVECTORS = HERE / 'w2v200.res.npy'


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(description=predict.__doc__)
    ap.add_argument(
        'model_path', type=Path, metavar='MODEL',
        help='model weights in H5 format')
    ap.add_argument(
        '-t', '--targetdir', type=Path, metavar='PATH', default=TGTDIR,
        help='target directory for predictions (default: %(default)s)')
    args = ap.parse_args()
    predict(**vars(args))


def predict(model_path: Path, targetdir: Path):
    """Create predictions for dev and test (submission)."""
    train.setup_logging()
    logging.info('Loading model %s', model_path)
    label_path = model_path.with_suffix('.labels')
    data, etype, fold, docs = load_data(label_path)
    model = load_model(model_path, data)
    subm_dir, n_run = _mk_subm_dir(etype, fold, targetdir)

    dev_dir = targetdir / 'run{}'.format(n_run) / etype
    train.run_test(data, docs['dev'], model, dev_dir)

    _export_pred(data, docs['test'], model, subm_dir)
    (subm_dir/'labels').symlink_to(label_path.resolve())


def load_data(label_path: Path):
    """Load data from default locations."""
    etype, fold = _spec_from_filename(label_path.name)
    conll_files = (CORPUS/etype).glob('*')
    vocab = train.read_vocab(VOCAB)
    labels = train.read_vocab(label_path, reserved=0)
    abbrevs = train.read_json(ABBREVS)
    data = train.Dataset.from_files(conll_files, vocab=vocab, concept_ids=labels,
                                    abbrevs=abbrevs)
    docs = train.read_json(SPLITS)[fold]
    return data, etype, fold, docs


def load_model(model_path: Path, data: train.Dataset):
    """Build the graph and load weights from disk."""
    pre_wemb = np.load(str(WVECTORS), mmap_mode='r')
    model = train.build_network(
        pre_wemb, len(data.concept_ids), len(train.NER_TAGS), data.n_features)
    model.load_weights(str(model_path.with_suffix('.weights')))
    return model


def _export_pred(data, docids, model, subm_dir):
    for docid in docids:
        path = str(subm_dir/'{}.npz'.format(docid))
        logging.info('Exporting prediction vectors to %s', path)
        test_x, _ = data.x_y([docid])
        ner, nen = model.predict(test_x, batch_size=train.BATCH)
        np.savez(path, ner=ner, nen=nen)


def _spec_from_filename(filename):
    """
    Extract "CHEBI_EXT" and `3` from something like
    "subm.CHEBI_EXT-fold-3.efeat.c.onto.20190721-135239.h5".
    """
    etype, _, fold = filename.split('.')[1].split('-')
    return etype, int(fold)


def _mk_subm_dir(etype, fold, targetdir):
    for n_run in it.count(1):
        path = targetdir/'subm'/'{}.fold-{}.run-{}'.format(etype, fold, n_run)
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            pass
        else:
            break
    return path, n_run


if __name__ == '__main__':
    main()
