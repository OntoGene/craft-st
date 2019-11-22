#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


"""
Create predictions for the dev set.

Predictions for the dev set (test set of the cross validation)
are written in CoNLL format.

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
ALPHABET = HERE / 'alphabet.txt'
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
    ap.add_argument(
        '-g', '--agreement', metavar='STRATEGY', default='none',
        choices=train.Dataset.agreement_strategies,
        help='strategy for enforcing agreement in case of contradicting '
             'predictions for NER and NEN (default: %(default)s)')
    args = ap.parse_args()
    predict(**vars(args))


def predict(model_path: Path, targetdir: Path, agreement: str = 'mutual'):
    """Create predictions for the dev set."""
    train.setup_logging()
    logging.info('Loading model %s', model_path)
    etype, fold = _spec_from_filename(model_path.name)
    symlink, n_run = _model_symlink(etype, fold, targetdir/'models',
                                    model_path.resolve())
    try:
        data, docs = load_data(model_path, etype, fold)
        model = load_model(model_path, data)
        dev_dir = targetdir / 'run{}'.format(n_run) / etype
        train.run_test(data, docs['dev'], model, dev_dir, agreement)
    except Exception:
        symlink.unlink()
        raise


def load_data(model_path: Path, etype: str, fold: int):
    """Load data from default locations."""
    conll_files = (CORPUS/etype).glob('*')
    vocab = train.read_vocab(VOCAB)
    alphabet = train.read_vocab(ALPHABET)
    labels = train.read_vocab(model_path.with_suffix('.labels'), reserved=0)
    abbrevs = train.read_json(ABBREVS)
    data = train.Dataset.from_files(conll_files, vocab=vocab, concept_ids=labels,
                                    abbrevs=abbrevs, alphabet=alphabet)
    docs = train.read_json(SPLITS)[fold]
    return data, docs


def load_model(model_path: Path, data: train.Dataset):
    """Build the graph and load weights from disk."""
    pre_wemb = np.load(str(WVECTORS), mmap_mode='r')
    model = train.build_network(
        pre_wemb, len(data.concept_ids), len(train.NER_TAGS), data.n_features)
    model.load_weights(str(model_path))
    return model


def _spec_from_filename(filename):
    """
    Extract "CHEBI_EXT" and `3` from something like
    "subm.CHEBI_EXT-fold-3.efeat.c.onto.20190721-135239.h5".
    """
    etype, _, fold = filename.split('.')[1].split('-')
    return etype, int(fold)


def _model_symlink(etype, fold, targetdir, model_path):
    targetdir.mkdir(exist_ok=True)
    for n_run in it.count(1):
        path = targetdir/'{}.fold-{}.run-{}.h5'.format(etype, fold, n_run)
        try:
            path.symlink_to(model_path)
        except FileExistsError:
            if path.resolve() == model_path:
                raise RuntimeError('model already covered: {}'
                                   .format(model_path))
        else:
            break
    return path, n_run


if __name__ == '__main__':
    main()
