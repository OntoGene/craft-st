#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


"""
Convert between 4-column and 2-column CoNLL format.
"""


import csv
import json
import argparse
from pathlib import Path
from typing import Tuple, Iterator, Iterable

from abbrevs import AbbrevMapper, TSV_FORMAT


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        'tgt_fmt', choices=('bert', 'craft'),
        help='convert to BERT or CRAFT format?')
    ap.add_argument(
        '-l', '--label-format', choices=('spans', 'ids'), default='spans',
        help='IOBES tags or IDs as labels?')
    ap.add_argument(
        '-t', '--tgt-dir', type=Path, required=True, metavar='PATH',
        help='directory for output files')
    ap.add_argument(
        '-c', '--craft-dir', type=Path, required=True, metavar='PATH',
        help='input directory containing documents in 4-column CoNLL format')
    ap.add_argument(
        '-p', '--pred-dir', type=Path, metavar='PATH',
        help='input directory containing BERT predictions')
    ap.add_argument(
        '-a', '--abbrevs', type=Path, required=True, metavar='PATH',
        help='a JSON file with short/long mappings per document '
             '(format: {"docid": {"xyz": "xtra young zebra", ...}})')
    ap.add_argument(
        '-s', '--splits', type=Path, required=True, metavar='PATH',
        help='a JSON file specifying document IDs for the train/dev/test split')
    ap.add_argument(
        '-S', '--subsets', nargs='+', required=True,
        choices=('train', 'dev', 'test'), help='corpus subset')
    args = ap.parse_args()
    convert(**vars(args))


def convert(tgt_fmt: str, tgt_dir: Path, pred_dir: Path, label_format: str,
            **kwargs) -> None:
    """
    Convert between BERT and CRAFT format.
    """
    docs = _iter_input_docs(**kwargs)
    tgt_dir.mkdir(exist_ok=True)
    if tgt_fmt == 'bert':
        filename = '_'.join(kwargs['subsets']) + '.tsv'
        to_bert_fmt(docs, tgt_dir/filename, label_format)
    elif tgt_fmt == 'craft':
        to_craft_conll(docs, pred_dir, tgt_dir, label_format)
    else:
        raise ValueError(f'unknown target format: {tgt_fmt}')


def _iter_input_docs(craft_dir: Path,
                     abbrevs: Path,
                     splits: Path,
                     subsets: Iterable[str]  # train/test/dev
                    ) -> Iterator[Tuple[str, AbbrevMapper, Path]]:
    with splits.open() as f:
        folds = json.load(f)
    with abbrevs.open() as f:
        docwise_abbrevs = json.load(f)
    for subset in subsets:
        for docid in folds[0][subset]:
            abb = AbbrevMapper(docwise_abbrevs[docid])
            yield docid, abb, craft_dir / f'{docid}.conll'


# CRAFT to BERT
# =============

def to_bert_fmt(docs, tgt_path, label_format):
    """
    Create one long file in 2-column format.
    """
    with tgt_path.open('w', encoding='utf8') as f:
        writer = csv.writer(f, **TSV_FORMAT)
        writer.writerows(_iter_bert_fmt(docs, label_format))


def _iter_bert_fmt(docs, label_format):
    _fmt_label = {
        'spans': _span_label_fmt,
        'ids': _id_label_fmt,
    }[label_format]
    for _, abb, path in docs:
        with path.open(encoding='utf8') as f:
            rows = csv.reader(f, **TSV_FORMAT)
            rows = abb.expand(rows)
            for row in rows:
                if not any(row):
                    yield ()
                    continue
                token, _, _, label, *_ = row
                if len(token) > 50:  # trim overly long DNA sequences
                    token = token[:50]
                label = _fmt_label(label)
                yield token, label


def _span_label_fmt(label):
    return label[0]


def _id_label_fmt(label):
    if label == 'O':
        label = 'O-NIL'
    elif label.startswith(('B-', 'E-', 'S-')):
        label = f'I-{label[2:]}'
    return label


# BERT to CRAFT
# =============

def to_craft_conll(docs, pred_dir, tgt_dir, label_format):
    """
    Split BERT predictions into document-wise 4-column files.
    """
    predicted = _undo_wordpiece(pred_dir, label_format)
    for docid, abb, ref_path in docs:
        tgt_path = tgt_dir / f'{docid}.conll'
        with tgt_path.open('w', encoding='utf8') as f:
            writer = csv.writer(f, **TSV_FORMAT)
            writer.writerows(_iter_conll_fmt(ref_path, abb, predicted))


def _iter_conll_fmt(ref_path, abb, predicted):
    with ref_path.open(encoding='utf8') as f:
        orig = list(csv.reader(f, **TSV_FORMAT))
        expanded = abb.expand(orig)
        merged = _merge_craft_bert(expanded, predicted)
        yield from abb.restore(merged, scored=True)


def _merge_craft_bert(ref_rows, pred_rows):
    for row in ref_rows:
        if not any(row):
            yield ()
            continue
        tok, start, end, *_ = row
        ptok, label, logprob = next(pred_rows, None)  # raise ValueError on early end
        assert tok == ptok or ptok == '[UNK]' or tok.startswith(ptok), \
            f'conflicting tokens: {tok} vs. {ptok}'
        yield tok, start, end, label, logprob


def _undo_wordpiece(pred_dir: Path, label_format: str
                   ) -> Iterator[Tuple[str, str, float]]:
    """Iterate over triples <token, label, logprob>."""
    ctrl_labels = _get_ctrl_labels(label_format)
    tp, lp, pp = (pred_dir/f'{x}_test.txt' for x in ('token', 'label', 'logits'))
    with tp.open(encoding='utf8') as t, lp.open() as l, pp.open() as p:
        previous = None  # type: Tuple[str, str, float]
        for token, label, logits in zip(t, l, p):
            token, label = token.strip(), label.strip()
            if token.startswith('##'):
                # Merge word pieces.
                token = previous[0] + token[2:]
                # Ignore the predictions for this token.
                previous = (token, *previous[1:])
            else:
                # A new word started. Yield what was accumulated.
                if previous is not None:
                    yield previous
                if token in CTRL_TOKENS:
                    # Silently skip control tokens.
                    previous = None
                else:
                    # Regular case.
                    label = ctrl_labels.get(label, label)  # replace with 'O'
                    logprob = max(map(float, logits.split()[1:]))  # best score
                    previous = token, label, logprob
        if previous is not None:
            yield previous
        # Sanity check: all file iterators must be exhausted.
        if any(map(list, (t, l, p))):
            raise ValueError(f'unequal length: {tp} {lp} {pp}')


def _get_ctrl_labels(label_format):
    outside = {
        'spans': 'O',
        'ids': 'O-NIL'
    }[label_format]
    return dict.fromkeys(['[CLS]', '[SEP]', 'X'], outside)

CTRL_TOKENS = ('[CLS]', '[SEP]')


if __name__ == '__main__':
    main()
