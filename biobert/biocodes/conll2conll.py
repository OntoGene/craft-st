#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


"""
Convert between 4-column and 2-column CoNLL format.
"""


import csv
import json
import pickle
import argparse
from math import exp
from pathlib import Path
from collections import namedtuple
from typing import List, Tuple, Iterator, Iterable

from abbrevs import AbbrevMapper, TSV_FORMAT


NIL = 'NIL'


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
        help='IOBES tags or IDs as labels? (ignored for CRAFT output) '
             '(default: %(default)s)')
    ap.add_argument(
        '-t', '--tgt-dir', type=Path, required=True, metavar='PATH',
        help='directory for output files')
    ap.add_argument(
        '-c', '--craft-dir', type=Path, required=True, metavar='PATH',
        help='input directory containing documents in 4-column CoNLL format')
    ap.add_argument(
        '-p', '--span-dir', type=Path, metavar='PATH',
        help='input directory containing BERT span predictions')
    ap.add_argument(
        '-i', '--id-dir', type=Path, metavar='PATH',
        help='input directory containing BERT ID predictions')
    ap.add_argument(
        '-m', '--merge-strategy', metavar='STRATEGY', default='ids-first',
        choices=('spans-only', 'ids-only', 'spans-first', 'ids-first',
                 'spans-alone'),
        help='strategy for span/ID predictions (default: %(default)s)')
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


def convert(tgt_fmt: str, tgt_dir: Path,
            label_format: str = 'spans',  # bert2craft only
            span_dir: Path = None, id_dir: Path = None,  # craft2bert only
            merge_strategy: str = 'ids-first',           # craft2bert only
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
        to_craft_conll(docs, tgt_dir, span_dir, id_dir, merge_strategy)
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

def to_craft_conll(docs, tgt_dir, span_dir=None, id_dir=None,
                   merge_strategy='ids-first'):
    """
    Split BERT predictions into document-wise 4-column files.
    """
    with PredictionMerger(span_dir, id_dir, merge_strategy) as predictions:
        for docid, abb, ref_path in docs:
            tgt_path = tgt_dir / f'{docid}.conll'
            with tgt_path.open('w', encoding='utf8') as f:
                writer = csv.writer(f, **TSV_FORMAT)
                writer.writerows(predictions.iter_merge(ref_path, abb))


class PredictionMerger:
    """Handler for iteratively joining span/ID predictions."""

    def __init__(self, span_dir: Path = None, id_dir: Path = None,
                 merge_strategy: str = 'ids-first'):
        self.spans = (self._get_predictions(span_dir, 'spans')
                      if merge_strategy != 'ids-only' else None)
        self.ids = (self._get_predictions(id_dir, 'ids')
                    if merge_strategy not in ('spans-only', 'spans-alone')
                    else None)
        method_name = f'_next_label_{merge_strategy}'.replace('-', '_')
        self._next_label = getattr(self, method_name)

    @staticmethod
    def _get_predictions(pred_dir, fmt):
        iter_pred = _undo_wordpiece(pred_dir, fmt)
        with open(pred_dir/'label2id.pkl', 'rb') as f:
            label2id = pickle.load(f)
            if fmt == 'ids':  # remove the (redundant) O-/I- prefix
                label2id = {l.split('-', 1)[-1]: i for l, i in label2id.items()}
        return PredictionInfo(iter_pred, label2id)

    def close(self):
        """Make sure all files are closed."""
        for fmt in (self.spans, self.ids):
            if fmt is not None:
                if next(fmt.pred, None) is not None:
                    raise ValueError('left-over predictions!')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False

    def iter_merge(self, ref_path, abb):
        """Iterate over merged rows."""
        with ref_path.open(encoding='utf8') as f:
            orig = csv.reader(f, **TSV_FORMAT)
            expanded = abb.expand(orig)
            merged = self._merge(expanded)
            yield from abb.restore(merged, scored=True)

    def _merge(self, ref_rows):
        for row in ref_rows:
            if not any(row):
                yield ()
                continue
            tok, start, end, _, feat = row
            feat = NIL if feat == 'O' else min(feat.split('-', 1)[1].split(';'))
            label, logprob = self._next_label(tok, feat)
            yield tok, start, end, label, logprob

    def _next_label_spans_alone(self, ref_tok, _):
        tag, logits = self._next_prediction(self.spans, ref_tok)
        # Append dummy ID labels in order for abbrev restoration and
        # conll2standoff conversion to work properly.
        tag += '-NIL' if tag == 'O' else '-MISC'
        return tag, exp(max(logits))

    def _next_label_spans_only(self, ref_tok, feat):
        tag, logits = self._next_prediction(self.spans, ref_tok)
        score = max(logits)
        if tag != 'O' and feat != NIL:
            label = f'{tag}-{feat}'
        else:
            label = f'O-{NIL}'
            if tag != 'O':  # prediction corrected due to missing feature
                score = logits[self.spans.label2id['O']]
        return label, exp(score)

    def _next_label_ids_only(self, ref_tok, _):
        label, logits = self._next_prediction(self.ids, ref_tok)
        return label, exp(max(logits))

    def _next_label_spans_first(self, ref_tok, feat):
        return self._next_label_both(ref_tok, feat, spans_first=True)

    def _next_label_ids_first(self, ref_tok, feat):
        return self._next_label_both(ref_tok, feat, spans_first=False)

    def _next_label_both(self, ref_tok, feat, spans_first):
        span, logits_span = self._next_prediction(self.spans, ref_tok)
        id_, logits_id = self._next_prediction(self.ids, ref_tok)
        id_ = id_.split('-', 1)[1]  # strip leading I/O tag
        if span != 'O':
            if feat != NIL and (spans_first or id_ == NIL):
                id_ = feat
            elif id_ == NIL:
                span = 'O'
        elif id_ != NIL:
            span = max('BIES', key=lambda t: logits_span[self.spans.label2id[t]])
        label = f'{span}-{id_}'
        score = logits_span[self.spans.label2id[span]]
        try:
            score += logits_id[self.ids.label2id[id_]]
        except KeyError:             # feat might be an unknown label,
            score += min(logits_id)  # back off to the lowest score of all
        return label, exp(score)

    def _next_prediction(self, pred, ref_tok):
        try:
            pred_tok, label, logits = next(pred.pred)
        except StopIteration:
            raise ValueError('predictions exhausted early!')
        self._assert_same_token(ref_tok, pred_tok)
        return label, logits

    @staticmethod
    def _assert_same_token(ref_tok, pred_tok):
        if ref_tok == pred_tok:  # regular case
            return
        if pred_tok == '[UNK]':  # rare unknown token
            return
        if len(ref_tok) > 50 and ref_tok.startswith(pred_tok):  # long DNA seq.
            return
        raise ValueError(f'conflicting tokens: {ref_tok} vs. {pred_tok}')


PredictionInfo = namedtuple('PredictionInfo', 'pred label2id')


def _undo_wordpiece(pred_dir: Path, label_format: str
                   ) -> Iterator[Tuple[str, str, List[float]]]:
    """Iterate over triples <token, label, logits>."""
    ctrl_labels = _get_ctrl_labels(label_format)
    tp, lp, pp = (pred_dir/f'{x}_test.txt' for x in ('token', 'label', 'logits'))
    with tp.open(encoding='utf8') as t, lp.open() as l, pp.open() as p:
        previous = None  # type: Tuple[str, str, List[float]]
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
                    logits = list(map(float, logits.split()))
                    previous = token, label, logits
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
