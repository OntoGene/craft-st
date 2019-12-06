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
        'tgt_fmt', metavar='TARGET_FORMAT', choices=('bert', 'craft'),
        help='convert to BERT or CRAFT format?')
    ap.add_argument(
        '-t', '--tgt-dir', type=Path, required=True, metavar='PATH',
        help='directory for output files')
    ap.add_argument(
        '-i', '--craft-dir', type=Path, required=True, metavar='PATH',
        help='input directory containing documents in 4-column CoNLL format')
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


def convert(tgt_fmt: str, tgt_dir: Path, **kwargs) -> None:
    """
    Convert between BERT and CRAFT format.
    """
    docs = _iter_input_docs(**kwargs)
    tgt_dir.mkdir(exist_ok=True)
    if tgt_fmt == 'bert':
        filename = '_'.join(kwargs['subsets']) + '.tsv'
        to_bert_fmt(docs, tgt_dir/filename)
    elif tgt_fmt == 'craft':
        to_craft_conll(docs, tgt_dir)
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


def to_bert_fmt(docs, tgt_path):
    """
    Create one long file in 2-column format.
    """
    with tgt_path.open('w', encoding='utf8') as f:
        writer = csv.writer(f, **TSV_FORMAT)
        writer.writerows(_iter_bert_fmt(docs))


def _iter_bert_fmt(docs):
    for _, abb, path in docs:
        with path.open(encoding='utf8') as f:
            rows = csv.reader(f, **TSV_FORMAT)
            rows = abb.expand(rows)
            for row in rows:
                if not any(row):
                    yield row
                    continue
                token, _, _, label, *_ = row
                if label == 'O':
                    label = 'O-NIL'
                elif label.startswith(('B-', 'E-', 'S-')):
                    label = f'I-{label[2:]}'
                yield token, label


def to_craft_conll(docs, tgt_dir):
    """
    Split BERT predictions into document-wise 4-column files.
    """


if __name__ == '__main__':
    main()
