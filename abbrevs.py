#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


"""
Expand or restore abbreviations in CoNLL format.
"""


import re
import csv
import json
import logging
import argparse
import itertools as it


NIL = 'NIL'
TOKEN = r'[^\W\d_]+|\d+|[^\w\s]|_'
TSV_FORMAT = dict(delimiter='\t', quotechar=None, lineterminator='\n')


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        'method', choices={'expand', 'restore'},
        help='expand short to long forms or restore original short?')
    ap.add_argument(
        'abbrevs', type=argparse.FileType(encoding='utf8'), metavar='ABB',
        help='a JSON file mapping short forms to long forms')
    ap.add_argument(
        '-i', '--input', type=argparse.FileType(encoding='utf8'),
        default='-', metavar='PATH', dest='in_stream',
        help='source TSV (default: read STDIN)')
    ap.add_argument(
        '-o', '--output', type=argparse.FileType('w', encoding='utf8'),
        default='-', metavar='PATH', dest='out_stream',
        help='target TSV (default: write to STDOUT)')
    ap.add_argument(
        '-t', '--token', default=TOKEN, metavar='REGEX',
        help='regular expression for tokenising long forms '
             '(default: %(default)s)')
    args = ap.parse_args()

    with args.in_stream, args.out_stream, args.abbrevs:
        args.abbrevs = json.load(args.abbrevs)
        _process(**vars(args))


def expand(in_stream, out_stream, abbrevs, **kwargs):
    """Expand short forms in TSV lines."""
    _process('expand', in_stream, out_stream, abbrevs, **kwargs)


def restore(in_stream, out_stream, abbrevs, **kwargs):
    """Restore long forms in TSV lines."""
    _process('restore', in_stream, out_stream, abbrevs, **kwargs)


def _process(method, in_stream, out_stream, abbrevs, **kwargs):
    rows = csv.reader(in_stream, **TSV_FORMAT)
    rows = getattr(AbbrevMapper(abbrevs, **kwargs), method)(rows)
    csv.writer(out_stream, **TSV_FORMAT).writerows(rows)


class AbbrevMapper:
    """Converter between short and long forms."""

    def __init__(self, abbrevs, token=TOKEN, memory=None):
        abbr = abbrevs.items() if hasattr(abbrevs, 'items') else list(abbrevs)
        self.short2long = {s: tuple(re.findall(token, l)) for s, l in abbr}
        self.long2short = {l: s for s, l in self.short2long.items()}
        self.memory = memory if memory is not None else {}
        if len(self.short2long) != len(abbr):
            logging.warning('ambiguous short form(s)')
        if len(self.long2short) != len(abbr):
            logging.warning('ambiguous long form(s)')

    def expand(self, rows):
        """Convert occurrences of short forms to long forms."""
        for row in rows:
            if not any(row):
                yield row
                continue
            tok, start, end, label, *feat = row
            try:
                expanded = self.short2long[tok]
            except KeyError:
                yield row
            else:
                self.memory[expanded, start, end] = tok
                labels = self._expand_label(label, len(expanded))
                for tok, label in zip(expanded, labels):
                    yield [tok, start, end, label, *feat]

    def _expand_label(self, label, n):
        tag, label = label[0], label[1:]
        for tag in self._expand_tag(tag, n):
            yield ''.join((tag, label))

    @staticmethod
    def _expand_tag(tag, n, allow_empty=False):
        """
        Expand a single sequence tag to many.

        This method is compatible with IO, IOB, and IOBES.

        Examples (n=4):
            I -> IIII
            O -> OOOO
            B -> BIII
            E -> IIIE
            S -> BIIE

        Any other tag is simply repeated n times.
        """
        if n < 1:
            if allow_empty:
                return
            raise ValueError('zero-length label sequence')
        if tag == 'B':
            yield 'B'
            yield from it.repeat('I', n-1)
        elif tag == 'E':
            yield from it.repeat('I', n-1)
            yield 'E'
        elif tag == 'S' and n > 1:
            yield 'B'
            yield from it.repeat('I', n-2)
            yield 'E'
        else:
            yield from it.repeat(tag, n)

    def restore(self, rows, scored=False):
        """Reinsert original short forms for the long forms."""
        for offsets, grouped in it.groupby(rows, key=lambda r: r[1:3]):
            grouped = list(grouped)
            if self._needs_merging(grouped, offsets):
                yield self._merge_rows(grouped, *offsets, scored)
            else:
                yield from grouped

    @staticmethod
    def _needs_merging(grouped, offsets):
        if not any(offsets):  # blank line(s)
            return False
        if len(grouped) > 1:  # multi-token expansion
            return True
        tok, start, end = grouped[0][:3]
        if int(end)-int(start) != len(tok):  # single-token expansion
            return True
        return False

    def _merge_rows(self, rows, start, end, scores=None):
        toks, _, _, labels, *feat = zip(*rows)
        if scores:
            scores = list(map(float, feat[-1]))  # scores are in the last column
        try:
            short = self.memory[toks, start, end]
        except KeyError:
            short = self.long2short[toks]
        label = self._merge_labels(labels, short, scores)
        feat = (';'.join(map(str, uniq(f))) for f in feat)
        return [short, start, end, label, *feat]

    def _merge_labels(self, labels, short, scores):
        tags = ''.join(uniq(l[0] for l in labels))
        all_ids = [l[2:] for l in labels]
        ids = list(uniq(all_ids))
        if len(ids) > 1:  # remove NILs and check again
            ids = list(filter(NIL.__ne__, ids))
        if len(ids) > 1:
            if scores:
                id_ = all_ids[argmax(scores)]
            else:
                logging.warning('multiple ids for %s: %s', short, ids)
                id_ = ';'.join(ids)
        else:
            (id_,) = ids
        return '-'.join((self._merge_tags(tags), id_))

    @staticmethod
    def _merge_tags(tags):
        if len(tags) == 1:
            return tags
        rtags = tags.replace('O', '')
        if rtags == 'S' or (rtags.startswith('B') and rtags.endswith('E')):
            return 'S'
        if rtags.startswith('B'):
            return 'B'
        if rtags.endswith('E'):
            return 'E'
        elif 'I' in rtags:
            return 'I'
        else:
            logging.warning('unusual tag sequence: %s', tags)
            return rtags[0]


def uniq(iterable):
    """Delete repeated adjacent elements in iterable."""
    return (u for u, _ in it.groupby(iterable))


def argmax(sequence):
    """Determine the index of the largest element in sequence."""
    return max(range(len(sequence)), key=sequence.__getitem__)


if __name__ == '__main__':
    main()
