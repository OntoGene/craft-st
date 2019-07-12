#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


"""
Parse and filter abbreviations detected by Ab3P.
"""


import json
import argparse
from pathlib import Path
from collections import defaultdict


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        'documents', nargs='+', type=Path, metavar='DOC',
        help='paths to the Ab3P output files')
    ap.add_argument(
        '-o', '--output', type=argparse.FileType('w', encoding='utf8'),
        metavar='PATH', default='-',
        help='destination for the resulting JSON file '
             '(default: write to STDOUT)')
    args = ap.parse_args()

    with args.output as f:
        json.dump(collect_abbrevs(args.documents), f, indent=2)


def collect_abbrevs(paths):
    """Get abbreviations for multiple documents."""
    abbrevs = {}
    for path in map(Path, paths):
        with path.open(encoding='utf8') as f:
            abbrevs[path.stem] = get_abbrevs(f)
    return abbrevs


def get_abbrevs(lines):
    """Get the best short-long pairs from Ab3P's output."""
    abbrevs = defaultdict(lambda: defaultdict(set))

    # Collect all and index by short form and normalised long form.
    for short, long_, pprec in parse_ab3p(lines):
        if pprec < .9 or short.islower() or len(short) < 2:
            continue
        normalised = long_.lower().replace('-', ' ')
        abbrevs[short][normalised].add((pprec, long_))

    # Keep only one long form per short form.
    # Disambiguate with max:
    # - sort by pseudo-precision
    # - break ties by preferring lower-case over upper-case
    for short, long_ in abbrevs.items():
        # Resolve case/hyphenation variants.
        for normalised, variants in long_.items():
            long_[normalised] = max(variants)
        # Resolve true ambiguities.
        abbrevs[short] = max(long_.values())[1]

    return dict(abbrevs)  # discard default factory


def parse_ab3p(lines):
    """Extract candidates from Ab3P's output."""
    for line in lines:
        if line.startswith('  '):
            short, long, pprec = line.strip().split('|')
            yield short, long, float(pprec)


if __name__ == '__main__':
    main()
