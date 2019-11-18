#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


"""
Train and test a model for one entity type of the CRAFT corpus.
"""


import re
import csv
import json
import random
import logging
import argparse
import tempfile
import itertools as it
import subprocess as sp
from pathlib import Path
from functools import wraps
from collections import defaultdict, Counter, namedtuple

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.layers import concatenate as concat, TimeDistributed as td, Masking
from keras.layers import LSTM, Bidirectional
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.utils import Sequence

from abbrevs import AbbrevMapper


BATCH = 32
PRE_EPOCHS = 20  # pretraining epochs
MAX_EPOCHS = 100
# Default vocab size (used without pretrained embeddings).
VOCAB_SIZE = 10000
ALPHABET_SIZE = 200  # character vocabulary
ONTO_ENTRIES = 7000  # random terminology entries per epoch

NER_TAGS = tuple('OBIES')  # outside-label O is at 0
NIL = 'NIL'

TOKEN = re.compile(r'[^\W_]+|[^\w\s]|_')

# CSV flavour for reading and writing TSV files.
# Treat every character literally (including quotes).
TSV_FORMAT = dict(
    lineterminator='\n',  # ignored by csv.reader
    delimiter='\t',
    quotechar=None,
)

# Directory containing this script.
HERE = Path(__file__).parent


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '-f', '--folds', nargs='+', type=int, default=[0], metavar='N',
        help='run which fold(s) of the n-fold cross-validation? '
             '(default: first fold only, ie. 0)')
    ap.add_argument(
        '-s', '--splits', type=read_json, metavar='PATH',
        default=str(HERE/'splits.json'),
        help='a JSON file specifying document IDs for the train/dev/test '
             'split of every fold (default: %(default)s)')
    ap.add_argument(
        '-i', '--input-dir', type=Path, metavar='PATH',
        help='directory with documents in CoNLL format '
             '(4-column TSV with character offsets, '
             'ie. <token, start, end, tag>)')
    ap.add_argument(
        '-t', '--terminology', type=Path, metavar='PATH',
        help='terminology file in Bio Term Hub TSV format '
             '(additional training samples)')
    ap.add_argument(
        '-a', '--abbrevs', type=read_json, metavar='PATH',
        help='a JSON file with short/long mappings per document '
             '(format: {"docid": {"xyz": "xtra young zebra", ...}})')
    ap.add_argument(
        '-o', '--output-dir', type=Path, metavar='PATH',
        help='target directory for the test-set predictions')
    ap.add_argument(
        '-m', '--model-path', type=Path, metavar='PATH',
        help='dump the weights of the Keras model')
    ap.add_argument(
        '-c', '--concept-ids', type=Path, metavar='PATH',
        help='persist the mapping concept->index to disk')
    ap.add_argument(
        '-l', '--log-file', type=Path, metavar='PATH',
        help='write results and some progress information to a log file '
             '(in addition to output on STDOUT/STDERR)')
    ap.add_argument(
        '-w', '--word-vectors', type=lambda p: np.load(p, mmap_mode='r'),
        metavar='PATH',
        help='numpy matrix with pretrained word vectors, '
             'first two rows reserved for <PAD> and <UNK>')
    ap.add_argument(
        '-V', '--vocab', type=read_vocab, metavar='PATH',
        help='vocabulary file, one token per line, corresponding to '
             'the vectors in the given matrix (starting from the third row)')
    ap.add_argument(
        '-A', '--alphabet', type=read_vocab, metavar='PATH',
        help='character vocabulary, one character per line')
    args = ap.parse_args()

    run(args.input_dir.glob('*'),
        onto=args.terminology,
        abbrevs=args.abbrevs,
        folds=args.folds,
        splits=args.splits,
        pred_dir=args.output_dir,
        dumpfn=args.model_path,
        concept_ids=args.concept_ids,
        log_file=args.log_file,
        pre_wemb=args.word_vectors,
        vocab=args.vocab,
        alphabet=args.alphabet)


def run(*args, log_level='INFO', log_file=None, **kwargs):
    """Perform n-fold cross-validation."""
    setup_logging(log_level, log_file)
    try:
        return list(iter_run(*args, **kwargs))
    except Exception:
        logging.exception('Crash!')
        raise


def iter_run(conll_files, vocab=None, onto=None, concept_ids=None,
             abbrevs=None, alphabet=None, **kwargs):
    """Iteratively perform n-fold cross-validation."""
    logging.info('Last commit: %s', get_commit_info('H'))
    logging.info('Commit message: %s', get_commit_info('B'))
    logging.info('Working directory %s', get_wd_state())

    data = Dataset.from_files(conll_files, vocab=vocab, abbrevs=abbrevs,
                              alphabet=alphabet)
    if onto is not None:
        logging.info('Loading ontology %s', onto)
        with Path(onto).open(encoding='utf8') as f:
            data.add_onto(f)

    concept_ids = temp_fallback(concept_ids, suffix='.labels')
    logging.info('Persist concept-label indices to %s', concept_ids)
    with concept_ids.open('w', encoding='utf8') as f:
        print(*data.concept_ids, sep='\n', file=f)

    return _iter_run(data, **kwargs)


def _iter_run(data, folds=range(1), splits=None, dumpfn=None, **kwargs):
    if splits is None:
        # 6-fold CV with a dev set (almost) half the size of the test set.
        splits = fold(sorted(data.docs), 6, dev_ratio=.45)

    for i, docs in enumerate(splits):
        if i in folds:
            logging.info('Running fold %d (%d/%d/%d)',
                         i, *(len(docs[s]) for s in ('train', 'dev', 'test')))
            if dumpfn is not None:
                dumpfn = Path(str(dumpfn).format(fold=i))
            yield run_fold(data, docs, dumpfn=dumpfn, **kwargs)


def run_fold(data, docs, pre_wemb=None, dumpfn=None, **kwargs):
    """Train and evaluate one fold of cross-validation."""
    logging.info('Compiling graph')
    model = build_network(
        pre_wemb, len(data.concept_ids), len(NER_TAGS), data.n_features)
    dumpfn = temp_fallback(dumpfn, suffix='.weights')
    earlystopping = EarlyStoppingFScore((data, docs['dev']), dumpfn,
                                        patience=5)

    try:
        if data.onto:
            logging.info('Start pretraining')
            batches = data.batches([], onto=ONTO_ENTRIES)
            model.fit_generator(batches, epochs=PRE_EPOCHS, shuffle=False)

        logging.info('Start training')
        batches = data.batches(docs['train'])
        model.fit_generator(batches, epochs=MAX_EPOCHS, shuffle=False,
                            callbacks=[earlystopping])
    except KeyboardInterrupt:
        logging.error('Training aborted')  # jump to evaluation

    # Recover from different situations:
    # - if the model was ever saved, load that one (test on the best);
    # - if it was never saved, save it now.
    if Path(dumpfn).exists() and Path(dumpfn).stat().st_size:
        model.load_weights(str(dumpfn))
    else:
        logging.info('Saving model weights to %s', dumpfn)
        model.save_weights(str(dumpfn))

    return run_test(data, docs['test'], model, **kwargs)


def run_test(data, docids, model, pred_dir=None):
    """Make predictions and score them."""
    logging.info('Evaluating on test set')
    if pred_dir is None:
        pred_dir = tempfile.mkdtemp()
    return _run_test(data, docids, model, pred_dir)


def _run_test(data, docids, model, pred_dir=None):
    # Process each article separately to avoid memory problems.
    # Model.predict_generator is tricky because each batch is padded
    # differently (shape mismatch when Keras concatenates).
    scores = [PRF() for _ in range(2)]
    for docid in docids:
        test_x, test_y = data.x_y([docid])
        pred = model.predict(test_x, batch_size=BATCH)
        if pred_dir is not None:
            data.dump_conll(pred_dir, [docid], pred)
        for y, p, s in zip(test_y, pred, scores):
            s.update(**vars(PRF.from_one_hot(y, p)))
    for task, score in zip(('NER', 'NEN'), scores):
        logging.info('%s: %s', task, score)
    return scores


def build_network(pre_wemb, n_concepts, n_spans, n_features=0):
    """Compile the graph with task-specific dimensions."""
    chars = Input(shape=(None, None), dtype='int32')
    char_emb = embedding_layer(voc=ALPHABET_SIZE, dim=50)(chars)
    char_rep = td(Conv1D(filters=50, kernel_size=3, activation='tanh'))(char_emb)
    char_rep = td(GlobalMaxPooling1D())(char_rep)

    words = Input(shape=(None,), dtype='int32')
    word_emb = embedding_layer(matrix=pre_wemb)(words)

    features = [Input(shape=(None, n_concepts)) for _ in range(n_features)]
    word_rep = concat([word_emb, char_rep, *features])

    lstm = LSTM(100, dropout=.1, recurrent_dropout=.1, return_sequences=True)
    mask = Masking(0).compute_mask(chars)
    bilstm = Bidirectional(lstm)(word_rep, mask=mask)

    spans = Dense(n_spans, activation='softmax')(bilstm)
    concepts = Dense(n_concepts, activation='softmax')(concat([bilstm, spans]))

    model = Model(inputs=[words, chars, *features],
                  outputs=[spans, concepts])
    model.compile(optimizer=Adam(lr=1e-3, amsgrad=True),
                  loss='categorical_crossentropy')

    logging.info('embeddings: %(input_dim)d, %(output_dim)d',
                 vars(model.get_layer('embedding_2')))
    logging.info('features: %d', n_features)
    logging.info('span labels: %d', n_spans)
    logging.info('concept labels: %d', n_concepts)

    return model


def embedding_layer(voc=VOCAB_SIZE, dim=50, matrix=None, **kwargs):
    '''A layer for word/character/... embeddings.'''
    if matrix is not None:
        voc, dim = matrix.shape
        kwargs.update(weights=[matrix])
    return Embedding(voc, dim, **kwargs)


def fold(elements, n, dev_ratio=1.):
    """
    Divide elements into n folds.

    For each fold, elements are exhaustively split into
    a <train, dev, test> triple.
    The test sets are contiguous slices of the input sequence;
    concatenating them across folds will restore the original
    input.
    The remainder is divided into a training and a development
    (tuning) set. By default, the dev set has the same size as
    the test set; the dev_ratio parameter allows changing this:
        dev_ratio == len(dev) / len(test)
    """
    div, mod = divmod(len(elements), n)
    n_samples = [div]*n
    for i in range(mod):
        n_samples[i] += 1
    start = 0
    for test_size in n_samples:
        test = elements[start:start+test_size]
        train = elements[start+test_size:] + elements[:start]
        dev_size = round(dev_ratio*test_size)
        dev = train[:dev_size]
        train = train[dev_size:]
        yield dict(train=train, dev=dev, test=test)
        start += test_size


class Dataset:
    """Container for the whole data."""

    def __init__(self, vocab=None, concept_ids=None, abbrevs=None,
                 alphabet=None):
        """Create an empty instance."""
        self.vocab = vocab
        self._alphabet = alphabet
        self.flat = []  # all sentences, original data (words)
        self.docs = {}  # map doc IDs to sentence offsets
        self.onto = []  # indices of the ontology terms
        self._vec = None
        self._concept_ids = concept_ids
        self._index2concept = None
        self.abbrevs = {docid: AbbrevMapper(a)
                        for docid, a in abbrevs.items()} if abbrevs else None

    @classmethod
    def from_files(cls, paths, **kwargs):
        """Construct a populated instance from docs in CoNLL format."""
        paths = list(map(Path, paths))
        logging.info('Loading %d documents from %s...',
                     len(paths), set(str(p.parent) for p in paths))
        ds = cls(**kwargs)
        for p in paths:
            with p.open(encoding='utf8') as f:
                ds.add_doc(p.stem, f)
        logging.info('Loaded %d sentences', len(ds.flat))
        return ds

    def add_doc(self, docid, lines):
        """Add a document in 4-column CoNLL TSV format."""
        if docid in self.docs:
            logging.warning('duplicate document %s (was %d-%d)',
                            docid, *self.docs[docid])
        self.clear_cache()
        offset = len(self.flat)
        filter_ = self.abbrevs[docid].expand if self.abbrevs else None
        self.flat.extend(load_conll(lines, filter_))
        self.docs[docid] = offset, len(self.flat)

    def add_onto(self, lines):
        """Add ontology entries in Bio Term Hub TSV format."""
        self.clear_cache()
        offset = len(self.flat)
        self.flat.extend(load_onto(lines))
        span = range(offset, len(self.flat))
        logging.info('Loaded %d dictionary entries', len(span))
        if not self.onto:
            self.onto = span
        else:
            self.onto = [*self.onto, *span]

    def clear_cache(self):
        """Invalidate the cache whenever self.flat changes."""
        self._vec = None
        self._index2concept = None

    @property
    def vec(self):
        """Numeric representation of all items."""
        if self._vec is None:
            self._vec = vectorised(self.flat, self.vocab,
                                   concept_ids=self._concept_ids,
                                   alphabet=self._alphabet)
            self._concept_ids = self._vec.concept_ids  # keep this up-to-date
        return self._vec

    @property
    def concept_ids(self):
        """Concept labels, ordered by their one-hot encoding."""
        if self._index2concept is None:
            self._index2concept = [None] * len(self.vec.concept_ids)
            for c, i in self.vec.concept_ids.items():
                self._index2concept[i] = c
        return self._index2concept

    @property
    def n_features(self):
        """Number of features."""
        return int(bool(self.vec.features))

    def x_y(self, docids, onto=0):
        """Padded input and output ndarrays."""
        ranges = self._ranges(docids)
        if onto >= len(self.onto):
            ranges.extend(self.onto)
        else:
            ranges.extend(random.sample(self.onto, onto))
        return padded(self.vec, ranges)

    def batches(self, docids, onto=0):
        """A Keras Sequence of padded batches."""
        return PaddedBatches(self.vec, self._ranges(docids), self.onto, onto)

    def _ranges(self, docids):
        return [self.docs[d] for d in docids]

    def dump_conll(self, targetdir, docids, predictions, force_agreement=True):
        """Export predictions in CoNLL format."""
        Path(targetdir).mkdir(parents=True, exist_ok=True)
        for docid, rows in self.iter_conll(docids, predictions, force_agreement):
            path = (Path(targetdir)/docid).with_suffix('.conll')
            logging.info('Exporting predictions to %s', path)
            with path.open('w', encoding='utf8') as f:
                csv.writer(f, **TSV_FORMAT).writerows(rows)

    def iter_conll(self, docids, predictions, force_agreement=True):
        """Iterate over lines in CoNLL format."""
        if force_agreement:
            fix_disagreements(*predictions)
        terms, concepts = (iter(predictions[i].argmax(-1)) for i in (0, 1))
        scores = iter(predictions[0].max(-1) * predictions[1].max(-1))
        for docid in docids:
            rows = self._conll_rows(docid, terms, concepts, scores)
            if self.abbrevs:
                rows = self.abbrevs[docid].restore(rows, scored=True)
            yield docid, rows

    def _conll_rows(self, docid, terms, concepts, scores):
        concept_tags = self.concept_ids
        sentences = select(self.flat, [self.docs[docid]])
        for sent, ts, cs, sc in zip(sentences, terms, concepts, scores):
            tokens = zip(sent, ts, cs, sc)
            for (tok, _, _, start, end, *_), term, conc, score in tokens:
                tag = '{}-{}'.format(NER_TAGS[term], concept_tags[conc])
                yield tok, start, end, tag, score
            yield ()


class PaddedBatches(Sequence):
    """Keras Sequence for padded batches of x and y."""

    def __init__(self, vec, ranges, onto, n_onto, batch_size=BATCH):
        self.vec = vec
        self.onto = onto
        self.indices = list(select(range(len(vec.words)), ranges))
        self.n_onto = min(n_onto, len(onto))
        self.batch_size = batch_size

        self.current = None
        self.on_epoch_end()

    def __getitem__(self, idx):
        selection = self.current[idx*self.batch_size : (idx+1)*self.batch_size]
        return padded(self.vec, selection)

    def __len__(self):
        samples = len(self.indices) + self.n_onto
        div, mod = divmod(samples, self.batch_size)
        return div + bool(mod)

    def on_epoch_end(self):
        """Get an onto sample and shuffle with the corpus data."""
        self.current = self.indices + self._onto_sample()
        random.shuffle(self.current)

    def _onto_sample(self):
        if self.n_onto >= len(self.onto):
            return list(self.onto)
        return random.sample(self.onto, self.n_onto)


def load_conll(lines, filter_=None):
    """Parse verticalised text with char offsets and IOB[ES] tags."""
    rows = csv.reader(lines, **TSV_FORMAT)
    if filter_ is not None:
        rows = filter_(rows)
    for has_content, group in it.groupby(rows, key=any):
        if not has_content:
            continue
        sent = []
        for token, start, end, tag, *feat in group:
            if tag == 'O':
                label, concept = 'O', NIL
            else:
                label, concept = tag.split('-', 1)
            feat = [[NIL] if f == 'O' else f.split('-', 1)[1].split(';')
                    for f in feat]
            sent.append((token, label, concept, int(start), int(end), *feat))
        yield sent


def load_onto(lines):
    """Parse an ontology in Bio Term Hub format."""
    rows = csv.reader(lines, **TSV_FORMAT)
    for _, _, concept, term, _, _ in rows:
        tokens = TOKEN.findall(term)
        tags = ['S'] if len(tokens) == 1 else ['B', *'I'*(len(tokens)-2), 'E']
        yield list(zip(tokens, tags, it.repeat(concept)))


def read_vocab(path, reserved=2):
    """Map vocabulary from a file (one word per line) to its position."""
    with Path(path).open(encoding='utf8') as f:
        return {l.strip(): i for i, l in enumerate(f, reserved)}


def read_json(path):
    """Read a JSON file from disk."""
    with Path(path).open(encoding='utf8') as f:
        return json.load(f)


def index(elements, n, reserved=2):
    """Map the n most frequent elements to their rank."""
    n -= reserved  # default: reserve 0 and 1 for <PAD> and <UNK>
    counts = Counter(elements)
    return {e: i for i, (e, _) in enumerate(counts.most_common(n), reserved)}


def concept_enumerator(*args, **kwargs):
    """An auto-increment defaultdict for mapping concept labels to indices."""
    concept_ids = defaultdict(it.count().__next__, *args, **kwargs)
    _ = concept_ids[NIL]  # make sure NIL has index 0
    return concept_ids


def vectorised(data, vocab=None, vocab_size=None, concept_ids=None,
               alphabet=None):
    """Enumerate tokens, characters, terms, and concepts."""
    logging.info('Vectorising %d sentences', len(data))
    if vocab is None:
        tokens = (tok for sent in data for tok, *_ in sent)
        vocab = index(tokens, vocab_size or VOCAB_SIZE)

    words = [[vocab.get(tok, 1) for tok, *_ in sent] for sent in data]

    if alphabet is None:
        chars = (c for sent in data for tok, *_ in sent for c in tok)
        alphabet = index(chars, ALPHABET_SIZE)
    chars = [[[alphabet.get(c, 1) for c in tok] for tok, *_ in sent]
             for sent in data]

    term_ids = dict(zip(NER_TAGS, it.count()))
    terms = [[term_ids[tag] for _, tag, *_ in sent] for sent in data]
    if concept_ids is None:
        concept_ids = concept_enumerator()
    concepts = [[concept_ids[tag] for _, _, tag, *_ in sent] for sent in data]

    # Include optional dictionary features, while also updating concept_ids.
    if any(len(e) > 5 for sent in data for e in sent):
        # Take the real dict feature for the corpus data,
        # but reuse the label for the pretraining ontology terms.
        features = [[[concept_ids[i] for i in (e[5] if len(e)>5 else [e[2]])]
                     for e in sent]
                    for sent in data]
    else:
        features = []

    return Vec(words, chars, terms, concepts, features, term_ids, concept_ids)

Vec = namedtuple('Vec',
                 'words chars terms concepts features term_ids concept_ids')


def padded(vec, selection=((None, None),)):
    """Prepare x and y as padded ndarrays."""
    sel = list(select(vec.words, selection))
    words = np.zeros((len(sel), max(map(len, sel))), dtype=int)
    for i, sent in enumerate(sel):
        words[i, :len(sent)] = sent

    sel = list(select(vec.chars, selection))
    max_chars = max(len(w) for sent in sel for w in sent)
    chars = np.zeros(words.shape + (max_chars,), dtype=int)
    for i, sent in enumerate(sel):
        for j, word in enumerate(sent):
            chars[i, j, :len(word)] = word

    one_hot = [(vec.terms, vec.term_ids), (vec.concepts, vec.concept_ids)]
    if vec.features:
        one_hot.append((vec.features, vec.concept_ids))
    terms, concepts, *features = (
        to_one_hot((*words.shape, len(ids)), select(ind, selection))
        for ind, ids in one_hot)

    return [words, chars, *features], [terms, concepts]


def to_one_hot(shape, indices):
    """Create a 3D array with one-hot encoding."""
    arr = np.zeros(shape)
    for i, sent in enumerate(indices):
        for j, pos in enumerate(sent):
            if pos is not None:
                arr[i, j, pos] = 1.
    return arr


def select(sequence, selections):
    """Iterate over multiple elements and slices from sequence."""
    for selection in selections:
        if isinstance(selection, int):
            yield sequence[selection]
        else:
            yield from sequence[slice(*selection)]


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


class EarlyStoppingFScore(Callback):
    '''
    Stop training when CR F-score has stopped improving.

    Based on keras.callbacks.EarlyStopping.
    '''

    def __init__(self, val_data, dumpfn, baseline=0, patience=3, min_delta=0):
        super().__init__()

        self.val_data = val_data
        self.dumpfn = str(dumpfn)

        self.patience = patience
        self.min_delta = min_delta

        self.wait = 0
        self.best = baseline

    def on_epoch_end(self, epoch, logs=None):
        logging.info('Epoch %d: evaluate validation set', epoch + 1)
        current = self.evaluate()
        if current - self.min_delta > self.best:
            logging.info('Improved F1 -- saving weights to %s', self.dumpfn)
            self.wait = 0
            self.best = current
            self.model.save_weights(self.dumpfn)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                if epoch > 0:
                    logging.info('Epoch %d: early stopping', epoch + 1)

    def evaluate(self):
        '''
        Compute F1 for the CR layer.
        '''
        scores = _run_test(*self.val_data, self.model)
        return scores[-1].fscore  # F-score for the NEN output


def zerodivision(fallback):
    '''
    Decorator factory.

    If calling func raises a ZeroDivisionError, return fallback instead.
    '''
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ZeroDivisionError:
                return fallback
        return _wrapper
    return _decorator


class PRF:
    """Convenient container for precision, recall, F1."""

    def __init__(self, tp=0, relevant=0, selected=0):
        self.tp = int(tp)
        self.relevant = int(relevant)  # TP + FN
        self.selected = int(selected)  # TP + FP

    @property
    def fp(self):
        """False positives."""
        return self.selected - self.tp

    @property
    def fn(self):
        """False negatives."""
        return self.relevant - self.tp

    def update(self, tp=0, relevant=0, selected=0):
        """Add to TP/rel/sel counts."""
        self.tp += tp
        self.relevant += relevant
        self.selected += selected

    @classmethod
    def from_one_hot(cls, y, pred, neg=0):
        """
        Compute P/R/F1 from one-hot encoded arrays.

        Args:
            y: ground-truth labels
            pred: predicted labels
            neg: index of the negative class
        """
        y, pred = y.argmax(-1), pred.argmax(-1)
        tp = np.sum((y != neg) & (y == pred))
        relevant = np.sum(y != neg)
        selected = np.sum(pred != neg)
        return cls(tp, relevant, selected)

    @property
    @zerodivision(1.)  # no FP -- perfect precision
    def prec(self):
        """Precision: TP/(TP+FP)."""
        return self.tp / self.selected

    @property
    @zerodivision(1.)  # no FN -- perfect recall
    def rec(self):
        """Recall: TP/(TP+FN)."""
        return self.tp / self.relevant

    @property
    @zerodivision(1.)  # no FP, no FN -- no errors
    def fscore(self):
        """F1: 2*P*R/(P+R)."""
        return 2 * self.tp / (self.relevant + self.selected)

    def __str__(self):
        return 'P: {0.prec:.3}, R: {0.rec:.3}, F1: {0.fscore:.3}'.format(self)


def temp_fallback(path, **kwargs):
    """Create a temporary file if path is None."""
    if path is None:
        with tempfile.NamedTemporaryFile(delete=False, **kwargs) as f:
            path = f.name
    return Path(path)


def setup_logging(log_level='INFO', log_file=None):
    """Messages go to console and (optionally) a log file."""
    logging.basicConfig(level=log_level,
                        format='%(asctime)s: %(message)s')
    if log_file is not None:
        logger = logging.getLogger()  # root logger
        handler = logging.FileHandler(str(log_file))
        handler.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(handler)


def get_commit_info(spec):
    '''Get some info about the current git commit.'''
    args = ['git', 'log', '-1', '--pretty=%{}'.format(spec)]
    compl = sp.run(args, stdout=sp.PIPE, cwd=str(HERE))
    if compl.returncode == 0:
        return compl.stdout.decode('utf8').strip()
    return '<no commit info>'


def get_wd_state():
    """Is the working directory clean or not?"""
    args = ['git', 'diff', 'HEAD', '--exit-code']
    compl = sp.run(args, cwd=str(HERE))
    if compl.returncode == 0:
        return 'clean'
    return 'dirty'


if __name__ == '__main__':
    main()
