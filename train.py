#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2019


"""
Train and test a model for one entity type of the CRAFT corpus.
"""


import re
import csv
import random
import logging
import argparse
import tempfile
import itertools as it
from pathlib import Path
from functools import wraps
from collections import defaultdict, Counter, namedtuple

import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.layers import concatenate as concat, TimeDistributed as td, Masking
from keras.layers import LSTM, Bidirectional
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.utils import Sequence


BATCH = 32
MAX_EPOCHS = 100
# Default vocab size (used without pretrained embeddings).
VOCAB_SIZE = 10000
ALPHABET_SIZE = 200  # character vocabulary
ONTO_ENTRIES = 5000  # random terminology entries per epoch

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


def main():
    '''
    Run as script.
    '''
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '-f', '--folds', nargs='+', type=int, default=[0], choices=range(6),
        metavar='{0..5}',
        help='run which folds of the predefined 6-fold cross-validation?')
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
        '-o', '--output-dir', type=Path, metavar='PATH',
        help='target directory for the test-set predictions')
    ap.add_argument(
        '-m', '--model-path', type=Path, metavar='PATH',
        help='dump the Keras model in H5 format')
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
    args = ap.parse_args()

    run(args.input_dir.glob('*'),
        onto=args.terminology,
        folds=args.folds,
        pred_dir=args.output_dir,
        dumpfn=args.model_path,
        log_file=args.log_file,
        pre_wemb=args.word_vectors,
        vocab=args.vocab)


def run(*args, log_level='INFO', log_file=None, **kwargs):
    """Perform 6-fold cross-validation."""
    setup_logging(log_level, log_file)
    return list(iter_run(*args, **kwargs))


def iter_run(conll_files, folds=range(1), vocab=None, onto=None, **kwargs):
    """Iteratively perform 6-fold cross-validation."""
    data = Dataset.from_files(conll_files, vocab=vocab)
    if onto is not None:
        with Path(onto).open(encoding='utf8') as f:
            data.add_onto(f)

    for i, docs in enumerate(fold(sorted(data.docs), 6, dev_ratio=.45)):
        if i in folds:
            yield run_fold(data, docs, **kwargs)


def run_fold(data, docs, pre_wemb=None, dumpfn=None, pred_dir=None):
    """Train and evaluate one fold of cross-validation."""
    train, dev, test = docs

    logging.info('Compiling graph')
    model = build_network(
        pre_wemb, len(data.concept_ids), len(NER_TAGS), data.n_features)
    if dumpfn is None:
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            dumpfn = f.name
    earlystopping = EarlyStoppingFScore(data.x_y(dev), dumpfn)

    logging.info('Start training')
    try:
        batches = data.batches(train, onto=ONTO_ENTRIES)
        model.fit_generator(batches, epochs=MAX_EPOCHS, shuffle=False,
                            callbacks=[earlystopping])
    except KeyboardInterrupt:
        logging.info('Training aborted')  # jump to evaluation

    logging.info('Evaluating on test set')
    if Path(dumpfn).exists() and Path(dumpfn).stat().st_size:
        model = load_model(str(dumpfn))
    else:
        model.save(str(dumpfn))
    test_x, test_y = data.x_y(test)
    pred = model.predict(test_x, batch_size=BATCH)
    data.dump_conll(pred_dir or tempfile.mkdtemp(), test, pred)
    scores = [PRF.from_one_hot(y, p) for y, p in zip(test_y, pred)]
    for task, score in zip(('NER', 'NEN-1', 'NEN-2'), scores):
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
    word_rep = concat([word_emb, char_rep])

    lstm = LSTM(100, dropout=.1, recurrent_dropout=.1, return_sequences=True)
    mask = Masking(0).compute_mask(chars)
    bilstm = Bidirectional(lstm)(word_rep, mask=mask)

    features = [Input(shape=(None, n_concepts)) for _ in range(n_features)]
    if features:
        bilstm = concat([bilstm, *features])

    concepts_aux = Dense(n_concepts, activation='softmax')(bilstm)
    spans = Dense(n_spans, activation='softmax')(concat([bilstm, concepts_aux]))
    concepts = Dense(n_concepts, activation='softmax')(concat([bilstm, spans]))

    model = Model(inputs=[words, chars, *features],
                  outputs=[spans, concepts_aux, concepts])
    model.compile(optimizer=Adam(lr=1e-3, amsgrad=True),
                  loss='categorical_crossentropy')
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
        yield train, dev, test
        start += test_size


class Dataset:
    """Container for the whole data."""

    def __init__(self, vocab=None):
        """Create an empty instance."""
        self.vocab = vocab
        self.flat = []  # all sentences, original data (words)
        self.docs = {}  # map doc IDs to sentence offsets
        self.onto = []  # indices of the ontology terms
        self._vec = None
        self._index2concept = None

    @classmethod
    def from_files(cls, paths, **kwargs):
        """Construct a populated instance from docs in CoNLL format."""
        paths = list(map(Path, paths))
        logging.info('Loading %d documents...', len(paths))
        ds = cls(**kwargs)
        for p in paths:
            with p.open(encoding='utf8') as f:
                ds.add_doc(p.stem, f)
        return ds

    def add_doc(self, docid, lines):
        """Add a document in 4-column CoNLL TSV format."""
        if docid in self.docs:
            logging.warning('duplicate document %s (was %d-%d)',
                            docid, *self.docs[docid])
        self.clear_cache()
        offset = len(self.flat)
        self.flat.extend(load_conll(lines))
        self.docs[docid] = offset, len(self.flat)

    def add_onto(self, lines):
        """Add ontology entries in Bio Term Hub TSV format."""
        self.clear_cache()
        offset = len(self.flat)
        self.flat.extend(load_onto(lines))
        span = range(offset, len(self.flat))
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
            self._vec = vectorised(self.flat, self.vocab)
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

    def dump_conll(self, targetdir, docids, predictions):
        """Export predictions in CoNLL format."""
        logging.info('Exporting predictions to %s', targetdir)
        for docid, rows in self.iter_conll(docids, predictions):
            path = (Path(targetdir)/docid).with_suffix('.conll')
            with path.open('w', encoding='utf8') as f:
                csv.writer(f, **TSV_FORMAT).writerows(rows)

    def iter_conll(self, docids, predictions):
        """Iterate over lines in CoNLL format."""
        terms, concepts = (iter(predictions[i].argmax(-1)) for i in (0, 2))
        for docid in docids:
            yield docid, self._conll_rows(docid, terms, concepts)

    def _conll_rows(self, docid, terms, concepts):
        concept_tags = self.concept_ids
        sentences = select(self.flat, [self.docs[docid]])
        for sent, ts, cs in zip(sentences, terms, concepts):
            for (tok, _, _, start, end, *_), term, conc in zip(sent, ts, cs):
                tag = '{}-{}'.format(NER_TAGS[term], concept_tags[conc])
                yield tok, start, end, tag
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
            return self.onto
        return random.sample(self.onto, self.n_onto)


def load_conll(lines):
    """Parse verticalised text with char offsets and IOB[ES] tags."""
    rows = csv.reader(lines, **TSV_FORMAT)
    for has_content, group in it.groupby(rows, key=any):
        if not has_content:
            continue
        sent = []
        for token, start, end, tag, *feat in group:
            if tag == 'O':
                label, concept = 'O', NIL
            else:
                label, concept = tag.split('-', 1)
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


def index(elements, n, reserved=2):
    """Map the n most frequent elements to their rank."""
    n -= reserved  # default: reserve 0 and 1 for <PAD> and <UNK>
    counts = Counter(elements)
    return {e: i for i, (e, _) in enumerate(counts.most_common(n), reserved)}


def vectorised(data, vocab=None, vocab_size=None):
    """Enumerate tokens, characters, terms, and concepts."""
    logging.info('Vectorising %d sentences', len(data))
    if vocab is None:
        tokens = (tok for sent in data for tok, *_ in sent)
        vocab = index(tokens, vocab_size or VOCAB_SIZE)

    words = [[vocab.get(tok, 1) for tok, *_ in sent] for sent in data]

    chars = (c for sent in data for tok, *_ in sent for c in tok)
    alphabet = index(chars, ALPHABET_SIZE)
    chars = [[[alphabet.get(c, 1) for c in tok] for tok, *_ in sent]
             for sent in data]

    term_ids = dict(zip(NER_TAGS, it.count()))
    terms = [[term_ids[tag] for _, tag, *_ in sent] for sent in data]
    concept_ids = defaultdict(it.count().__next__)
    _ = concept_ids[NIL]  # make sure NIL has index 0
    concepts = [[concept_ids[tag] for _, _, tag, *_ in sent] for sent in data]

    # Include optional dictionary features, while also updating concept_ids.
    if any(len(e) > 5 for sent in data for e in sent):
        features = [[concept_ids[e[5]] if len(e) > 5 else None for e in sent]
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

    return [words, chars, *features], [terms, concepts, concepts]


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
            logging.info('Improved F1 -- saving model to %s', self.dumpfn)
            self.wait = 0
            self.best = current
            self.model.save(self.dumpfn)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                if epoch > 0:
                    logging.info('Epoch %d: early stopping', epoch + 1)

    def evaluate(self):
        '''
        Compute F1 for the last CR layer.
        '''
        x, y = self.val_data
        pred = self.model.predict(x)
        for ys, preds, task in zip(y, pred, ('NER', 'NEN-1', 'NEN-2')):
            scores = PRF.from_one_hot(ys, preds)
            logging.info('%s: %s', task, scores)
        return scores.fscore  # F-score for the second NEN output


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

    def __init__(self, tp, relevant, selected):
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


def setup_logging(log_level='INFO', log_file=None):
    logging.basicConfig(level=log_level,
                        format='%(asctime)s: %(message)s')
    if log_file is not None:
        logger = logging.getLogger()  # root logger
        handler = logging.FileHandler(str(log_file))
        handler.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(handler)


if __name__ == '__main__':
    main()
