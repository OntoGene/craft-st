#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:zhoukaiyin
"""

import collections
import os
import pickle
import time

import tensorflow as tf
from tensorflow.python.ops import math_ops
import tf_metrics

import modeling
import optimization
import tokenization


# ----------------------------- FLAGS ------------------------------------------

flags = tf.flags

FLAGS = flags.FLAGS


#! JOSEPH - Possible  BIOES, IOB, GLOBAL, CHEBI, PR, GO_BP
flags.DEFINE_string(
    "label_format", None, "The name of the format"
)

#! JOSEPH - Possible  CHEBI, PR, GO_BP, ...
flags.DEFINE_string(
    "onto", None, "The name of the Ontology"
)

#! JOSEPH
flags.DEFINE_string(
    "configuration", None, "The configuration to run"
)

# Symbol for "outside" (irrelevant), ie. 'O', 'NIL' etc.
flags.DEFINE_string(
    "outside_symbol", 'O', 'symbol for irrelevant tokens'
)


#-------------------------------------------------------------------------------

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", True,
                  "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

#! 1e-5 RECOMMENDED FOR NER JOSEPH (before 5e-5)
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# ------------------------- GET NUM LABELS -------------------------------------


def print_set_up(ontology, num_labels, path_to_data):
    print('\n{:{align}{width}}'.format('*' * 60, align='^', width='80'))
    print('{:{align}{width}}'.format(ontology, align='^', width='80'))
    print('{:{width}}{}: {}'.format(' ', 'num_labels', num_labels, width='20'))
    print('{:{width}}{}: {}'.format(' ', 'PATH TO DATA', path_to_data, width='20'))
    print('{:{align}{width}}'.format('*' * 60 +'\n', align='^', width='80'))
    time.sleep(5.5)


# ------------------------- FUNCTIONS ------------------------------------------


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        #self.label_mask = label_mask


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Read BIO data."""
        with open(input_file) as f:
            words = []
            labels = []
            for line in f:
                line = line.strip()
                if line:
                    word, *_, label = line.strip().split()
                    words.append(word)
                    labels.append(label)
                else:
                    assert len(words) == len(labels)
                    while len(words) > 30:
                        tmplabel = labels[:30]
                        for _ in range(len(tmplabel)):
                            if tmplabel.pop().startswith('O'):
                                break
                        cutoff = len(tmplabel) + 1
                        l = ' '.join(filter(None, labels[:cutoff]))
                        w = ' '.join(filter(None, words[:cutoff]))
                        yield (l, w)
                        words = words[cutoff:]
                        labels = labels[cutoff:]

                    if not words:
                        continue
                    l = ' '.join(filter(None, labels))
                    w = ' '.join(filter(None, words))
                    yield (l, w)
                    words = []
                    labels = []
        if words:
            l = ' '.join(filter(None, labels))
            w = ' '.join(filter(None, words))
            yield (l, w)


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train_dev.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "devel.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, config='iob'):
        """Get all output labels."""
        return getattr(self, f'_get_labels_{config}')()

#* -----------------------------------------------------------------------------

    #? IOB FORMAT
    @staticmethod
    def _get_labels_iob():
        print('IOB')
        time.sleep(5.5)

        return ["B", "I", "O", "X", "[CLS]", "[SEP]"]

#* -----------------------------------------------------------------------------

    #? BIOES FORMAT -->  num_labels = 9 = 1*4 + 4 + 1   Joseph
    @staticmethod
    def _get_labels_bioes():
        print('BIOES')

        print('\n{:{align}{width}}'.format('*' * 60, align='^', width='80'))
        print('{:{align}{width}}'.format('BIOES', align='^', width='80'))
        print('{:{width}}{}: {}'.format(' ', 'num_labels', 9, width='20'))
        print('{:{align}{width}}'.format('*' * 60 +'\n', align='^', width='80'))
        time.sleep(5.5)

        return ["B", "I", "O", "X", "[CLS]", "[SEP]", "E", "S"]

#* -----------------------------------------------------------------------------

    #? IDS FORMAT -->  CHEBI num_labels = ...-> = 481   Joseph
    def _get_labels_ids(self):
        path_to_data = os.path.join(FLAGS.data_dir, 'tag_set.txt')
        return self._get_id_tagset(path_to_data)

    # #? IDS AND PRETRAIN
    def _get_labels_pretrained_ids(self):
        return self._get_labels_pretraining()

    # #? PRETRAIN
    def _get_labels_pretrain(self):
        return self._get_labels_pretraining()

    def _get_labels_pretraining(self):
        path_to_data = os.path.join(FLAGS.data_dir, 'tag_set_pretrained.txt')
        return self._get_id_tagset(path_to_data)

    @staticmethod
    def _get_id_tagset(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            tag_set = [line.rstrip() for line  in f]

        tag_set.append("X")
        tag_set.append("[CLS]")
        tag_set.append("[SEP]")

        print_set_up(FLAGS.onto, len(tag_set)+1, filename)

        print('labels:', len(tag_set), tag_set, '\n\n\n\n\n')
        print(FLAGS.configuration.upper())
        print('SET SIZE: ', len(tag_set)+1)

        return tag_set


#* -----------------------------------------------------------------------------

    # #? GLOBAL BIOES FORMAT -->  num_labels = 10*4 + 4 + 1 = 45 Joseph
    @staticmethod
    def _get_labels_global():
        print('\n{:{align}{width}}'.format('*' * 60, align='^', width='80'))
        print('{:{align}{width}}'.format('GLOBAL', align='^', width='80'))
        print('{:{width}}{}: {}'.format(' ', 'num_labels', 45, width='20'))
        print('{:{align}{width}}'.format('*' * 60 +'\n', align='^', width='80'))
        time.sleep(5.5)

        return ["B-CHEBI", "I-CHEBI", "E-CHEBI", "S-CHEBI",
                "B-CL", "I-CL", "E-CL", "S-CL",
                "B-GO_BP", "I-GO_BP", "E-GO_BP", "S-GO_BP",
                "B-GO_CC", "I-GO_CC", "E-GO_CC", "S-GO_CC",
                "B-GO_MF", "I-GO_MF", "E-GO_MF", "S-GO_MF",
                "B-MOP", "I-MOP", "E-MOP", "S-MOP",
                "B-NCBITaxon", "I-NCBITaxon", "E-NCBITaxon", "S-NCBITaxon",
                "B-PR", "I-PR", "E-PR", "S-PR",
                "B-SO", "I-SO", "E-SO", "S-SO",
                "B-UBERON", "I-UBERON", "E-UBERON", "S-UBERON",
                "O", "X", "[CLS]", "[SEP]"]

#* -----------------------------------------------------------------------------

    @staticmethod
    def _create_example(lines, set_type):
        examples = []
        for i, (label, word) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(word)
            label = tokenization.convert_to_unicode(label)
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def write_tokens(tokens, mode):
    if mode == "test":
        path = os.path.join(FLAGS.output_dir, f"token_{mode}.txt")
        wf = open(path, 'a')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token+'\n')
        wf.close()


def convert_single_example(ex_index, example, label_list,
                           max_seq_length, tokenizer, mode):
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    with open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
        pickle.dump(label_map, w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    #label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        #label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    #assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        #tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        #label_mask = label_mask
    )
    write_tokens(ntokens, mode)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, mode=None
):
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        #features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        #"label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn


# ----------------------------- Create Model -----------------------------------

def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities, axis=-1)
        return (loss, per_example_loss, logits, log_probs, predict)
        ##########################################################################


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        #label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, per_example_loss, logits, log_probs, predicts = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits):
            # def metric_fn(label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids, predictions, num_labels,
                                                 [1, 2], average="macro")
                recall = tf_metrics.recall(label_ids, predictions, num_labels,
                                           [1, 2], average="macro")
                f = tf_metrics.f1(label_ids, predictions, num_labels,
                                  [1, 2], average="macro")
                #
                return {
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f": f,
                    #"eval_loss": loss,
                }
            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"prediction": predicts, "log_probs": log_probs},
                scaffold_fn=scaffold_fn
            )
        return output_spec
    return model_fn

# -------------------------------- Main ----------------------------------------

def main(_):
    for key in tf.app.flags.FLAGS.flag_values_dict():
        print(key, FLAGS[key].value)
    time.sleep(5)

    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ner": NerProcessor
    }
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels(FLAGS.configuration)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list)+1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
        with open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file,mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        tokens = list()
        with open(token_path, 'r') as reader:
            for line in reader:
                tok = line.strip()
                if tok == '[CLS]':
                    tmp_toks = [tok]
                elif tok == '[SEP]':
                    tmp_toks.append(tok)
                    tokens.append(tmp_toks)
                else:
                    tmp_toks.append(tok)

        prf = estimator.evaluate(input_fn=predict_input_fn, steps=None)
        tf.logging.info("***** token-level evaluation results *****")
        for key in sorted(prf.keys()):
            tf.logging.info("  %s = %s", key, str(prf[key]))

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        output_logits_file = os.path.join(FLAGS.output_dir, "logits_test.txt")
        with open(output_predict_file, 'w') as p_writer:
            with open(output_logits_file, 'w') as l_writer:
                for pidx, prediction in enumerate(result):
                    slen = len(tokens[pidx])

                    output_line = "\n".join(  # change 0 predictions to 'O'
                        id2label.get(id, FLAGS.outside_symbol)
                        for id in prediction['prediction'][:slen])
                    p_writer.write(output_line + "\n")

                    output_line = "\n".join(
                        '\t'.join(str(log_prob) for log_prob in log_probs)
                        for log_probs in prediction['log_probs'][:slen])
                    l_writer.write(output_line + "\n")


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")

    tf.app.run()
