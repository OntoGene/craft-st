# CRAFT-NN: Concept Recognition for CRAFT v3

CNN-BiLSTM for joint medical-entity recognition and normalisation.


## Howto

- Get [CRAFT version 3](https://github.com/UCDenver-ccp/CRAFT/).
- Convert concept annotations to CoNLL format (see below).
- Run the stand-alone script `train.py` (see `./run.py -h` for options).
- Evaluate with the [official evaluation suite](https://github.com/UCDenver-ccp/craft-shared-tasks).


## Format Conversion

- Follow [the instructions](https://github.com/UCDenver-ccp/CRAFT/wiki/Alternative-annotation-file-formats) to create standoff-annotations in BioNLP format. Place them in a _bionlp_ subdirectory for each entity type.
- Run `git submodule init` to get a clone of Pyysalo's standoff2conll converter.
- Make sure the CRAFT corpus is available as a directory or link named _CRAFT_ in this directory.
- Run `./bionlp2conll.sh <NAME> <PATH>`, where NAME is "CHEBI", "CL" etc. and PATH is the target directory.
- For converting predicted CoNLL files back to standoff, run `standoff2conll/conll2standoff.py < path/to/doc.conll > path/to/doc.bionlp` for each document.


## Python Dependencies

- `keras`
