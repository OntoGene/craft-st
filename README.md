# CRAFT-NN: Concept Recognition for CRAFT v3

CNN-BiLSTM for joint medical-entity recognition and normalisation.


## Quick Guide

- Get [CRAFT version 3](https://github.com/lfurrer/CRAFT).
- Convert concept annotations to CoNLL format (see below).
- Run the stand-alone script `train.py` (see `./run.py -h` for options) to train in a cross-validation setting.
- Use `predicty.py` and `ensemble.py` to create predictions from the trained models.
- Convert the predictions from .conll back to .bionlp.
- Evaluate with the [official evaluation suite](https://github.com/UCDenver-ccp/craft-shared-tasks).


## Format Conversion

- Follow [the instructions](https://github.com/UCDenver-ccp/CRAFT/wiki/Alternative-annotation-file-formats) to create standoff-annotations in BioNLP format. Place them in a _bionlp_ subdirectory for each entity type.
- Run `git submodule init` to get a clone of Pyysalo's standoff2conll converter.
- Make sure the CRAFT corpus is available as a directory or link named _CRAFT_ in this directory.
- Run `./bionlp2conll.sh <NAME> <PATH>`, where NAME is "CHEBI", "CL" etc. and PATH is the target directory.
- For converting predicted CoNLL files back to standoff, run `standoff2conll/conll2standoff.py < path/to/doc.conll > path/to/doc.bionlp` for each document.


## Labels for Ontology Pretraining

The labels (IDs) selected for ontology pretraining (y<sup>C</sup><sub>P</sub> in the paper) are listed in [this archive](top-1000-ids.tar.gz).


## Python Dependencies

- `keras` (BiLSTM)
- `tensorflow` (BioBERT)


## License

The code in this repository is licensed under the [AGPL-3.0](LICENSE).  
However, the code in the [biobert](/biobert) subdirectory uses an [Apache License](/biobert/LICENSE).


## Citation

If you use this code, please cite us:

Lenz Furrer, Joseph Cornelius, and Fabio Rinaldi (2019):
**UZH@CRAFT-ST: a Sequence-labeling Approach to Concept Recognition**.
In: *Proceedings of the BioNLP Open Shared Tasks Workshop (BioNLP-OST 2019)*.
| [PDF](https://github.com/OntoGene/craft-st/wiki/uploads/furrer-et-al-2019.pdf)
| [bibtex](https://github.com/OntoGene/craft-st/wiki/uploads/furrer-et-al-2019.bib) |
