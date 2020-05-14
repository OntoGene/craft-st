# CRAFT-NN: Parallel Concept Recognition for CRAFT v4

Parallel named entity recognition (NER) and normalisation (NEN) based on sequence labeling with either BiLSTM or BioBERT.

## Introduction

This repository hosts the code of our participation in the [CRAFT shared task 2019](https://sites.google.com/view/craft-shared-task-2019/).
If you are interested in training a similar system for biomedical concept recognition, keep reading.
If you rather want to use the trained system to predict entities in other texts, have a look at our [Zenodo deposit](https://doi.org/10.5281/zenodo.3822363).

## Quick Guide

- Get [CRAFT v4](https://github.com/UCDenver-ccp/CRAFT).
- Convert concept annotations to CoNLL format (see below).
- Create dictionary-based predictions using [OGER](https://github.com/OntoGene/OGER) (optional part of both the BiLSTM and BioBERT system).
- Train models with the code in _bilstm/_ or _biobert/_.
- Convert the predictions from .conll back to .bionlp.
- Evaluate with the [official evaluation suite](https://github.com/UCDenver-ccp/craft-shared-tasks).


## Format Conversion

- Follow [the instructions](https://github.com/UCDenver-ccp/CRAFT/wiki/Alternative-annotation-file-formats) to create standoff-annotations in BioNLP format. Place them in a _bionlp_ subdirectory for each entity type.
- Run `git submodule init` to get a clone of Pyysalo's standoff2conll converter.
- Make sure the CRAFT corpus is available as a directory or link named _CRAFT_ in this directory.
- Run `./bionlp2conll.sh <NAME> <PATH>`, where NAME is "CHEBI", "CL" etc. and PATH is the target directory. This creates a 4-column CoNLL file for each article.
- If you use dictionary-based predictions, add them as a 5th column.
- For the BioBERT system, use _biobert/biocodes/conll2conll.py_ to convert the documents to 2-column CoNLL, and the same script again to convert the predictions back to 4-column format (including prediction harmonisation).
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
In: *Proceedings of The 5th Workshop on BioNLP Open Shared Tasks (BioNLP-OST 2019)*.
| [PDF](https://www.aclweb.org/anthology/D19-5726.pdf)
| [bibtex](https://github.com/OntoGene/craft-st/wiki/uploads/furrer-et-al-2019.bib) |
