# CRAFT-BiLSTM

CNN-BiLSTM for joint medical-entity recognition and normalisation with CRAFT v4.


## Usage

- Follow the [instructions for format conversion](../README.md) to create input documents in CoNLL format (4 columns).
- Optionally add a 5th column with OGER predictions.
- Run the stand-alone script `train.py` (see `./train.py -h` for options) to train in a cross-validation setting.
- Use `predicty.py` and `ensemble.py` to create predictions from the trained models.
