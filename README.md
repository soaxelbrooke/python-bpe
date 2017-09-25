
# BPE [![Build Status](https://travis-ci.org/soaxelbrooke/python-bpe.svg?branch=master)](https://travis-ci.org/soaxelbrooke/python-bpe)

AKA Byte Pair Encoding.  Learns a vocab and byte pair encoding for provided white-space separated text.

## Usage

```bash
$ pip install -e git+https://github.com/soaxelbrooke/python-bpe.git#egg=bpe
```

```python
""" Artificially small vocab size chosen for demonstration.  my_corpus.txt file is line separated text. """
from bpe import Encoder

encoder = Encoder(vocab_size=1024)
line_iter = (line for line in open('my_corpus.txt') if len(line) > 0)
encoder.fit(line_iter)

example = 'Rare words handled inconcievably well.'
print(encoder.tokenize(example))
print(encoder.transform([example]))
print(encoder.inverse_transform(encoder.transform([example])))
```
