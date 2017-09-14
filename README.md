
# BPE

AKA Byte Pair Encoding.  Learns a vocab and byte pair encoding for provided white-space separated text.

## Usage

```bash
$ pip install bpe
```

```python
""" Artificially small vocab size chosen for demonstration.  my_corpus.txt file is line separated text. """
from bpe import Encoder

encoder = Encoder(vocab_size=1024)
line_iter = (line.strip() for line in open('my_corpus.txt') if len(line) > 0)
encoder.learn_vobcab(line_iter)

example = 'Rare words handled inconcievably well.'
print(encoder.tokenize(example))
print(encoder.encode(example))
print(encoder.decode(encoder.encode(example)))
```
