
# BPE [![Build Status](https://travis-ci.org/soaxelbrooke/python-bpe.svg?branch=master)](https://travis-ci.org/soaxelbrooke/python-bpe)

AKA Byte Pair Encoding.  Learns a vocab and byte pair encoding for provided white-space separated text.

## Usage

```bash
$ pip3 install --user bpe
```

```python
from bpe import Encoder

# Generated with http://pythonpsum.com
test_corpus = '''
    Object raspberrypi functools dict kwargs. Gevent raspberrypi functools. Dunder raspberrypi decorator dict didn't lambda zip import pyramid, she lambda iterate?
    Kwargs raspberrypi diversity unit object gevent. Import fall integration decorator unit django yield functools twisted. Dunder integration decorator he she future. Python raspberrypi community pypy. Kwargs integration beautiful test reduce gil python closure. Gevent he integration generator fall test kwargs raise didn't visor he itertools...
    Reduce integration coroutine bdfl he python. Cython didn't integration while beautiful list python didn't nit!
    Object fall diversity 2to3 dunder script. Python fall for: integration exception dict kwargs dunder pycon. Import raspberrypi beautiful test import six web. Future integration mercurial self script web. Return raspberrypi community test she stable.
    Django raspberrypi mercurial unit import yield raspberrypi visual rocksdahouse. Dunder raspberrypi mercurial list reduce class test scipy helmet zip?
'''

encoder = Encoder(200, pct_bpe=0.88)  # params chosen for demonstration purposes
encoder.fit(test_corpus.split('\n'))

example = "Vizzini: He didn't fall? INCONCEIVABLE!"
print(encoder.tokenize(example))
# ['__sow', 'vi', 'z', 'zi', 'ni', '__eow', '__sow', ':', '__eow', 'he', 'didn', "'", 't', 'fall', '__sow', '?', '__eow', '__sow', 'in', 'co', 'n', 'ce', 'iv', 'ab', 'le', '__eow', '__sow', '!', '__eow']
# ['__sow', 'vi', 'z', 'zi', 'ni', '__eow', '__sow', ':', '__eow', 'he', 'did', 'not', 'fall', '__sow', '?', '__eow', '__sow', 'in', 'co', 'n', 'ce', 'iv', 'ab', 'le', '__eow', '__sow', '!', '__eow']
print(next(encoder.transform([example])))
# [25, 102, 76, 77, 68, 24, 25, 149, 24, 13, 10, 11, 12, 25, 79, 24, 25, 135, 58, 37, 152, 81, 160, 108, 24, 25, 143, 24]
print(next(encoder.inverse_transform(encoder.transform([example]))))
# vizzini : he did not fall ? inconceivable !
```
