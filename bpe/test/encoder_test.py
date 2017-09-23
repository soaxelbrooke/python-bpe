# coding=utf-8
""" Tests that ensure the encoder works. """
from hypothesis import given
import hypothesis.strategies as st

from bpe.encoder import Encoder, EOW

# Generated with http://pythonpsum.com
test_corpus = '''Object raspberrypi functools dict kwargs. Gevent raspberrypi functools. Dunder raspberrypi decorator dict lambda zip import pyramid.
Kwargs raspberrypi diversity unit object gevent. Import integration decorator unit django yield functools twisted. Dunder integration decorator goat future. Python raspberrypi community pypy. Kwargs integration beautiful test reduce gil python closure. Gevent integration generator test kwargs raise itertools.
Reduce integration coroutine bdfl python. Cython integration beautiful list python.
Object raspberrypi diversity 2to3 dunder script. Python integration exception dict kwargs dunder pycon. Import raspberrypi beautiful test import six web. Future integration mercurial self script web. Return raspberrypi community test.
Django raspberrypi mercurial unit import yield raspberrypi rocksdahouse. Dunder raspberrypi mercurial list reduce class test scipy.
Object integration beautiful 2to3. Kwargs raspberrypi beautiful dict lambda class generator. Django integration coroutine dict import web map pyramid. Kwargs raspberrypi diversity dict future scipy raspberrypi rocksdahouse.
Import raspberrypi exception list object. Object raspberrypi coroutine unit lambda sys django. Method raspberrypi generator test reduce future tuple. Kwargs raspberrypi decorator list cython def import twisted.
Method raspberrypi beautiful unit method cython implicit zip. Dunder integration generator dict gevent def.
Gevent integration decorator test python object guido. Reduce integration beautiful goat.
Method raspberrypi diversity pypi return tuple list. Django integration functools. Method integration beautiful self return future kwargs. Gevent raspberrypi functools unit lambda zip python science functools. Future raspberrypi community pypy return six cython.
'''.split('\n')


@given(st.integers(min_value=1))
def test_encoder_creation(vocab_size):
    """ Should be able to instantiate an Encoder with expected params """
    Encoder(vocab_size=vocab_size)


@given(st.integers(max_value=0))
def test_encoder_creation_graceful_failure(vocab_size):
    """ Min vocab size is 1.  Anything lower should ValueError """
    died = False

    try:
        Encoder(vocab_size=vocab_size)
    except ValueError:
        died = True

    assert died, "Encoder should have raised a ValueError for < 1 vocab size"


def test_bpe_encoder_fit():
    """ Encoer should be able to fit to provided text data. """
    encoder = Encoder(silent=True, pct_bpe=1)
    encoder.fit(test_corpus)
    assert encoder.tokenize('from toolz import reduce') == ['f', 'ro', 'm' + EOW,
                                                            'tool', 'z' + EOW,
                                                            'impo', 'rt' + EOW,
                                                            'redu', 'ce' + EOW]


def test_single_letter_encoding():
    """ Should yield single letters when untrained """
    encoder = Encoder()
    assert encoder.tokenize('single letters') == list('singl') + ['e' + EOW] + list('letter') + ['s' + EOW]


def test_unseen_word_ending():
    """ The last character should come with a </w> even if it wasn't seen as the last letter of a word in the training
        set.
    """
    encoder = Encoder(silent=True, pct_bpe=1)
    encoder.fit(test_corpus)
    assert encoder.tokenize('import toolz') == ['impo', 'rt' + EOW, 'tool', 'z' + EOW]


def test_dump_and_load():
    """ Should be able to dump encoder to dict, then load it again. """
    encoder = Encoder(silent=True, pct_bpe=1)
    encoder.fit(test_corpus)
    assert encoder.tokenize('from toolz import reduce') == ['f', 'ro', 'm' + EOW,
                                                            'tool', 'z' + EOW,
                                                            'impo', 'rt' + EOW,
                                                            'redu', 'ce' + EOW]

    encoder_d = encoder.vocabs_to_dict()
    new_encoder = Encoder.from_dict(encoder_d)

    assert encoder.tokenize('from toolz import reduce') == ['f', 'ro', 'm' + EOW,
                                                            'tool', 'z' + EOW,
                                                            'impo', 'rt' + EOW,
                                                            'redu', 'ce' + EOW]

def test_required_tokens():
    """ Should be able to require tokens to be present in encoder """
    encoder = Encoder(silent=True, pct_bpe=1, required_tokens=['cats', 'dogs'])
    encoder.fit(test_corpus)
    assert 'cats' in encoder.word_vocab
    assert 'dogs' in encoder.word_vocab
