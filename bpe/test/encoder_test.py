# coding=utf-8
""" Tests that ensure the encoder works. """
from hypothesis import given
import hypothesis.strategies as st

from bpe.encoder import Encoder, DEFAULT_EOW, DEFAULT_SOW, DEFAULT_UNK, DEFAULT_PAD

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

EOW = DEFAULT_EOW
SOW = DEFAULT_SOW
UNK = DEFAULT_UNK
PAD = DEFAULT_PAD


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
    assert encoder.tokenize('from toolz import reduce') == [SOW, 'f', 'ro', 'm', EOW,
                                                            SOW, 'tool', 'z', EOW,
                                                            SOW, 'impo', 'rt', EOW,
                                                            SOW, 'redu', 'ce', EOW]


def test_single_letter_tokenizing():
    """ Should yield single letters when untrained """
    encoder = Encoder()
    assert encoder.tokenize('single letters') == \
        [SOW] + [UNK] * len('single') + [EOW, SOW] + [UNK] * len('letters') + [EOW]


def test_unseen_word_ending():
    """ The last character should come with a </w> even if it wasn't seen as the last letter of a 
        word in the training set.
    """
    encoder = Encoder(silent=True, pct_bpe=1)
    encoder.fit(test_corpus)
    assert encoder.tokenize('import toolz') == [SOW, 'impo', 'rt', EOW, SOW, 'tool', 'z', EOW]


def test_dump_and_load():
    """ Should be able to dump encoder to dict, then load it again. """
    encoder = Encoder(silent=True, pct_bpe=1)
    encoder.fit(test_corpus)
    assert encoder.tokenize('from toolz import reduce') == [SOW, 'f', 'ro', 'm', EOW,
                                                            SOW, 'tool', 'z', EOW,
                                                            SOW, 'impo', 'rt', EOW,
                                                            SOW, 'redu', 'ce', EOW]

    encoder_d = encoder.vocabs_to_dict()
    new_encoder = Encoder.from_dict(encoder_d)

    assert new_encoder.tokenize('from toolz import reduce') == [SOW, 'f', 'ro', 'm', EOW,
                                                                SOW, 'tool', 'z', EOW,
                                                                SOW, 'impo', 'rt', EOW,
                                                                SOW, 'redu', 'ce', EOW]


def test_required_tokens():
    """ Should be able to require tokens to be present in encoder """
    encoder = Encoder(silent=True, pct_bpe=1, required_tokens=['cats', 'dogs'])
    encoder.fit(test_corpus)
    assert 'cats' in encoder.word_vocab
    assert 'dogs' in encoder.word_vocab


def test_eow_accessible_on_encoders():
    encoder = Encoder()
    assert encoder.EOW == '</w>'


def test_subword_tokenize():
    encoder = Encoder(silent=True, pct_bpe=1)
    encoder.fit(test_corpus)
    assert list(encoder.subword_tokenize('this')) == [SOW, 'th', 'is', EOW]


def test_tokenize():
    encoder = Encoder(silent=True, pct_bpe=1)
    encoder.fit(test_corpus)
    assert list(encoder.tokenize('this is how')) == [SOW, 'th', 'is', EOW, SOW, 'is', EOW, SOW,
                                                     'ho', 'w', EOW]


def test_basic_transform():
    encoder = Encoder(silent=True, pct_bpe=1)
    encoder.fit(test_corpus)
    assert len(list(encoder.transform(['this']))[0]) == 4


def test_inverse_transform():
    encoder = Encoder(silent=True, pct_bpe=1)
    encoder.fit(test_corpus)

    transform = lambda text: list(encoder.inverse_transform(encoder.transform([text])))[0]

    assert transform('this is how we do it') == 'this is how we do it'

    assert transform('looking at the promotional stuff, it looks good.') == \
        'looking at the promotional stuff {} it looks good .'.format(UNK)

    assert transform('almost nothing should be recognized! let\'s see...') == \
        'almost nothing should be recognized {unk} let {unk} s see ...'.format(unk=UNK)


@given(st.lists(st.text()))
def test_encoder_learning_from_random_sentences(sentences):
    encoder = Encoder(silent=True)
    encoder.fit(test_corpus)
    encoded = encoder.transform(sentences)


def test_fixed_length_encoding():
    encoder = Encoder(silent=True, pct_bpe=1, required_tokens=[PAD])
    encoder.fit(test_corpus)

    result = list(encoder.transform([''], fixed_length=10))
    assert len(result) == 1
    assert len(result[0]) == 10

    result = list(encoder.transform(['', 'import ' * 50], fixed_length=10))
    assert len(result) == 2
    assert len(result[0]) == 10
    assert len(result[1]) == 10


def test_unknown_char_handling():
    encoder = Encoder(silent=True, pct_bpe=1)
    encoder.fit(test_corpus)

    result = list(encoder.inverse_transform(encoder.transform([';'])))[0]
    assert encoder.UNK in result
    assert ';' not in result


def test_mixed_encoder():
    encoder = Encoder(silent=True, vocab_size=1000, pct_bpe=0.98)
    encoder.fit(test_corpus)
    assert encoder.tokenize('import this yield toolz') == ['import', SOW, 'th', 'is', EOW, SOW,
                                                           'yiel', 'd', EOW, SOW, 'tool', 'z', EOW]
