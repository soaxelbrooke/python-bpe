# coding=utf-8
""" Tests that ensure the encoder works. """
from hypothesis import given
import hypothesis.strategies as st

from bpe.encoder import Encoder


def test_encoder_creation():
    """ Should be able to instantiate an Encoder with expected params """
    Encoder(vocab_size=20)


@given(st.integers(max_value=0))
def test_encoder_creation_graceful_failure(vocab_size):
    died = False

    try:
        Encoder(vocab_size=vocab_size)
    except ValueError:
        died = True

    assert died, "Encoder should have raised a ValueError for < 1 vocab size"
