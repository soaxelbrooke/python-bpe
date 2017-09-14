# coding=utf-8
""" Tests that ensure the encoder works. """
from bpe.encoder import Encoder


def test_encoder_creation():
    """ Should be able to instantiate an Encoder with expected params """
    encoder = Encoder(vocab_size=20)
