# coding=utf-8
""" An encoder which learns byte pair encodings for white-space separated text.  Can tokenize, encode, and decode. """


class Encoder:
    """ Encodes white-space separated text using byte-pair encoding.  See https://arxiv.org/abs/1508.07909 for details.
    """

    def __init__(self, vocab_size=32768):
        if vocab_size < 1:
            raise ValueError('vocab_size must be greater than 0.')

        self.vocab_size = vocab_size
