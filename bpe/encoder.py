# coding=utf-8
""" An encoder which learns byte pair encodings for white-space separated text.  Can tokenize, encode, and decode. """
from collections import Counter

from nltk.tokenize import casual_tokenize
from tqdm import tqdm
import toolz

EOW = '</w>'


class Encoder:
    """ Encodes white-space separated text using byte-pair encoding.  See https://arxiv.org/abs/1508.07909 for details.
    """

    def __init__(self, vocab_size=32768, bpe_vocab_size=None, word_tokenizer=casual_tokenize, silent=False, ngram_min=2,
                 ngram_max=4, batch_size=1000000):
        if vocab_size < 1 and (bpe_vocab_size is not None and bpe_vocab_size < 0):
            raise ValueError('vocab size must be greater than 0.')

        self.vocab_size = vocab_size
        self.bpe_vocab_size = bpe_vocab_size or vocab_size
        self.word_tokenizer = word_tokenizer
        self.vocab = {}
        self.inverted_vocab = {}
        self._progress_bar = iter if silent else tqdm
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.batch_size = batch_size

    def byte_pair_counts(self, text):
        """ Counts space separated token character pairs:
            [('T h i s </w>', 4}] -> {'Th': 4, 'hi': 4, 'is': 4, 's</w>': 4}
        """
        for token, count in self._progress_bar(self.count_tokens(text).items()):
            bp_counts = Counter()
            for ngram in token.split(' '):
                bp_counts[ngram] += count
            for ngram_size in range(self.ngram_min, min([self.ngram_max, len(token)])):
                ngrams = [''.join(ngram) for ngram in toolz.sliding_window(ngram_size, token.split(' '))]

                for ngram in ngrams:
                    bp_counts[''.join(ngram)] += count

            if EOW in bp_counts:
                del bp_counts[EOW]
            yield bp_counts

    def count_tokens(self, text):
        """ Count tokens into a BPE vocab """
        token_counts = Counter(toolz.concat(map(self.word_tokenizer, self._progress_bar(text))))
        return {' '.join(token) + ' ' + EOW: count for token, count in token_counts.items()}

    def fit(self, text):
        """ Learn vocab from text. """
        vocab = Counter()
        for batch in toolz.partition_all(self.batch_size, self.byte_pair_counts(text)):
            for counter in batch:
                vocab += counter

            self.trim_vocab(10 * self.bpe_vocab_size, vocab)

        self.trim_vocab(self.bpe_vocab_size, vocab)
        self.vocab = {pair: idx for idx, pair in enumerate(sorted(vocab.keys()))}
        self.inverted_vocab = {idx: pair for pair, idx in self.vocab.items()}

    @staticmethod
    def trim_vocab(n, vocab):
        """  Deletes all pairs below 10 * vocab size to prevent memory problems """
        pair_counts = sorted(vocab.items(), key=lambda p: -p[1])
        pairs_to_trim = [pair for pair, count in pair_counts[n:]]
        for pair in pairs_to_trim:
            del vocab[pair]

    def subword_tokenize(self, word):
        word += EOW
        end_idx = len(word)
        sw_tokens = []
        start_idx = 0

        while start_idx < len(word):
            subword = word[start_idx:end_idx]
            if subword in self.vocab:
                yield subword
                start_idx = end_idx
                end_idx = len(word)
            elif len(subword) == 1:
                yield subword
                start_idx = end_idx
                end_idx = len(word)
            elif len(subword) == 1 + len(EOW) and subword.endswith(EOW):
                yield subword
                start_idx = end_idx
                end_idx = len(word)
            else:
                end_idx -= 1

        return sw_tokens

    def tokenize(self, sentence):
        """  """
        word_tokens = self.word_tokenizer(sentence)

        tokens = []
        for word_token in word_tokens:
            if word_token + EOW in self.vocab:
                tokens.append(word_token + EOW)
            else:
                tokens.extend(self.subword_tokenize(word_token))

        return tokens

    def encode(self, sentences):
        """ Turns space separated tokens into vocab idxs """
        for sentence in sentences:
            yield [self.vocab[token] for token in self.tokenize(sentence)]

    @classmethod
    def tokens_to_words(cls, tokens):
        words = []
        current_word = []

        for token in tokens:
            if token.endswith(EOW):
                current_word.append(token[-len(EOW):])
                words.append(''.join(current_word))
            else:
                current_word.append(token)

        return words

    def decode(self, idx_rows):
        """ Turns vocab indexes into tokens """
        for idxs in idx_rows:
            yield self.tokens_to_words([self.vocab[idx] for idx in idxs])
