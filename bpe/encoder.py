# coding=utf-8
""" An encoder which learns byte pair encodings for white-space separated text.  Can tokenize, encode, and decode. """
from collections import Counter
from typing import List, Iterable, Dict, Iterator, Callable

from nltk.tokenize import casual_tokenize
from tqdm import tqdm
import toolz
import json

DEFAULT_EOW = '</w>'


class Encoder:
    """ Encodes white-space separated text using byte-pair encoding.  See https://arxiv.org/abs/1508.07909 for details.
    """

    def __init__(self, vocab_size=8192, pct_bpe=0.5, word_tokenizer=casual_tokenize, silent=False, ngram_min=2,
                 ngram_max=4, batch_size=1000000, required_tokens=None, EOW: str=DEFAULT_EOW):
        if vocab_size < 1:
            raise ValueError('vocab size must be greater than 0.')

        self.word_vocab_size = max([int(vocab_size * (1 - pct_bpe)), len(required_tokens or [])])
        self.bpe_vocab_size = vocab_size - self.word_vocab_size
        self.word_tokenizer = word_tokenizer
        self.word_vocab = {}
        self.bpe_vocab = {}
        self.inverse_word_vocab = {}
        self.inverse_bpe_vocab = {}
        self._progress_bar = iter if silent else tqdm
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.batch_size = batch_size
        self.required_tokens = required_tokens
        self.EOW = EOW
        self.eow_len = len(EOW)
        self.UNK = '<unknown/>'

    def byte_pair_counts(self, words: Iterable[str]):
        """ Counts space separated token character pairs:
            [('T h i s </w>', 4}] -> {'Th': 4, 'hi': 4, 'is': 4, 's</w>': 4}
        """
        for token, count in self._progress_bar(self.count_tokens(words).items()):
            bp_counts = Counter()
            for ngram in token.split(' '):
                bp_counts[ngram] += count
            for ngram_size in range(self.ngram_min, min([self.ngram_max, len(token)]) + 1):
                ngrams = [''.join(ngram) for ngram in toolz.sliding_window(ngram_size, token.split(' '))]

                for ngram in ngrams:
                    bp_counts[''.join(ngram)] += count

            if self.EOW in bp_counts:
                del bp_counts[self.EOW]
            yield bp_counts

    def count_tokens(self, words: Iterable[str]):
        """ Count tokens into a BPE vocab """
        token_counts = Counter(self._progress_bar(words))
        return {' '.join(token) + ' ' + self.EOW: count for token, count in token_counts.items()}

    def learn_word_vocab(self, sentences: Iterable[str], tokenize: Callable[[str], List[str]],
                         max_size: int) -> Dict[str, int]:
        word_counts = Counter(word + self.EOW for word in toolz.concat(map(tokenize, sentences)))
        for token in (self.required_tokens or []):
            word_counts[token + self.EOW] = int(2**63)
        sorted_word_counts = sorted(word_counts.items(), key=lambda p: -p[1])
        return {word: idx for idx, (word, count) in enumerate(sorted_word_counts[:max_size])}

    def learn_bpe_vocab(self, words: Iterable[str]) -> Dict[str, int]:
        """ Learns a vocab of byte pair encodings """
        vocab = Counter()
        vocab[self.EOW] = int(2**63)
        vocab[self.UNK] = int(2**63)
        for idx, byte_pair_count in enumerate(self.byte_pair_counts(words)):
            for byte_pair, count in byte_pair_count.items():
                vocab[byte_pair] += count

            if (idx + 1) % 10000 == 0:
                self.trim_vocab(10 * self.bpe_vocab_size, vocab)

        sorted_bpe_counts = sorted(vocab.items(), key=lambda p: -p[1])[:self.bpe_vocab_size]
        return {bp: idx + self.word_vocab_size for idx, (bp, count) in enumerate(sorted_bpe_counts)}

    def fit(self, text: Iterable[str]):
        """ Learn vocab from text. """
        _text = [l.lower().strip() for l in text]

        # First, learn word vocab
        self.word_vocab = self.learn_word_vocab(_text, self.word_tokenizer, self.word_vocab_size)

        remaining_words = [word for word in toolz.concat(map(self.word_tokenizer, _text))
                           if (word + self.EOW) not in self.word_vocab]
        self.bpe_vocab = self.learn_bpe_vocab(remaining_words)

        self.inverse_word_vocab = {idx: token for token, idx in self.word_vocab.items()}
        self.inverse_bpe_vocab = {idx: token for token, idx in self.bpe_vocab.items()}

    @staticmethod
    def trim_vocab(n, vocab: Dict[str, int]):
        """  Deletes all pairs below 10 * vocab size to prevent memory problems """
        pair_counts = sorted(vocab.items(), key=lambda p: -p[1])
        pairs_to_trim = [pair for pair, count in pair_counts[n:]]
        for pair in pairs_to_trim:
            del vocab[pair]

    def subword_tokenize(self, word: str):
        word += self.EOW
        end_idx = len(word)
        sw_tokens = []
        start_idx = 0

        while start_idx < len(word):
            subword = word[start_idx:end_idx]
            if subword in self.bpe_vocab:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = len(word)
            elif len(subword) == 1:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = len(word)
            elif len(subword) == 1 + self.eow_len and subword.endswith(self.EOW):
                if subword in self.bpe_vocab:
                    sw_tokens.append(subword)
                    start_idx = end_idx
                    end_idx = len(word)
                else:
                    sw_tokens.append(subword[:1])
                    start_idx += 1
            elif subword == self.EOW:
                sw_tokens.append(subword)
                start_idx = end_idx
                end_idx = len(word)
            else:
                end_idx -= 1

        return sw_tokens

    def tokenize(self, sentence: str):
        """  """
        word_tokens = self.word_tokenizer(sentence)

        tokens = []
        for word_token in word_tokens:
            if (word_token + self.EOW in self.word_vocab) or \
                    (word_token + self.EOW in self.bpe_vocab):
                tokens.append(word_token + self.EOW)
            else:
                tokens.extend(self.subword_tokenize(word_token))

        return tokens

    def transform(self, sentences: Iterable[str], reversed: bool=False) -> Iterator[List[int]]:
        """ Turns space separated tokens into vocab idxs """
        direction = -1 if reversed else 1
        for sentence in sentences:
            encoded = []
            tokens = list(self.tokenize(sentence.lower().strip()))
            for token in tokens:
                if token in self.word_vocab:
                    encoded.append(self.word_vocab[token])
                elif token in self.bpe_vocab:
                    encoded.append(self.bpe_vocab[token])
                else:
                    encoded.append(self.UNK)

            yield encoded[::direction]

    def inverse_transform(self, rows: Iterable[List[int]]) -> Iterator[str]:
        """ Turns token indexes back into text """
        for row in rows:
            words = []

            rebuilding_word = False
            current_word = ''
            for idx in row:
                if rebuilding_word and (idx in self.inverse_bpe_vocab):

                    word = self.inverse_bpe_vocab[idx]
                    current_word += word

                    if word.endswith(self.EOW):
                        current_word = current_word[:-self.eow_len]
                        rebuilding_word = False
                        words.append(current_word)
                        current_word = ''

                elif idx in self.inverse_word_vocab:
                    words.append(self.inverse_word_vocab[idx][:-self.eow_len])

                elif idx in self.inverse_bpe_vocab:
                    word = self.inverse_bpe_vocab[idx]
                    if word.endswith(self.EOW):
                        words.append(word[:-self.eow_len])
                    else:
                        rebuilding_word = True
                        current_word += self.inverse_bpe_vocab[idx]

                else:
                    raise RuntimeError("Unable to unpack token IDX {}!".format(idx))

            yield ' '.join(words)

    def vocabs_to_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            'byte_pairs': self.bpe_vocab,
            'words': self.word_vocab,
        }

    def save(self, outpath: str):
        with open(outpath, 'w') as outfile:
            json.dump(self.vocabs_to_dict(), outfile)

    @classmethod
    def from_dict(cls, vocabs: Dict[str, Dict[str, int]]) -> 'Encoder':
        encoder = Encoder()
        encoder.word_vocab = vocabs['words']
        encoder.bpe_vocab = vocabs['byte_pairs']

        encoder.inverse_bpe_vocab = {v: k for k, v in encoder.bpe_vocab.items()}
        encoder.inverse_word_vocab = {v: k for k, v in encoder.word_vocab.items()}

        return encoder

    @classmethod
    def load(cls, in_path: str) -> 'Encoder':
        with open(in_path) as infile:
            obj = json.load(infile)
        return cls.from_dict(obj)
