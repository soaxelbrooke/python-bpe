import json
import sys

from bpe.encoder import Encoder


def main(corpus_path):
    # type: (str) -> None
    """ Loads corpus, learns word and BPE vocab, and writes to stdout.  Assumes corpus is
        line-separated text.
    """
    with open(corpus_path) as infile:
        lines = list(map(str.strip, infile))

    encoder = Encoder(silent=True)
    encoder.fit(lines)
    print(json.dumps(encoder.vocabs_to_dict()))


if __name__ == '__main__':
    main(sys.argv[1])
