# -- This implementation aims at demonstrating chunking
#   Name: Anurag Banerjee
# Course: CS7017 - Selected topics in NLP
#   Roll: 17071003
# -- Trainer and evaluator


import random
from collections import Iterable
from nltk import ChunkParserI, ClassifierBasedTagger
from nltk.stem.snowball import SnowballStemmer
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.corpus import conll2000
from nltk import word_tokenize, pos_tag


classifier_chunker = 0     # using this name as a global variable


def extractFeatures(tokens, index, history):
    """
    :param tokens: a POS-tagged sentence
    :param index: index of the token we want to extract features for
    :param history: previous predicted IOB tags
    :return:
    """

    # using the standard Porter english stemming algorithm
    stemmer = SnowballStemmer('english')
    # Pad the sequence with placeholders
    tokens = [('__START2__', '__START2__'), ('__START1__', '__START1__')] + list(tokens) + [('__END1__', '__END1__'), ('__END2__', '__END2__')]
    history = ['__START2__', '__START1__'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2
    word, pos = tokens[index]
    prevWord, prevPos = tokens[index - 1]
    prevprevWord, prevprevPos = tokens[index - 2]
    nextWord, nextPos = tokens[index + 1]
    nextnextWord, nextnextPos = tokens[index + 2]

    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'next-word': nextWord,
        'next-pos': nextPos,
        'next-next-word': nextnextWord,
        'next-next-Pos': nextnextPos,
        'prev-word': prevWord,
        'prev-pos': prevPos,
        'prev-prev-word': prevprevWord,
        'prev-prev-pos': prevprevPos,
    }


class ClassifierChunkParser(ChunkParserI):
    def __init__(self, chunked_sents, **kwargs):
        assert isinstance(chunked_sents, Iterable)

        # Transform the trees in IOB annotated sentences [(word, pos, chunk), ...]
        chunked_sents = [tree2conlltags(sent) for sent in chunked_sents]

        # Transform the triplets in pairs, make it compatible with the
        # tagger interface [((word, pos), chunk), ...]
        def triplets2tagged_pairs(iob_sent):
            return [((word, pos), chunk) for word, pos, chunk in iob_sent]

        chunked_sents = [triplets2tagged_pairs(sent) for sent in chunked_sents]

        # self.feature_detector = extractFeatures
        self.tagger = ClassifierBasedTagger(
            train=chunked_sents,
            feature_detector=extractFeatures,
            **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)


def trainer():
    global classifier_chunker

    print("Training on CoNLL 2000 corpus...")
    # get the sentences from the corpus and shuffle them
    shuffled_conll_sents = list(conll2000.chunked_sents())
    random.shuffle(shuffled_conll_sents)

    # use 90% of corpus for training
    train_sents = shuffled_conll_sents[:int(len(shuffled_conll_sents) * 0.9)]
    # now train the classifier
    classifier_chunker = ClassifierChunkParser(train_sents)
    # use remaining 10% of corpus for testing
    test_sents = shuffled_conll_sents[int(len(shuffled_conll_sents) * 0.9 + 1):]
    # now test the classifier performance
    print classifier_chunker.evaluate(test_sents)
    print("Training complete!")


def evaluator():
    print("Performing chunking on sample sentence 1...")
    # Piece of text from TOI 15 Dec 2017, Times Global, "Biased US cannot be a mediator between Israel and Palestine":
    print classifier_chunker.parse(pos_tag(word_tokenize(
            "From now on, it is out of the question for a biased US to be a mediator between Israel and Palestine, that period is over.")))
    print("Performing chunking on sample sentence 2...")
    # random sentence
    print classifier_chunker.parse(pos_tag(word_tokenize(
            "This affectionate behaviour towards the masses was one of the reasons why he could build an instant rapport with them.")))


if __name__ == '__main__':
    # Running a trainer on CoNLL 2000 corpus
    trainer()
    # trainer completed
    # Trying to chunk a sample sentence
    evaluator()
    print("Completed action successfully.")

