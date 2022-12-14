import string

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

stops = stopwords.words("english")


def _build_example_set(examples):
    """convert example lists from wordnet senses into sets"""
    example_set = set()
    for example in examples:
        example_toks = nltk.word_tokenize(example)
        example_set = example_set | set(example_toks)
    return example_set


def _get_gloss(poss_sense):
    """helper, serves to show the user a token was not in wordnet's corpus"""
    return "<UNK>" if isinstance(poss_sense, str) else poss_sense.definition()


def simplified_lesk(word: str, sent: str):
    """An implementation of the simplified lesk algorithm
    it returns the best sense of a given word based on
    the token overlap of metalinguistic features
    of the wordnet entry and the token overlap of the sentence"""
    if not sent.islower():
        sent = sent.lower()
    senses = wn.synsets(word)
    if not senses:
        return word
    best_sense = senses[0]
    max_overlap = 0
    context = set(nltk.word_tokenize(sent))
    for sense in senses:
        signature = _build_example_set(sense.examples()) | set(
            nltk.word_tokenize(sense.definition())
        )
        overlap = len(signature.intersection(context))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    return best_sense


def all_lesk_senses(corpus: str, glosses: bool = False):
    """calculates the most likely sense for every token in a corpus using a lesk algorithm
    WARNING: the return type is List[List[Tuple]] if glosses is True
    otherwise, it is List[List[wordsense]]"""
    sents = nltk.sent_tokenize(corpus)
    all_senses = []
    for sent in sents:
        sent_senses = []
        toks = [
            tok
            for tok in nltk.word_tokenize(sent.lower())
            if tok not in stops and tok not in string.punctuation
        ]
        for tok in toks:
            best_sense = simplified_lesk(tok, sent)
            if glosses:
                sent_senses.append((best_sense, _get_gloss(best_sense)))
            else:
                sent_senses.append(best_sense)
        all_senses.append(sent_senses)
    return all_senses


if __name__ == "__main__":
    # Getting the word senses for all words in a text
    sentence = "Shall I compare thee to a summer's day? " \
               "Thou art more lovely and more temperate."
    sense_list_sents = all_lesk_senses(sentence, glosses=True)

    # printing out every sense
    print("Full text senses")
    for sent in sense_list_sents:
        for sense in sent:
            print(sense)

    print("--------")  # for readability

    # Getting the word senses for one word in a sentence
    sent = "Time flies like an arrow"
    word = "flies"
    print("Single word sense")
    sense = simplified_lesk(word, sent)
    print(sense, sense.definition())
