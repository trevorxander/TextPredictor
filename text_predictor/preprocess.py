def pad(sentence):
    return '<s> ' + sentence + ' </s>'


def to_lowercase(sentence: str):
    return sentence.lower()


def remove_newline(sentence):
    return sentence.rstrip('\n\r')
