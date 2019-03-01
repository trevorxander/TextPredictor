
def pad(sentence, ngram, start_tag, end_tag):
    processed = '{pad_start} {sentence} {pad_end}'
    pad_start = start_tag
    for pad_count in range(ngram - 1):
        pad_start += ' ' + start_tag

    return processed.format(pad_start=pad_start, pad_end=end_tag, sentence=sentence)

def to_lowercase(sentence: str):
    return sentence.lower()

def remove_newline(sentence):
    return sentence.rstrip('\n\r')

prepocess_funcs = [remove_newline, to_lowercase]
