import pickle


class LanguageModel:
    _SMOOTHENING = 1

    def __init__(self, ngram=1):
        self.dict = {}
        self.word_prob = {}
        self.ngram = ngram

    def train(self, file_loc):
        with open(file_loc) as train_data:
            sentences = train_data.readlines()
        tokens = []
        features = []
        for sentence in sentences:
            processed_text = self.preprocess(sentence)
            tokens = processed_text.split(' ')
            features += self.featurize(tokens)

        for feature in features:
            if feature not in self.word_prob:
                self.word_prob[feature] = 1
            else:
                self.word_prob[feature] += 1
        print(len(self.word_prob))

    def evaluate(self):
        pass

    sentence_tag = '<s>'
    end_sentence_tag = '</s>'

    def featurize(self, tokens):
        features = []
        feature_grams = []
        for token_index in range(len(tokens) - 1):
            feature = tokens[token_index]
            for look_ahead in range(1, self.ngram):
                if tokens[token_index + look_ahead] == self.end_sentence_tag:
                    feature += ' ' + self.end_sentence_tag
                    break
                feature += ' ' + tokens[token_index + look_ahead]
            features.append(feature)
        if self.ngram == 1:
            features.append(self.end_sentence_tag)
        return features

    def preprocess(self, text):
        from text_predictor.preprocess import pad, to_lowercase, remove_newline
        processed = to_lowercase(pad(remove_newline(text)))
        return processed

    def _store_model(self, file_loc):
        pass

    def _load_model(self, file_loc):
        pass
