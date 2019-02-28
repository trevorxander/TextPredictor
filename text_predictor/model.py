import pickle


class LanguageModel:
    _SMOOTHING = 1
    sentence_tag = '<s>'
    end_sentence_tag = '</s>'
    unknown_tag = '<ukn>'

    def __init__(self, ngram=1):
        self.raw_dict = {}
        self.ngram_count = {}
        self.ngram = ngram

    def train(self, file_loc):
        with open(file_loc) as train_data:
            sentences = train_data.readlines()

        token_collection = []
        for sentence in sentences:
            processed_text = self.preprocess(sentence)
            token_collection.extend(processed_text.split(' '))

        self.raw_dict = self.unique_count(token_collection)

        features = self.featurize(token_collection, self.ngram)
        self.ngram_count = self.unique_count(features)

        self.processed_dict = self.add_tag_threshold(self.raw_dict, self.unknown_tag, 1)
        self.processed_ngram_count = self.add_tag_threshold(self.ngram_count, self.unknown_tag, 1)

        if self.ngram > 1:
            features = self.featurize(token_collection, self.ngram - 1)
            self.n_minus_one_gram =  self.unique_count(features)
        else:
            self.n_minus_one_gram = self.processed_dict

        

    def add_tag_threshold(self, count_list: dict, tag, threshold: int):
        processed_count = {tag: 0}
        for item, count in count_list.items():
            if count <= threshold:
                processed_count[tag] += 1
            else:
                processed_count[item] = count
        return processed_count

    def unique_count(self, list):
        item_count = {}
        for item in list:
            if item not in item_count:
                item_count[item] = 1
            else:
                item_count[item] += 1
        return item_count

    def evaluate(self):
        pass

    def featurize(self, tokens, gram):
        features = []
        for token_index in range(len(tokens) - 1):
            feature = tokens[token_index]
            for look_ahead in range(1, gram):
                if tokens[token_index + look_ahead] == self.end_sentence_tag:
                    feature += ' ' + self.end_sentence_tag
                    break
                feature += ' ' + tokens[token_index + look_ahead]
            features.append(feature)
        if gram == 1:
            features.append(self.end_sentence_tag)
        return features

    def preprocess(self, text):
        from text_predictor.preprocess import pad, prepocess_funcs
        processed = text
        for preprocess in prepocess_funcs:
            processed = preprocess(processed)
        return pad(processed, self.ngram, self.sentence_tag, self.end_sentence_tag)

    def store_model(self, file_loc):
        model_file = open(file_loc, 'wb')

        model_file.close()

    def load_model(self, file_loc):
        model_file = open(file_loc, 'rb')
        self._category_prob = pickle.load(model_file)
        self._feature_set = pickle.load(model_file)
        self._category_features_freq = pickle.load(model_file)
        self._error = pickle.load(model_file)
        model_file.close()
