import pickle
from sys import float_info
from math import log


class LanguageModel:
    sentence_tag = '<s>'
    end_sentence_tag = '</s>'
    unknown_tag = '<ukn>'
    base = 2

    def __init__(self, ngram=1, train_file = None, smoothing=0):
        self.raw_dict = {}
        self.ngram_counts = {}
        self.corpus_size = 0
        self.ngram = ngram
        self.smoothing = smoothing
        self.train(train_file)


    def tokenize(self, sentences):
        collection = []
        for sentence in sentences:
            processed_text = self.preprocess(sentence)
            tokens = processed_text.split(' ')
            collection.extend(tokens)
        return collection

    def word_types(self):
        return len(self.ngram_counts)

    def word_count(self):
        return self.sum_count(self.ngram_counts)
        pass

    def train(self, file_loc):
        with open(file_loc) as train_data:
            sentences = train_data.readlines()

        token_collection = self.tokenize(sentences)

        self.raw_dict = self.unique_count(token_collection)
        self.processed_dict = self.smooth(self.raw_dict)
        self.corpus_size = self.sum_count(self.processed_dict)

        sum = 0
        for key, value in self.processed_dict.items():
            sum += value

        features = self.featurize(token_collection, self.ngram)
        self.ngram_counts = self.unique_count(features)
        self.processed_ngram_counts = self.smooth(self.ngram_counts)
        self.corpus_size += self.smoothing * len(self.processed_ngram_counts)

        if self.ngram > 1:
            features = self.featurize(token_collection, self.ngram - 1)
            ngram_counts = self.unique_count(features)
            self.n_minus_one_gram = self.smooth(ngram_counts)
        else:
            self.n_minus_one_gram = self.processed_dict

    def smooth(self, count_list: dict):
        processed_count = {}
        for item, count in count_list.items():
            processed_count[item] = count + self.smoothing

        return processed_count

    def unique_count(self, collection):
        item_count = {}
        for item in collection:
            if item not in item_count:
                item_count[item] = 1
            else:
                item_count[item] += 1
        return item_count

    @staticmethod
    def log_to_high_prec(log_prob):
        from _decimal import Decimal
        return LanguageModel.base ** Decimal(log_prob)

    def evaluate(self, test_file):
        with open(test_file) as test_data:
            sentences = test_data.readlines()

        perplexity = self.perplexity(*sentences)

        seen_words = 0
        not_seen_words = 0
        unique_not_seen_words = 0
        total_word_count = 0
        processed_words = set()
        for sentence in sentences:
            processed = self.preprocess(sentence)
            words = processed.split(' ')
            for word in words:
                if word not in {self.sentence_tag, self.end_sentence_tag, self.unknown_tag}:
                    total_word_count += 1
                    if word in self.raw_dict:
                        seen_words += 1
                    else:
                        not_seen_words += 1
                        if word not in processed_words:
                            unique_not_seen_words += 1
                    processed_words.add(word)

        unique_word_count = len(processed_words)

        model_stats =  {'perplexity': perplexity,
                        'unseen percent': not_seen_words/total_word_count,
                        'unseen percent unique': unique_not_seen_words/ unique_word_count
                        }
        return model_stats

    def perplexity(self, *sentences):
        log_sum = 0
        word_count = 0
        for sentence in sentences:
            processed_sentence = self.preprocess(sentence)
            log_sum += self.log_prob(processed_sentence)
            word_count += len(sentence)

        entropy = (1 / word_count) * log_sum

        return self.base ** -entropy

    def log_prob(self, sentence):
        processed_sentence = self.preprocess(sentence)
        tokens = processed_sentence.split(' ')
        total_log_prob = 0
        for token_index in range(len(tokens)):
            if tokens[token_index] == self.sentence_tag:
                continue
            history = []
            for x in range(1, self.ngram):
                history.append(tokens[token_index - x])

            prob = self.posterior_prob(tokens[token_index], history)
            if prob == 0:
                log_prob = float('-inf')
            else:
                log_prob = log(prob, self.base)
            total_log_prob += log_prob

        return total_log_prob

    def sum_count(self, word_count: dict):
        sum = 0
        for word, count in word_count.items():
            sum += count
        return sum

    def posterior_prob(self, word, history=None):
        # print(word, '|', history)

        if word not in self.processed_dict:
            word = self.unknown_tag

        for x in range(len(history)):
            if history[x] not in self.n_minus_one_gram:
                history[x] = self.unknown_tag

        if self.ngram > 1:
            hist_str = ' '.join(history)
            ngram = hist_str + ' ' + word
            if ngram not in self.processed_ngram_counts:
                num = self.smoothing
            else:
                num = self.processed_ngram_counts[ngram]
            denom = self.n_minus_one_gram[hist_str]

        else:
            if word not in self.processed_ngram_counts:
                word = self.unknown_tag

            num = self.processed_ngram_counts[word]
            denom = self.corpus_size

        prob = num / denom
        return prob

    def featurize(self, tokens, gram):
        features = []
        for token_index in range(len(tokens) - 1):
            if self.raw_dict[tokens[token_index]] - self.smoothing <= 1:
                feature = self.unknown_tag
            else:
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
