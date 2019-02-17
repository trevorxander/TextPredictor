from text_predictor import LanguageModel

def predict_text ():
    model = LanguageModel(ngram=1)
    model.train('dataset/brown-train.txt')

if __name__ == '__main__':
    predict_text()


