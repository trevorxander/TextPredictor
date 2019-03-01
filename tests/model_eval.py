from text_predictor import LanguageModel

test_sentences = ['He was laughed off the screen .',
                  'There was no compulsion behind them .',
                  'I look forward to hearing your reply .']

test_files = ['dataset/brown-test.txt',
              'dataset/learner-test.txt']

train = 'dataset/brown-train.txt'

test_models = [LanguageModel(ngram=1, train_file='dataset/brown-train.txt'),
               LanguageModel(ngram=2, train_file='dataset/brown-train.txt'),
               LanguageModel(ngram=2, smoothing=1, train_file='dataset/brown-train.txt')]


def run_test():
    model = test_models[0]
    print('Words in training set: ' + str(model.word_count()))
    print('Unique words in training set: ' + str(model.word_types()) + '\n')

    print('MODEL EVALUATION\n')
    for test_file in test_files:
        for model in test_models:
            eval_stats = model.evaluate(test_file)
            print('Model: n-gram = {ngram}, smoothing = {smooth} \n'
                  'Test Data: {file}\n'
                  'Unseen Words Percentage: {unseen}%\n'
                  'Unseen Unique Words Percentage: {unique}%\n'
                  'Perplexity: {perplexity} \n'
                  .format(ngram=model.ngram, smooth=model.smoothing,
                          file=test_file,
                          unseen=eval_stats['unseen percent'] * 100,
                          unique=eval_stats['unseen percent unique'] * 100,
                          perplexity=eval_stats['perplexity']))

    print('\nSENTENCE TEST\n')
    for sentence in test_sentences:
        for model in test_models:
            log_prob = model.log_prob(sentence)
            print('Model: n-gram = {ngram}, smoothing = {smooth} \n'
                  'Sentence: {sentence} \n'
                  'Log(base-2) Probability: {log_prob}\n'
                  'Probability: {prob} \n'
                  'Perplexity: {perplexity} \n'
                  .format(ngram=model.ngram, smooth=model.smoothing,
                          sentence=sentence,
                          prob=model.log_to_high_prec(log_prob),
                          log_prob=log_prob,
                          perplexity=model.perplexity(sentence)))


if __name__ == '__main__':
    run_test()
