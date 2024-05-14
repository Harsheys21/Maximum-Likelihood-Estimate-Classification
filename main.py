import pandas as pd
from classifiers import *
from utils import *
import numpy as np
import time
import argparse


def accuracy(pred, labels):
    correct = (np.array(pred) == np.array(labels)).sum()
    accuracy = correct / len(pred)
    print("Accuracy: %i / %i = %.4f " % (correct, len(pred), correct / len(pred)))


def read_data(path):
    #train_frame = pd.read_csv(path + '1b_benchmark.train.tokens')

    train_frame = open(path + '1b_benchmark.train.tokens', encoding='utf8')

    try:
        #test_frame = pd.read_csv(path + '1b_benchmark.dev.tokens')
        test_frame = open(path + '1b_benchmark.dev.tokens')
    except FileNotFoundError:
        test_frame = train_frame

    return train_frame, test_frame


def main():
    # parse the argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='AlwaysPredictZero',
                        choices=['AlwaysPredictZero', 'NaiveBayes', 'LogisticRegression', 'BonusClassifier'])
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'customized'])
    parser.add_argument('--path', type=str, default='./A2-Data/', help='path to datasets')
    args = parser.parse_args()
    print(args)

    # obtain the train and test frame
    train_frame, test_frame = read_data(args.path)


    # tokenize the train text
    tokenized_text = [tokenize(line) for line in train_frame.readlines()]


    # determine which feature extractor to use
    if args.feature == "unigram":
        feat_extractor = UnigramFeature()
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
    elif args.feature == "trigram":
        feat_extractor = TrigramFeature()
    else:
        raise Exception("Pass unigram, bigram, or trigram to --feature")

    # fit the tokenized onto the feature extractor
    feat_extractor.fit(tokenized_text)
    print("Vocab size of train:",len(feat_extractor.unigram))

    X_train = feat_extractor.transform_list(tokenized_text)
    print(feat)

    exit()
    # read the test frame
    #tokenized_text = [tokenize(line) for line in test_frame.readlines()]
    X_test = feat_extractor.transform_list(tokenized_text)

    # ---------------------------------------------------------------------------
    if args.model == "AlwaysPredictZero":
        model = AlwaysPredictZero()
    elif args.model == "NaiveBayes":
        model = NaiveBayesClassifier()
    elif args.model == "LogisticRegression":
        model = LogisticRegressionClassifier()
    elif args.model == 'BonusClassifier':
        model = BonusClassifier()
    else:
        raise Exception("Pass AlwaysPredictZero, NaiveBayes, LogisticRegression, or BonusClassifier to --model")

    start_time = time.time()
    model.fit(X_train, Y_trqain)
    print("===== Train Accuracy =====")
    accuracy(model.predict(X_train), Y_train)
    
    print("===== Test Accuracy =====")
    accuracy(model.predict(X_test), Y_test)

    print("Time for training and test: %.2f seconds" % (time.time() - start_time))

    # Calculate and print the most distinctly positive and negative words
    if isinstance(model, NaiveBayesClassifier):
        likelihood_1 = model.likelihood[1]
        likelihood_0 = model.likelihood[0]
        ratios = likelihood_1 / (likelihood_0 + 1e-10)  # Add a small value to avoid division by zero
        sorted_indices = np.argsort(ratios)

        # Sort the unigram dictionary by values (indices)
        sorted_unigram = {k: v for k, v in sorted(feat_extractor.unigram.items(), key=lambda item: item[1])}

        # Extract the top 10 positive and negative words based on the sorted indices
        top_10_positive_words = [(word, ratios[index]) for word, index in sorted_unigram.items() if index in sorted_indices[-10:]]
        top_10_negative_words = [(word, ratios[index]) for word, index in sorted_unigram.items() if index in sorted_indices[:10]]

        print("\nTop 10 positive words with ratios:")
        for word, ratio in top_10_positive_words:
            print(f"{word}: {ratio}")

        print("\nTop 10 negative words with ratios:")
        for word, ratio in top_10_negative_words:
            print(f"{word}: {ratio}")



if __name__ == '__main__':
    main()
