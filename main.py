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
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'trigram'])
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
        model = MaximumLikelihoodEstimateUnigram()
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
        feat_extractor_b = UnigramFeature()
        model = MaximumLikelihoodEstimateBigram()
    elif args.feature == "trigram":
        feat_extractor = TrigramFeature()
        feat_extractor_b = BigramFeature()
        model = MaximumLikelihoodEstimateTrigram()
    else:
        raise Exception("Pass unigram, bigram, or trigram to --feature")

    # fit the train tokenized text onto the feature extractor
    feat_extractor.fit(tokenized_text)

    if args.feature != "unigram":
        feat_extractor_b.fit(tokenized_text)

    # obtain train vectors for MLE training
    X_train = feat_extractor.transform_list(tokenized_text)

    if args.feature != "unigram":
        X_train_b = feat_extractor.transform_list(tokenized_text)

    
    
    # read the test frame
    tokenized_text = [tokenize(line) for line in test_frame.readlines()]
    X_test = feat_extractor.transform_list(tokenized_text)

    # ---------------------------------------------------------------------------
    # start the time
    start_time = time.time()
    
    # fit the model
    if args.feature != 'unigram':
        model.fit(X_train, X_train_b)
    else:
        probabilities = model.fit(X_train)

    # output accuracy
    print("===== Train Accuracy =====")
    #accuracy(model.predict(X_train))
    
    print("===== Test Accuracy =====")
    #accuracy(model.predict(X_test))

    print("Time for training and test: %.2f seconds" % (time.time() - start_time))



if __name__ == '__main__':
    main()
