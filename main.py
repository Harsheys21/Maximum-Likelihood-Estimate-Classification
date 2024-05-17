import pandas as pd
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

    train_frame = open(path + '1b_benchmark.train.tokens', 'r', encoding='utf-8')

    try:
        #test_frame = pd.read_csv(path + '1b_benchmark.dev.tokens')
        test_frame = open(path + '1b_benchmark.dev.tokens', 'r', encoding='utf-8')
    except FileNotFoundError:
        test_frame = train_frame

    return train_frame, test_frame


def main():
    # parse the argument
    start_time = time.time()
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
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
    elif args.feature == "trigram":
        feat_extractor = TrigramFeature()
    else:
        raise Exception("Pass unigram, bigram, or trigram to --feature")

    # fit the train tokenized text onto the feature extractor
    feat_extractor.fit(tokenized_text)

    # obtain train vectors for MLE training
    X_train = feat_extractor.transform_list(tokenized_text)

    # read the test frame
    # tokenized_text = [tokenize(line) for line in test_frame.readlines()]
    # X_test = feat_extractor.transform_list(tokenized_text)

    # ---------------------------------------------------------------------------

    # output accuracy
    print("===== Unigram Perplexity Score =====")
    with open('test.txt', 'r') as f:
        test_text = [tokenize(line) for line in f.readlines()]
        indices = feat_extractor.transform_list(test_text)
        print("perplexity:", feat_extractor.perplexity(indices))


if __name__ == '__main__':
    main()
