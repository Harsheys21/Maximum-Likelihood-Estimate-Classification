import pandas as pd
from utils import *
from utils_additive import *
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
    parser.add_argument('--smoothing', '-s', type=str, default='None', choices=['linear', 'add1', 'None'])
    args = parser.parse_args()
    print(args)


    # obtain the train and test frame
    train_frame, test_frame = read_data(args.path)

    # tokenize the train text
    tokenized_text = [tokenize(line) for line in train_frame.readlines()]

    interpolation = False
    if args.smoothing == "linear":
        interpolation = True
    
    additive = False
    if args.smoothing == "add1":
        additive = True

    unigramIndices = None
    bigramIndices = None
    trigramIndices = None

    # determine which feature extractor to use
    if (interpolation):
        unigramModel = UnigramFeature()
        bigramModel = BigramFeature()
        trigramModel = TrigramFeature()
        unigramModel.fit(tokenized_text)
        bigramModel.fit(tokenized_text)
        trigramModel.fit(tokenized_text)
        unigramIndices = unigramModel.transform_list(tokenized_text)
        bigramIndices = bigramModel.transform_list(tokenized_text)
        trigramIndices = trigramModel.transform_list(tokenized_text)
    else:
        if args.feature == "unigram":
            if additive:
                feat_extractor = UnigramFeaturea1()
            else:
                feat_extractor = UnigramFeature()
        elif args.feature == "bigram":
            if additive:
                feat_extractor = BigramFeaturea1()
            else:
                feat_extractor = BigramFeature()
        elif args.feature == "trigram":
            if additive:
                feat_extractor = TrigramFeaturea1()
            else:
                feat_extractor = TrigramFeature()
        else:
            raise Exception("Pass unigram, bigram, or trigram to --feature")
    
    

        # fit the train tokenized text onto the feature extractor
        feat_extractor.fit(tokenized_text)

        # obtain train vectors for MLE training
        X_train = feat_extractor.transform_list(tokenized_text)

        # ---------------------------------------------------------------------------

        test = False #change value to enable/disable hdtv . test file

        if (test):
            with open('test.txt', 'r') as f:
                test_text = [tokenize(line) for line in f.readlines()]
                indices = feat_extractor.transform_list(test_text)
                print("train perplexity:", feat_extractor.perplexity(indices))
                return
        # output accuracy
        print("===== %s Perplexity Scores =====" % args.feature)
        with open('data/1b_benchmark.train.tokens', 'r') as f:
            test_text = [tokenize(line) for line in f.readlines()]
            indices = feat_extractor.transform_list(test_text)
            print("train perplexity:", feat_extractor.perplexity(indices))

        with open('data/1b_benchmark.dev.tokens', 'r') as f:
            test_text = [tokenize(line) for line in f.readlines()]
            indices = feat_extractor.transform_list(test_text)
            print("dev perplexity:", feat_extractor.perplexity(indices))

        with open('data/1b_benchmark.test.tokens', 'r') as f: # ONLY UNCOMMENT ON FINAL RUN
            test_text = [tokenize(line) for line in f.readlines()]
            indices = feat_extractor.transform_list(test_text)
            print("test perplexity:", feat_extractor.perplexity(indices))

    if (interpolation):
        y1 = .1
        y2 = .3
        y3 = .6

        test = False #change value to enable/disable hdtv . test file

        print("===== Linear Interpolation Perplexity Score %.1f %.1f %.1f =====" % (y1, y2, y3))
        if (test):
            with open('test.txt', 'r') as f:
                test_text = [tokenize(line) for line in f.readlines()]
                indices = trigramModel.transform_list(test_text)

                #perplexity for lienar interpolation
                sum = 0
                totalWords = 0
                for sentence in indices:
                    for word in sentence:
                        if word == ('<START>', '<START>', '<START>'):
                            continue
                        if word[1:] == ('<STOP>', '<STOP>'):
                            pass
                        sum += np.log2(sentence[word] * ((y1 * unigramModel.prob[word[2]]) + (y2 * bigramModel.prob[word[1:]]) + (y3 * trigramModel.prob[word] )))
                        totalWords += sentence[word]
                val = 2 ** (-sum/totalWords)

                print("test perplexity:", val)
        
        with open('data/1b_benchmark.train.tokens', 'r') as f:
            test_text = [tokenize(line) for line in f.readlines()]
            indices = trigramModel.transform_list(test_text)

            #perplexity for lienar interpolation
            sum = 0
            totalWords = 0
            for sentence in indices:
                for word in sentence:
                    if word == ('<START>', '<START>', '<START>'):
                        continue
                    if word[1:] == ('<STOP>', '<STOP>'):
                        pass
                    sum += np.log2(sentence[word] * ((y1 * unigramModel.prob[word[2]]) + (y2 * bigramModel.prob[word[1:]]) + (y3 * trigramModel.prob[word] )))
                    totalWords += sentence[word]
            val = 2 ** (-sum/totalWords)

            print("train perplexity:", val)

        with open('data/1b_benchmark.dev.tokens', 'r') as f:
            test_text = [tokenize(line) for line in f.readlines()]
            indices = trigramModel.transform_list(test_text)

            #perplexity for lienar interpolation
            sum = 0
            totalWords = 0
            for sentence in indices:
                for word in sentence:
                    if word == ('<START>', '<START>', '<START>'):
                        continue
                    if word[1:] == ('<STOP>', '<STOP>'):
                        pass
                    sum += np.log2(sentence[word] * ((y1 * unigramModel.prob[word[2]]) + (y2 * bigramModel.prob[word[1:]]) + (y3 * trigramModel.prob[word] )))
                    totalWords += sentence[word]
            val = 2 ** (-sum/totalWords)

            print("dev perplexity:", val)
        
        with open('data/1b_benchmark.test.tokens', 'r') as f: # ONLY UNCOMMENT ON FINAL RUN
            test_text = [tokenize(line) for line in f.readlines()]
            indices = trigramModel.transform_list(test_text)

            #perplexity for lienar interpolation
            sum = 0
            totalWords = 0
            for sentence in indices:
                for word in sentence:
                    if word == ('<START>', '<START>', '<START>'):
                        continue
                    if word[1:] == ('<STOP>', '<STOP>'):
                        pass
                    sum += np.log2(sentence[word] * ((y1 * unigramModel.prob[word[2]]) + (y2 * bigramModel.prob[word[1:]]) + (y3 * trigramModel.prob[word] )))
                    totalWords += sentence[word]
            val = 2 ** (-sum/totalWords)

            print("train perplexity:", val)
        




if __name__ == '__main__':
    main()
