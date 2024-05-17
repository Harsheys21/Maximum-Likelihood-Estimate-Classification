#from nltk.tokenize import regexp_tokenize
import numpy as np
import re

# Here is a default pattern for tokenization, you can substitute it with yours
default_pattern =  r'\s+'

def tokenize(text, pattern = default_pattern):
    """Tokenize senten with specific pattern
    
    Arguments:
        text {str} -- file to be tokenized"
    
    Keyword Arguments:
        pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
    
    Returns:
        list -- list of tokenized words, such as ['I', 'love', 'nlp']
    """
    tokens = text.split()
    return tokens

def get_indices(lst, item):
    arr = np.array(lst)
    return list(np.where(arr == item)[0])


class FeatureExtractor(object):
    """Base class for feature extraction.
    """
    def __init__(self):
        pass
    def fit(self, text_set):
        pass
    def transform(self, text):
        pass  
    def transform_list(self, text_set):
        pass



class UnigramFeature(FeatureExtractor):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        self.unigram = {'<UNK>', '<STOP>'}
        self.prob = {}
        
    def fit(self, text_set: list):
        """Fit a feature extractor based on given data 
        
        Arguments:
            text_set {list} -- list of tokenized sentences and words are lowercased, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        """
        # add start token
        token_count = {}

        for sentence in text_set:
            for token in sentence:
                token_count[token] = token_count.get(token, 0) + 1
                self.unigram.add(token)
            token_count['<STOP>'] = token_count.get('<STOP>', 0) + 1

        unk_value = 0
        for token in token_count.keys():
            if token != "<STOP>" and token != "<UNK>" and token_count[token] < 3:
                self.unigram.remove(token)
                unk_value += token_count[token]

        if unk_value > 0:
            token_count['<UNK>'] = unk_value

        # total words
        total = 0
        for word in self.unigram:
            total += token_count[word]

        for word in self.unigram:
            self.prob[word] = token_count[word] / total

    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0]) 
        """
        feature = {'<STOP>': 1}
        # set start token
        
        unk_value = 0

        for token in text:
            if token in self.unigram:
                if token in feature:
                    feature[token] += 1
                else:
                    feature[token] = 1
            else:
                unk_value += 1
        if unk_value > 0:
            feature['<UNK>'] = unk_value

        return feature
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = [self.transform(text) for text in text_set]
        return features
    

    def perplexity(self, features):
        logSum = 0
        totalWords = 0
        for line in features:
            for word in line:
                logSum += np.log2(line[word] * self.prob[word])
                totalWords += line[word]
        print(f"{logSum=}")
        print(f"{totalWords=}")
        sol = 2 ** (-logSum/totalWords)
        
        return sol
        


class BigramFeature(FeatureExtractor):

    def __init__(self):
        self.bigram = set()
        self.unigram = {'<UNK>', '<STOP>', '<START>'}
        self.prob = {}
        
    def fit(self, text_set: list):
        # add start token 
        unigram_count = {}

        for sentence in text_set:
            for token in sentence:
                unigram_count[token] = unigram_count.get(token,0) + 1
                self.unigram.add(token)
            unigram_count['<STOP>'] = unigram_count.get('<STOP>', 0) + 1

        unk_value = 0
        
        for token in unigram_count.keys():
            if token != "<STOP>" and unigram_count[token] < 3:
                self.unigram.remove(token)
                unk_value += unigram_count[token]

        if unk_value > 0:
            unigram_count['<UNK>'] = unk_value

        unigram_count['<START>'] = unigram_count['<STOP>']
        bigram_count = {}
        
        for sentence in text_set:
            sentence.insert(0, '<START>')
            sentence.append('<STOP>')
            for j in range(len(sentence) - 1):  # Adjusted range to consider bigrams
                if sentence[j] in self.unigram:
                    word_a =  sentence[j]
                else:
                    word_a = '<UNK>'
                
                if sentence[j+1] in self.unigram:
                    word_b =  sentence[j+1]
                else:
                    word_b = '<UNK>'

                bigram = (word_a, word_b)
                bigram_count[bigram] = bigram_count.get(bigram, 0) + 1
                self.bigram.add(bigram)
        
        for word in self.bigram:
            self.prob[word] = bigram_count[word] / unigram_count[word[0]]


    def transform(self, text: list):
        # create a 2d numpy array that contains numerator and denominator count of word
        feature = {}

        unk_value = 0
        text.insert(0, '<START>')
        text.append("<STOP>")
        for i in range(len(text) - 1):  # Adjusted range to consider bigrams
            bigram = (text[i], text[i + 1])
            if bigram in self.bigram:
                if bigram in feature:
                    feature[bigram] += 1
                else:
                    feature[bigram] = 1
            else:
                unk_value += 1
        
        if unk_value > 0:
            feature[('UNK','UNK')] = unk_value

        return feature
    
    def transform_list(self, text_set: list):
        features = [self.transform(text) for text in text_set]
        return features

    def perplexity(self, features):
        logSum = 0
        totalWords = 0
        for line in features:
            for word in line:
                logSum += np.log2(line[word] * self.prob[word])
                totalWords += line[word]
        print(f"{logSum=}")
        print(f"{totalWords=}")
        sol = 2 ** (-logSum/totalWords)
        
        return sol

class TrigramFeature(FeatureExtractor):

    def __init__(self):
        self.trigram = {}
        self.special_tokens = {'<START>', '<STOP>', '<UNK>'}
        
    def fit(self, text_set: list):
        # add start token
        token_count = {}

        index = 0

        for i in range(len(text_set)):
            for j in range(len(text_set[i]) - 2):  # Adjusted range to consider bigrams
                trigram = (text_set[i][j], text_set[i][j + 1], text_set[i][j + 2])
                if trigram not in self.trigram:
                    self.trigram[trigram] = index
                    token_count[trigram] = 1
                    index += 1
                else:
                    token_count[trigram] += 1

        # add unc token
        for token, count in token_count.items():
            if count < 3:
                if self.trigram.get("<UNK>") == None:
                    self.trigram['<UNK>'] = index
                    index += 1
                    self.trigram.pop(token)
                else:
                    self.trigram.pop(token)
        
        # add stop token
        self.trigram["<STOP>"] = index

        index = 0
        for key in self.trigram.keys():
            self.trigram[key] = index
            index += 1


    def transform(self, text: list):
        feature = np.zeros(len(self.trigram))
        for i in range(len(text) - 2):  # Adjusted range to consider bigrams
            trigram = (text[i], text[i + 1], text[i + 2])
            feature[self.trigram.get(trigram, self.trigram["<UNK>"])] += 1
        
        # set stop token
        feature[self.trigram['<STOP>']] = 1
        return feature
    
    def transform_list(self, text_set: list):
        features = [self.transform(text) for text in text_set]
        return features

