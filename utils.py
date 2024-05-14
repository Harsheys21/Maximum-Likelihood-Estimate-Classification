from nltk.tokenize import regexp_tokenize
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
        self.unigram = {}
        
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

        index = 0
        unk_boolean = False
        # adding unc token
        for token, count in token_count.items():
            if count >= 3:
                self.unigram[token] = index
                index += 1
            else:
                unk_boolean = True

        if unk_boolean == True:
            self.unigram['<UNK>'] = index
            index += 1
        
        # add stop token
        self.unigram["<STOP>"] = index


    def transform(self, text: list):
        """Transform a given sentence into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text {list} -- a tokenized sentence (list of words), such as ["I", "love", "nlp"]
        
        Returns:
            array -- an unigram feature array, such as array([1,1,1,0,0]) 
        """
        feature = np.zeros(len(self.unigram))
        # set start token

        for token in text:
            feature[self.unigram.get(token, self.unigram["<UNK>"])] += 1

        # set stop token
        feature[self.unigram['<STOP>']] = 1
        return feature
    
    def transform_list(self, text_set: list):
        """Transform a list of tokenized sentences into vectors based on the extractor you got from self.fit()
        
        Arguments:
            text_set {list} --a list of tokenized sentences, such as [["I", "love", "nlp"], ["I", "like", "python"]]
        
        Returns:
            array -- unigram feature arraies, such as array([[1,1,1,0,0], [1,0,0,1,1]])
        """
        features = [self.transform(text) for text in text_set]
        return np.array(features)
            


class BigramFeature(FeatureExtractor):

    def __init__(self):
        self.bigram = {}
        
    def fit(self, text_set: list):
        # add start token 
        token_count = {}

        index = 0

        for i in range(len(text_set)):
            for j in range(len(text_set[i]) - 1):  # Adjusted range to consider bigrams
                bigram = (text_set[i][j], text_set[i][j + 1])
                if bigram not in self.bigram:
                    self.bigram[bigram] = index
                    token_count[bigram] = 1
                    index += 1
                else:
                    token_count[bigram] += 1

        # add unc token
        for token, count in token_count.items():
            if count < 3:
                if self.bigram.get("<UNK>") == None:
                    self.bigram['<UNK>'] = index
                    index += 1
                    self.bigram.pop(token)
                else:
                    self.bigram.pop(token)
        
        # add stop token
        self.bigram["<STOP>"] = index

        index = 0
        for key in self.bigram.keys():
            self.bigram[key] = index
            index += 1


    def transform(self, text: list):
        feature = np.zeros(len(self.bigram))
        for i in range(len(text) - 1):  # Adjusted range to consider bigrams
            bigram = (text[i], text[i + 1])
            feature[self.bigram.get(bigram, self.bigram["<UNK>"])] += 1
        
        # set stop token
        feature[self.bigram['<STOP>']] = 1
        return feature
    
    def transform_list(self, text_set: list):
        features = [self.transform(text) for text in text_set]
        return np.array(features)

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
        return np.array(features)

