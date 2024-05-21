#from nltk.tokenize import regexp_tokenize
import numpy as np
import re

# Here is a default pattern for tokenization, you can substitute it with yours
default_pattern =  r'\s+'

# additive smoothing value
smoothing = 0.1

def tokenizea1(text, pattern = default_pattern):
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

def get_indicesa1(lst, item):
    arr = np.array(lst)
    return list(np.where(arr == item)[0])


class FeatureExtractora1(object):
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



class UnigramFeaturea1(FeatureExtractora1):
    """Example code for unigram feature extraction
    """
    def __init__(self):
        self.unigram = {'<UNK>', '<STOP>'}
        self.prob = {}
        self.vocab_size = None
        
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

        self.vocab_size = len(self.unigram)

        for word in self.unigram:
            self.prob[word] = (token_count[word] + smoothing) / (total + (smoothing * self.vocab_size))


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
        sum = 0
        totalWords = 0
        for sentence in features:
            for word in sentence:
                sum += np.log2(sentence[word] * self.prob[word])
                totalWords += sentence[word]
        val = 2 ** (-sum/totalWords)
        
        return val
        


class BigramFeaturea1(FeatureExtractora1):

    def __init__(self):
        self.bigram = set()
        self.unigram = {'<UNK>', '<STOP>', '<START>'}
        self.prob = {}
        self.vocab_size = 0
        
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
        
        self.vocab_size = len(self.bigram)

        for word in self.bigram:
            self.prob[word] = (bigram_count[word] + smoothing) / (unigram_count[word[0]] + (smoothing * self.vocab_size))


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
            feature[('<UNK>','<UNK>')] = unk_value

        return feature
    
    def transform_list(self, text_set: list):
        features = [self.transform(text) for text in text_set]
        return features

    def perplexity(self, features):
        sum = 0
        totalWords = 0
        for sentence in features:
            for word in sentence:
                sum += np.log2(sentence[word] * self.prob[word])
                totalWords += sentence[word]
        val = 2 ** (-sum/totalWords)
        
        return val

class TrigramFeaturea1(FeatureExtractora1):

    def __init__(self):
        self.trigram = set()
        self.bigram = set()
        self.unigram = {'<UNK>', '<STOP>', '<START>'}
        self.prob = {}
        self.vocab_size = 0
        
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

        trigram_count = {}
        
        for sentence in text_set:
            sentence.insert(0,'<START>')
            for j in range(len(sentence) - 2):  # Adjusted range to consider bigrams
                if sentence[j] in self.unigram:
                    word_a =  sentence[j]
                else:
                    word_a = '<UNK>'
                
                if sentence[j+1] in self.unigram:
                    word_b =  sentence[j+1]
                else:
                    word_b = '<UNK>'

                if sentence[j+2] in self.unigram:
                    word_c =  sentence[j+2]
                else:
                    word_c = '<UNK>'

                trigram = (word_a, word_b, word_c)
                trigram_count[trigram] = trigram_count.get(trigram, 0) + 1
                self.trigram.add(trigram)
        
        self.vocab_size = len(self.trigram)

        for word in self.trigram:
            if word[0] == '<START>' and word[1] == '<START>':
                self.prob[word] = (bigram_count[(word[1],word[2])] + smoothing)/(unigram_count[word[1]] + (smoothing * self.vocab_size))
            else:
                self.prob[word] = (trigram_count[word] + smoothing) / (bigram_count[(word[0], word[1])] + (smoothing * self.vocab_size))


    def transform(self, text: list):
        # create a 2d numpy array that contains numerator and denominator count of word
        feature = {}

        unk_value = 0
        text.insert(0, '<START>')
        text.insert(0, '<START>')
        text.append("<STOP>")
        for i in range(len(text) - 2):  # Adjusted range to consider bigrams
            trigram = (text[i], text[i + 1], text[i+2])
            if trigram in self.trigram:
                if trigram in feature:
                    feature[trigram] += 1
                else:
                    feature[trigram] = 1
            else:
                unk_value += 1
        
        if unk_value > 0:
            feature[('<UNK>','<UNK>','<UNK>')] = unk_value

        return feature
    
    def transform_list(self, text_set: list):
        features = [self.transform(text) for text in text_set]
        return features

    def perplexity(self, features):
        sum = 0
        totalWords = 0
        for sentence in features:
            for word in sentence:
                sum += np.log2(sentence[word] * self.prob[word])
                totalWords += sentence[word]
        val = 2 ** (-sum/totalWords)
        
        return val


