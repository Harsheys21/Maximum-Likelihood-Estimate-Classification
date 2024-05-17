import numpy as np
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class BinaryClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the number of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass

# TODO: Implement this
class MaximumLikelihoodEstimateUnigram(BinaryClassifier): # Maximum Likelihood Estimate
    """Naive Bayes Classifier
    """
    def __init__(self):
        self.vocab_size = None
        self.frequency_list = None
        self.vocab_size = None
        self.prob_list = None
        

    def fit(self, transformed_list):
        # Add your code here!


        sum_list = np.sum(transformed_list, axis = 0)
        self.vocab_size = np.sum(sum_list)

        self.prob_list = sum_list / self.vocab_size

        
        #return prob_list
    
    def perplexity(self, X):
        # 
        # X should be tokenized data to get perplextiy score on

        data_probabilites = []
        
        print("takes time here a")
        #retrieve the probabilities of the data
        for item in X:
            data_probabilites.append(self.prob_list[item])

        inv_M = 1/3 # 1/3 because 3 tokens in corpus. Should be changed to variable later

        log_prob = np.log(data_probabilites)

        exponent_l = inv_M * np.sum(log_prob)

        return 2 ** -(exponent_l)
    
class MaximumLikelihoodEstimateBigram(BinaryClassifier): # Maximum Likelihood Estimate
    
    def __init__(self):
        self.vocab_size = None
        self.frequency_list = None
        self.vocab_size = None
        self.prob_list = None

    def fit(self, transformed_list_bi, transform_list_uni):
        sum_list = np.sum(transformed_list_bi, axis = 0)

        self.prob_list = sum_list[:, 0] / sum_list[:, 1]
        print(prob_list)
    
    def perplexity(self, X):
        # 
        # X should be tokenized data to get perplextiy score on

        data_probabilites = []
        
        print("takes time here a")
        #retrieve the probabilities of the data
        for item in X:
            data_probabilites.append(self.prob_list[item])

        inv_M = 1/3 # 1/3 because 3 tokens in corpus. Should be changed to variable later

        log_prob = np.log2(data_probabilites)

        exponent_l = inv_M * np.sum(log_prob)

        return 2 ** -(exponent_l)

class MaximumLikelihoodEstimateTrigram(BinaryClassifier): # Maximum Likelihood Estimate
    
    def __init__(self):
        self.frequency_list = None

    def fit(self, transformed_list):
        return 0

    def perplexity(self, X):
        # 
        # X should be tokenized data to get perplextiy score on

        data_probabilites = []
        
        print("takes time here a")
        #retrieve the probabilities of the data
        for item in X:
            data_probabilites.append(self.prob_list[item])

        inv_M = 1/3 # 1/3 because 3 tokens in corpus. Should be changed to variable later

        log_prob = np.log2(data_probabilites)

        exponent_l = inv_M * np.sum(log_prob)

        return 2 ** -(exponent_l)