from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time                                                
import pickle
import copy
import string
import operator
from collections import OrderedDict
from autocorrect import spell
from pprint import pprint
import re

import pandas as pd
import numpy as np

from nltk.tokenize import TreebankWordTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Decorators inside classes are not a good idea
# https://stackoverflow.com/questions/13852138/how-can-i-define-decorator-method-inside-class
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed



def pickle_load(filename):
    with open(filename, 'rb') as f:
        # https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
        return pickle.load(f)


class Haptik:
    
    def xy_separator(self, df):
        """Separates feature matrix (column 0) and target vector (the rest)
        """
        X = df.iloc[:, 0]
        y = df.iloc[:, 1:]
        return X, y
    
    def multi_label_binarizer(self, df):
        """Maps ["T", "F"] to [1, 0] in a given dataframe
        """
        df = df.astype(str).applymap(lambda x: 1 if x=='T' else 0)
        return df

    def __init__(self, path, random_state=42):
        self.random_state = random_state
        self.path = path
        self.train = pd.read_csv(path + '/train_data.csv', encoding='utf-8')
        self.test = pd.read_csv(path + '/test_data.csv', encoding='utf-8')
        self.X_train, self.y_train = self.xy_separator(self.train)
        self.X_test, self.y_test = self.xy_separator(self.test)
        self.target_names = self.y_train.columns
        
        self.y_train = self.multi_label_binarizer(self.y_train)
        self.y_test = self.multi_label_binarizer(self.y_test)
        
        self.X_train = pd.Series(self.X_train)
        self.X_test = pd.Series(self.X_test)
        
        # Compute distribution of label frequencies
        freq = np.ravel(self.y_train.sum(axis=0))
        freq = dict(zip(self.y_train.columns, freq))
        freq = OrderedDict(sorted(freq.items(), key=operator.itemgetter(1)))
        self.yhist = pd.DataFrame({'label':freq.keys(), 'count':freq.values()})
        self.yhist['normalized'] = self.yhist['count']/self.yhist['count'].sum()
        
        self.gridsearchcv = None
        self.model = None
        self.train_dtm = None
        self.test_dtm = None
        self.feature_names = None
        self.feature_selector = None
        
        return None


class Haptik(Haptik):
    
    def summary(self):
        """A chainable wrapper for dunder repr
        """
        print(self.__repr__())
        return self
    
    def __repr__(self):
        """Defines String representation of the class
        """
        output = '\nX_train shape: ' + str(self.X_train.shape) + '\ny_train shape: ' + str(self.y_train.shape) + \
                 '\nX_test shape: ' + str(self.X_test.shape) + '\ny_test shape: ' + str(self.y_test.shape) + '\n'
        return output


class Haptik(Haptik):
    
    def _preprocess(self, listlikeobj, stop_lists=None):
        """Applies pre-processing pipelines to lists of string
        """
        
        numeric = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', \
                    'ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', \
                    'Eighteen', 'Nineteen', 'Twenty', 'Twenty-one', 'Twenty-two', 'Twenty-three', \
                    'Twenty-four', 'Twenty-five', 'Twenty-six', 'Twenty-seven', 'Twenty-eight', \
                    'Twenty-nine', 'Thirty', 'Thirty-one']
        
        ordinal = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eight', 'ninth', \
                    'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', \
                    'seventeenth', 'eighteenth', 'nineteenth', 'twentieth', 'twenty-first', 'twenty-second', \
                    'twenty-third', 'twenty-fourth', 'twenty-fifth', \
                    'twenty-sixth', 'twenty-seventh', 'twenty eighth', 'twenty-ninth', 'thirtieth', 'thirty-first']
        
        
        en_stop = get_stop_words('en')
        tokenizer = TreebankWordTokenizer()
        p_stemmer = PorterStemmer()
        
        listlikeobj = listlikeobj.apply(lambda row: row.lower())
        listlikeobj = listlikeobj.apply(lambda row: tokenizer.tokenize(row))
        listlikeobj = listlikeobj.apply(lambda row: [i for i in row if i not in en_stop])
        listlikeobj = listlikeobj.apply(lambda row: [i for i in row if i not in string.punctuation])
        listlikeobj = listlikeobj.apply(lambda row: [p_stemmer.stem(i) for i in row])
        if stop_lists:
            for sw_dict in stop_lists:
                listlikeobj = listlikeobj.apply(lambda row: [i for i in row if i not in sw_dict])
        #listlikeobj = listlikeobj.apply(lambda row: [re.sub(r'\d', "#", i) for i in row])
        #listlikeobj = listlikeobj.apply(lambda row: ["#" for i in row if i in numeric])
        #listlikeobj = listlikeobj.apply(lambda row: ["#th" for i in row if i in ordinal])
        #print(listlikeobj)
        
        #listlikeobj = listlikeobj.apply(lambda row: [spell(i) for i in row if len(i)>6])
        
        
        return listlikeobj
    
    @timeit
    def preprocess(self, stop_lists=None):
        """Apply pre-processing on training and testing documents
        """
        self.X_train = self._preprocess(self.X_train, stop_lists)
        self.X_test = self._preprocess(self.X_test, stop_lists)
        
        return self
    
    """SUB-TASK 1: Creating preprocess_new to sanitize the data
    """
    @timeit
    def preprocess_new(self, sanitized_data, stop_lists=None):
        """Apply pre-processing on new documents
        """
        sanitized_data = pd.Series(sanitized_data[0])
        sanitized_data = self._preprocess(sanitized_data, stop_lists)
        
        return(sanitized_data)
    
    @timeit
    def vectorize(self, vectorizer=CountVectorizer(ngram_range=(1, 2), max_df=0.5, min_df=2)):
        """Vectorize the train and test data
        """
        X_train = pd.Series([' '.join(x) for x in self.X_train])
        X_test = pd.Series([' '.join(x) for x in self.X_test])
        
        # Vectorize
        """As a part of SUB-TASK 2: vectorizer_obj has been created
        """
        self.vectorizer_obj = vectorizer.fit(X_train)
        self.train_dtm = vectorizer.transform(X_train)
        self.test_dtm = vectorizer.transform(X_test)
        self.feature_names = vectorizer.get_feature_names()
        
        # token frequency count
        freq = np.ravel(self.train_dtm.sum(axis=0))
        vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]
        freq_sorted = dict(zip(vocab, freq))
        freq_dict = OrderedDict(sorted(freq_sorted.items(), key=operator.itemgetter(1)))
        self.wordfreq = pd.DataFrame({'word':freq_dict.keys(), 'count':freq_dict.values()})
        
        return self
    
    @timeit
    def reduce_dimensions(self, k=15000):
        """Reduce dimensionality"""
        ch2 = SelectKBest(chi2, k=15000)
        ch2.fit(self.train_dtm, self.y_train)
        self.train_dtm = ch2.transform(self.train_dtm)
        self.test_dtm = ch2.transform(self.test_dtm)

        # keep selected feature names
        self.feature_names = [self.feature_names[i] for i in ch2.get_support(indices=True)]
        self.feature_selector = ch2
        
        return self


class Haptik(Haptik):
    
    def label_accuracy(self, y_true, y_pred):
        """Compute label accuracy
        """
        res = (y_true == y_pred)
        return (res).sum().sum()/res.size
    
    @timeit
    def classify(self,
                 model=OneVsRestClassifier(MultinomialNB())):
        """Fit a model to the dataset
        """
        # Clone local copies
        X_train = copy.deepcopy(self.train_dtm)
        X_test = copy.deepcopy(self.test_dtm)
        y_train = copy.deepcopy(self.y_train)
        y_test = copy.deepcopy(self.y_test)
        

        # Fit and predict
        model.fit(X_train, y_train)
        if isinstance(model, GridSearchCV): # assumes GridSearchCV() has been imported
            self.gridsearchcv = copy.deepcopy(model)
            model = copy.deepcopy(model.best_estimator_)
            
        self.y_pred_class = model.predict(X_test)
        self.model = model
        
        # Compute metrics
        self.accuracy_subset = metrics.accuracy_score(y_test, self.y_pred_class)
        self.accuracy_label = self.label_accuracy(y_test, self.y_pred_class)
        self.c_report = metrics.classification_report(y_test, self.y_pred_class)

        return self
    
    def predict(self, new_data):
        predictions = self.model.predict(new_data)
        
        return(predictions)
    
    def results(self):
        """Print accuracy and other metrics
        """
        print('accuracy_label: ', self.accuracy_label)
        print('accuracy_subset: ', self.accuracy_subset)
        print('classification report: \n', self.c_report)
        return self
    
    def results_cv(self):
        """Print the best models and the details of cv rounds
        """
        print('cv results: \n', pd.DataFrame(self.gridsearchcv.cv_results_))
        print('best parameters: \n', self.gridsearchcv.best_params_)
        return self


