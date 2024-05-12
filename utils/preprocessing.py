import re
import numpy as np
import pandas as pd
from gensim.utils import tokenize
from gensim.models import Word2Vec
import torch
from nlp_id import StopWord, Lemmatizer

class TextPreprocessing():
    """class for preprocessing the text tobe ready for predict"""
    
    def __init__(self):
        self.w2v_model = Word2Vec.load('./model/pretrained_model/w2v_model.w2v')
    
    def __text_cleaning(self, text):
        """removing punctuations and case folding to be lowercase"""
        result = text.lower() # apply lowercas
        result = result.replace('-', ' ') # get rid of punctuations
        result = result.replace('+', ' ')
        result = result.replace('..', ' ')
        result = result.replace('.', ' ')
        result = result.replace(',', ' ')
        result = result.replace('\n', ' ') # get rid of new line
        result = re.findall('[a-z\s]', result, flags=re.UNICODE) # only use text character (a-z) and space
        result = "".join(result)
        final = " ".join(result.split())
        
        return final    # clean text
    
    def __remove_stopwords(self, text):
        """removing all stopword that contained on text"""
        stopword = StopWord()
        return stopword.remove_stopword(text)
    
    def __lemmatization(self, text):
        """lemmatizing the text"""
        lemmatizer = Lemmatizer()

        return lemmatizer.lemmatize(text=text)

    def __tokenize(self, text):
        """tokenizing text method"""
        text = list(tokenize(text))
        return text
    
    def __vectorize(self, text):
        """vectorizing text method"""
        vectorized_text = np.array([self.w2v_model.wv.get_vector(word) for word in text if word in self.w2v_model.wv])
        return vectorized_text
    
        
    
    def preprocess_text(self, text):
        """the main function for processing text"""
        # Cleaning text
        text = self.__text_cleaning(text)
        # remove stopwords from text
        text = self.__remove_stopwords(text)
        # lemmatization
        text = self.__lemmatization(text=text)
        # tokenize text
        text = self.__tokenize(text)
        # vectorize text
        text = self.__vectorize(text)
        # add 1 dimension
        text = np.expand_dims(text, axis=0)
        # convert to torch.tensor
        text = torch.from_numpy(text)

        return text
        