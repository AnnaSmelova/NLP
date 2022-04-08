from collections import OrderedDict, Counter
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        # <MY_CODE>
        voc = Counter()
        for train_text in X:
            train_text = train_text.split(' ')
            for word in train_text:
                voc[word] += 1
        #print(f'voc={voc}')
        #self.bow = voc.most_common(self.k)
        self.bow = sorted(voc, key = lambda x:voc[x], reverse = True)[:self.k]
        #print(f'bow={self.bow}')
        # raise NotImplementedError

        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """

        result = None
        # <MY_CODE>
        result = np.zeros(self.k)
        voc = {word : i for i, word in enumerate(self.bow)}
        #print(f'voc={voc}')
        for word in text.split():
          if word in voc:
            result[voc[word]] += 1
        # raise NotImplementedError
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # <MY_CODE>
        voc = Counter()
        for train_text in X:
            train_text = train_text.split(' ')
            for word in train_text:
                voc[word] += 1
        
        vocabulary = sorted(voc, key = lambda x:voc[x], reverse = True)[:self.k]
        
        for word in vocabulary:
            num = 0
            for train_text in X:
                if word in train_text:
                    num += 1
            self.idf[word] = np.log(len(X) / num)
        # raise NotImplementedError

        # fit method must always return self
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """

        result = None
        # <MY_CODE>
        result = []
        voc = Counter(text.split())
        for word in self.idf:
            result.append((voc[word] / len(text.split())) * self.idf[word])
        # raise NotImplementedError
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
