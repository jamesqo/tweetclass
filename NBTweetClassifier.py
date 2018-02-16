import logging as log
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

def _extract_vocab(docs):
    terms = [term for doc in docs for term in doc.split()]
    return list(set(terms))

class NBTweetClassifier(BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        tweets = X['TweetText']
        num_instances = X.shape[0]

        self.vocab_ = _extract_vocab(tweets)

        self.classes_ = list(set(y))
        self.class_freqs_ = y.value_counts()
        self.class_probas_ = self.class_freqs_ / num_instances
        self.class_term_counts_ = {}

        self.term_freqs_ = {}
        self.term_probas_ = {}

        for class_ in self.classes_:
            tweets_for_class = [tweet for index, tweet in enumerate(tweets) if y[index] == class_]
            terms_for_class = ''.join(tweets_for_class).split()
            self.class_term_counts_[class_] = len(terms_for_class)
            denom = self.class_term_counts_[class_] + len(self.vocab_)

            log.debug("Computing T_{ct} for all terms in class '%s'", class_)
            for term in terms_for_class:
                key = (term, class_)
                self.term_freqs_[key] = self.term_freqs_.get(key, 0) + 1

            log.debug("Computing P(t|c) for all terms in class '%s'", class_)
            for term in self.vocab_:
                key = (term, class_)
                if key in self.term_freqs_:
                    term_freq = self.term_freqs_[key]
                    self.term_probas_[key] = (term_freq + 1) / denom
    
    def predict(self, X):
        tweets = X['TweetText']
        y = [self._predict(tweet, index) for index, tweet in enumerate(tweets)]
        return pd.Series(y)
    
    def _predict(self, tweet, tweet_index):
        return max(self.classes_, key=lambda class_: self._score(class_, tweet, tweet_index))
    
    def _score(self, class_, tweet, tweet_index):
        log.debug("Computing log P(c|d) for class '%s' for tweet #%d", class_, tweet_index)

        score = np.log(self.class_probas_[class_])
        for term in tweet.split():
            key = (term, class_)
            if key in self.term_probas_:
                proba = self.term_probas_[key]
            else:
                denom = self.class_term_counts_[class_] + len(self.vocab_)
                term_freq = 0
                proba = (term_freq + 1) / denom
            score += np.log(proba)
        return score
