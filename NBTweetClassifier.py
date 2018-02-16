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
        log.debug("fit() called")

        tweets = X['tweet_text']
        num_instances = X.shape[0]

        self.vocab_ = _extract_vocab(tweets)

        self.classes_ = list(set(y))
        self.class_freqs_ = y.value_counts()
        self.class_probas_ = self.class_freqs_ / num_instances
        self.class_term_counts_ = {}
        self.class_term_probas_ = {}

        # PERF: Indexing a pd.Series seems to take a long time, so preprocess it into a list.
        y_list = list(y)

        for class_ in self.classes_:
            tweets_for_class = [tweet for index, tweet in enumerate(tweets) if y_list[index] == class_]
            terms_for_class = ''.join(tweets_for_class).split()
            self.class_term_counts_[class_] = len(terms_for_class)
            denom = self.class_term_counts_[class_] + len(self.vocab_)

            log.debug("Computing T_{ct} for all terms in class '%s'", class_)
            term_freqs = {}
            for term in terms_for_class:
                term_freqs[term] = term_freqs.get(term, 0) + 1

            log.debug("Computing P(t|c) for all terms in class '%s'", class_)
            term_probas = {}
            for term in self.vocab_:
                if term in term_freqs:
                    term_freq = term_freqs[term]
                    term_probas[term] = (term_freq + 1) / denom
            self.class_term_probas_[class_] = term_probas
    
    def predict(self, X):
        log.debug("predict() called")

        tweets = X['tweet_text']
        y = [self._predict(tweet, index) for index, tweet in enumerate(tweets)]
        return pd.Series(y, name='sentiment')
    
    def _predict(self, tweet, tweet_index):
        return max(self.classes_, key=lambda class_: self._score(class_, tweet, tweet_index))
    
    def _score(self, class_, tweet, tweet_index):
        # PERF: This takes too long, even for debug logging.
        #log.debug("Computing log P(c|d) for class '%s' for tweet #%d", class_, tweet_index)

        term_probas = self.class_term_probas_[class_]
        default_proba = 1 / (self.class_term_counts_[class_] + len(self.vocab_))

        score = np.log(self.class_probas_[class_])
        for term in tweet.split():
            proba = term_probas.get(term, default_proba)
            score += np.log(proba)
        return score
