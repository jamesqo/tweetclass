import logging as log

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

        self.term_freqs_ = {}
        self.term_probas_ = {}

        for class_ in self.classes_:
            tweets_for_class = [tweet for index, tweet in enumerate(tweets) if y[index] == class_]
            terms_for_class = ''.join(tweets_for_class).split()
            denom = len(terms_for_class) + len(self.vocab_)

            log.debug("Computing T_{ct} for all terms in class '%s'", class_)
            for term in terms_for_class:
                key = (term, class_)
                self.term_freqs_[key] = self.term_freqs_.get(key, 0) + 1

            log.debug("Computing P(t|c) for all terms in class '%s'", class_)
            for term in self.vocab_:
                key = (term, class_)
                term_freq = self.term_freqs_.get(key, 0)
                self.term_probas_[key] = (term_freq + 1) / denom

    def predict(self, X):
        pass
