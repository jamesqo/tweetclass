from sklearn.base import BaseEstimator

def _extract_vocab(docs):
    terms = [term for doc in docs for term in doc.split()]
    return list(set(terms))

class NBTweetClassifier(BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        tweets = X['TweetText']
        num_instances = X.shape[0]

        self.vocab_ = _extract_vocab(tweets)

        self.class_probas_ = {}
        self.term_probas_ = {}

        for class_ in Y.columns.values:
            class_proba = sum(Y[class_]) / num_instances
            self.class_probas_[class_] = class_proba

            tweets_for_class = [tweet for index, tweet in enumerate(tweets) if Y[class_][index]]
            terms_for_class = ''.join(tweets_for_class).split()
            denom = len(terms_for_class) + len(self.vocab_)

            for term in self.vocab_:
                term_count = terms_for_class.count(term)
                self.term_probas_[(term, class_)] = (term_count + 1) / denom

    def predict(self, X):
        pass
