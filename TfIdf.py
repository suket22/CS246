from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity


class TfIdf:

    def __init__(self):
        self.tfidf_matrix = None
        pass

    def create_tfidf_matrix(self, rawSamples):
        count_vect = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
        question_list = []
        for key in rawSamples:
            question_list.append(rawSamples[key][0])
        trained_counts = count_vect.fit_transform(question_list)
        self.tfidf_matrix = TfidfTransformer().fit_transform(trained_counts)
        print "Shape of TF-IDF matrix ", self.tfidf_matrix.shape
        print "\nNumber of reduced terms : ", self.tfidf_matrix.shape[1]
        # IMPORTANT - Ordering of question strings in TF-IDF matrix is same as order in rawSamples.

    def calc_cosine_similarity(self, index1, index2):
        return cosine_similarity(self.tfidf_matrix[index1].todense(), self.tfidf_matrix[index2].todense())
        # Compare similarity with some threshold here.
        pass
