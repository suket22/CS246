from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

class TfIdf:

    def __init__(self):
        self.tfidf_matrix = None
        self.tfidf_matrix_topics = None
        pass

    def create_tfidf_matrix(self, rawSamples):
        # This helps us calculate the vocabulary, given a list of all questions.
        count_vect = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
        question_list = []
        for key in rawSamples:
            question_list.append(rawSamples[key][0])
        trained_counts = count_vect.fit_transform(question_list)
        self.tfidf_matrix = TfidfTransformer().fit_transform(trained_counts)
        # # SVD is dropping the accuracy
        # lsa = TruncatedSVD(n_components=50)
        # dtm_lsa = lsa.fit_transform(self.tfidf_matrix)
        # dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
        # self.tfidf_matrix = sparse.csr_matrix(dtm_lsa)
        print "Shape of TF-IDF matrix ", self.tfidf_matrix.shape
        print "\nNumber of reduced terms : ", self.tfidf_matrix.shape[1]
        # IMPORTANT - Ordering of question strings in TF-IDF matrix is same as order in rawSamples.

    # TF-IDF matrix only for topics, context_topics.
    def create_tfidf_topics(self, rawSamples):
        count_vect = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
        topic_list = []
        for key in rawSamples:
            topic = ""
            if rawSamples[key][1] != "":
                topic += rawSamples[key][1] + " "
            topic += rawSamples[key][2]
            topic_list.append(topic)
        trained_counts = count_vect.fit_transform(topic_list)

        self.tfidf_matrix_topics = TfidfTransformer().fit_transform(trained_counts)
        print "Shape of TF-IDF matrix ", self.tfidf_matrix_topics.shape
        print "\nNumber of reduced terms : ", self.tfidf_matrix_topics.shape[1]
        # IMPORTANT - Ordering of question strings in TF-IDF matrix is same as order in rawSamples.

    def calc_cosine_similarity(self, index1, index2):
        return cosine_similarity(self.tfidf_matrix[index1].todense(), self.tfidf_matrix[index2].todense())

    def calc_cosine_similarity_topics(self, index1, index2):
        return cosine_similarity(self.tfidf_matrix_topics[index1].todense(), self.tfidf_matrix_topics[index2].todense())
