from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer


class TfIdf:

    def __init__(self):
        self.tfidf_matrix = None
        pass

    def add_to_samples(self, count_vect, rawSamples):
        stemmer = PorterStemmer()
        vocabulary = count_vect.get_feature_names() # Fetch existing vocabulary
        question_number = 1
        for key in rawSamples:
            question_text = rawSamples[key][0].split()
            temp = str()
            for word in question_text:  # For each word in the question
                synonyms = wn.synsets(word)  # Find synonym sets
                if len(synonyms) == 0:
                    continue
                lemma_names = synonyms[0].lemma_names() # Consider synonyms only from the first synonym set
                for lemma in lemma_names:   # For each such synonym
                    lemma = stemmer.stem(lemma)
                    if lemma in vocabulary:  # If synonym is in the vocabulary ( i.e exists in some other question)
                        temp += (lemma + " ")   # Add it to current question as well
            rawSamples[key][0] += temp

    def create_tfidf_matrix(self, rawSamples):
        # This helps us calculate the vocabulary, given a list of all questions.
        count_vect = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
        question_list = []
        for key in rawSamples:
            question_list.append(rawSamples[key][0])
        trained_counts = count_vect.fit_transform(question_list)

        # Comment following 6 lines if synonyms are not desired.
        self.add_to_samples(count_vect, rawSamples) # Count_vect contains feature_names which is == the vocabulary.
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
        pass
