import json
import numpy
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import text
from TfIdf import TfIdf
from nltk.corpus import wordnet as wn
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import linear_model


class LoadData:
    rawSampleCount = 0
    rawSamples = {}  # Map of Question Key -> Question Text

    trainingCount = 0
    trainingData = None  # Define numpy array once size is known

    testCount = 0
    testData = None  # Define numpy array once size is known
    threshold = 0.75
    model_logistic = None
    model_linear = None
    model_SVM = None
    rawSample_indexhash = dict()

    def __init__(self):
        self.read_data()
        pass

    # Load information from file into memory here
    def read_data(self):
        with open('duplicate_sample.in') as f:
            line_count = 0
            target_count = 0
            training_index = 0
            test_index = 0
            for line in f:
                if line_count == 0:
                    self.rawSampleCount = int(line)
                elif line_count <= self.rawSampleCount:
                    j = json.loads(line)
                    list_data = list()

                    # Adding question text
                    list_data.append(j[u'question_text'])

                    # Adding context topic
                    if j[u'context_topic'] is not None:
                        list_data.append(j[u'context_topic'][u'name'])
                    else:
                        list_data.append("")

                    # Adding all other topics together as single string
                    topics = ""
                    for i in range(0, len(j[u'topics'])):
                        topics += (j[u'topics'][i][u'name'] + " ")

                    list_data.append(topics)
                    self.rawSamples[j[u'question_key']] = list_data
                elif line_count == self.rawSampleCount + 1:
                    self.trainingCount = int(line)
                    self.trainingData = numpy.zeros(shape=(self.trainingCount, 3), dtype='a48')  # 48 is length of a key
                    target_count = line_count + self.trainingCount
                elif line_count <= target_count:
                    split_line = line.split(" ")
                    self.trainingData[training_index][0] = split_line[0]
                    self.trainingData[training_index][1] = split_line[1]
                    self.trainingData[training_index][2] = split_line[2]
                    training_index += 1
                elif line_count == target_count + 1:
                    self.testCount = int(line)
                    self.testData = numpy.zeros(shape=(self.testCount, 2), dtype='a48')
                else:
                    split_line = line.split(" ")
                    self.testData[test_index][0] = split_line[0]
                    self.testData[test_index][1] = split_line[1]
                    test_index += 1
                line_count += 1

    # Create Vocabulary from Raw Samples
    def parse_questions(self):
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')
        for questions_key in self.rawSamples:
            # Stem the Question Text
            question_text = self.rawSamples[questions_key][0]
            words_array = tokenizer.tokenize(question_text)
            question_text = ""
            for word in words_array:
                if word.isnumeric():
                    continue
                if word not in text.ENGLISH_STOP_WORDS:
                    word = stemmer.stem(word)
                word = stemmer.stem(word)
                question_text += (word + " ")
            self.rawSamples[questions_key][0] = question_text

            # Stem the topic names
            topics_text = self.rawSamples[questions_key][2]
            words_array = tokenizer.tokenize(topics_text)
            topics_text = ""
            for word in words_array:
                if word.isnumeric():
                    continue
                if word not in text.ENGLISH_STOP_WORDS:
                    word = stemmer.stem(word)
                word = stemmer.stem(word)
                topics_text += (word + " ")
            self.rawSamples[questions_key][2] = topics_text

    # For Debugging
    def load_statistics(self):
        print "Number of Raw Samples - ", len(self.rawSamples)
        print "Number of Training Samples - ", len(self.trainingData)
        print "Number of Test Samples - ", len(self.testData)

    def get_rawsamples(self):
        return self.rawSamples

    # Modify the threshold for defining whether two documents are similar or not.
    def boolean_similarity(self, similarity):
        if similarity > self.threshold:
            return 1
        else:
            return 0

    def heuristic_synscore(self, question_key1, question_key2):
        question_text1 = self.rawSamples[question_key1][0].split()
        question_text2 = self.rawSamples[question_key2][0].split()
        synonym_count = 0
        for word in question_text1:
            matched_lemmas = list()
            synonyms = wn.synsets(word)  # Find synonym sets for each word in question 1
            if len(synonyms) == 0:
                continue
            for i in range(0, len(synonyms)):
                lemma_names = synonyms[i].lemma_names() # Consider synonyms from synonym set 'i'
                for lemma in lemma_names:   # For each such synonym
                    # If synonym is in question 2
                    if lemma in question_text2 and lemma not in matched_lemmas and lemma not in question_text1:
                        matched_lemmas.append(lemma)
                        # DEBUG print "Lemma match!", lemma
                        synonym_count += 1
        return synonym_count

    # Original Function -> Writes test data similarity prediction into file
    def test_data(self, tfidf):
        index1 = -1
        index2 = -1
        f = open("sample_output.out", "w")
        for i in range(0, len(self.testData)):
            # Currently Linear search (Another level of Hash to be implemented here)
            j = 0
            for key in self.rawSamples:
                if key == self.testData[i][0]:
                    index1 = j
                elif key == self.testData[i][1]:
                    index2 = j
                j += 1

            similarity1 = tfidf.calc_cosine_similarity(index1, index2)[0][0]  # For Question Text
            similarity2 = tfidf.calc_cosine_similarity_topics(index1, index2)[0][0]  # For Topics
            heuristic = 0
            if similarity1 < self.threshold:  # Don't run heuristic if tf-idf is very confident.
                heuristic = self.heuristic_synscore(self.testData[i][0], self.testData[i][1])
                similarity1 += (heuristic * 0.15)
            if similarity2 < self.threshold - 0.3:    # If elements have low topic similarity, drop their overall similarity
                similarity1 -= (similarity2 * 0.20)
            f.write(self.testData[i][0] + " " + self.testData[i][1] + " " + str(self.boolean_similarity(similarity1)) + "\n")
        f.close()

    # Original Function -> Tests accuracy based on some prediction
    def test_accuracy(self):
        y = []  # Actual values
        yhat = []   # Predicted values
        truep, truen = 0, 0
        falsep, falsen = 0, 0

        with open('sample_output.out') as f:    # File we create in test_data()
            for line in f:
                yhat.append(line.split()[2])
        with open('duplicate_sample.out') as f:  # Answers file given to us by Quora
            for line in f:
                y.append(line.split()[2])

        correct = 0
        for i in range(0, len(y)):
            if y[i] == yhat[i] and y[i] == "1":
                truep += 1
            elif y[i] == yhat[i] and y[i] == "0":
                truen += 1
            elif y[i] != yhat[i] and yhat[i] == "1":
                falsep += 1
            else:
                falsen += 1
        print "True Positive - ", truep
        print "True Negative - ", truen
        print "False Positive - ", falsep
        print "False Negative - ", falsen
        precision = truep * 1.0 / (truep + falsep)
        recall = truep * 1.0 / (truep + falsen)
        f_score = (precision * recall * 2) / (precision + recall)
        accuracy = (truep + truen) * 1.0 / len(y)
        print "Precision is", precision
        print "Recall is", recall
        print "F Score is", f_score
        print "Accuracy is ", accuracy

    def create_training_data(self, tfidf):
        index1 = -1
        index2 = -1
        f = open("sample_output.out", "w")

        # Creating additional hash for (QID -> Index of QID in rawSamples)
        j = 0
        for key in self.rawSamples:
            self.rawSample_indexhash[key] = j
            j += 1

        for i in range(0, len(self.trainingData)):
            index1 = self.rawSample_indexhash[self.trainingData[i][0]]
            index2 = self.rawSample_indexhash[self.trainingData[i][1]]
            similarity1 = tfidf.calc_cosine_similarity(index1, index2)[0][0]  # For Question Text
            similarity2 = tfidf.calc_cosine_similarity_topics(index1, index2)[0][0]  # For Topics
            heuristic = self.heuristic_synscore(self.trainingData[i][0], self.trainingData[i][1])
            f.write(self.trainingData[i][0] + "," + self.trainingData[i][1] + "," + self.trainingData[i][2][0] + "," + str(similarity1) + "," + str(similarity2) + "," + str(heuristic) + "\n")
        f.close()

    def create_model(self):
            data = pd.read_csv('sample_output.out', header=None,
                               names=['QID1', 'QID2', 'Similarity', 'TFIDF', 'Topic', 'Synonym'])
            print data.head()
            print data.shape

            feature_cols = ['TFIDF', 'Topic', 'Synonym']
            X = data[feature_cols]
            y = data.Similarity

            # instantiate, fit
            # self.model_SVM = svm.SVC()
            # self.model_SVM.fit(X, y)
            self.model_logistic = LogisticRegression()
            self.model_logistic.fit(X, y)
            # self.model_linear = linear_model.LinearRegression()
            # self.model_linear.fit(X, y)

            # Model creation
            print "Accuracy on Training Data", self.model_logistic.score(X, y)

    def create_testing_data(self, tfidf):
        index1 = -1
        index2 = -1
        f = open("sample_output.out", "w")
        for i in range(0, len(self.testData)):
            index1 = self.rawSample_indexhash[self.testData[i][0]]
            index2 = self.rawSample_indexhash[self.testData[i][1]]
            similarity1 = tfidf.calc_cosine_similarity(index1, index2)[0][0]  # For Question Text
            similarity2 = tfidf.calc_cosine_similarity_topics(index1, index2)[0][0]  # For Topics
            heuristic = self.heuristic_synscore(self.testData[i][0], self.testData[i][1])
            f.write(self.testData[i][0] + "," + self.testData[i][1] + "," + str(similarity1) + ","
                    + str(similarity2) + "," + str(heuristic) + "\n")

    def test_model(self):
            data = pd.read_csv('sample_output.out', header=None,
                               names=['QID1', 'QID2', 'TFIDF', 'Topic', 'Synonym'])
            print data.head()
            print data.shape

            feature_cols = ['TFIDF', 'Topic', 'Synonym']
            X = data[feature_cols]

            # Predict Values
            yhat = self.model_logistic.predict(X)

            # Actual Values
            y = []
            with open('duplicate_sample.out') as f:  # Answers file given to us by Quora
                for line in f:
                    y.append(int(line.split()[2]))

            # Calculate accuracy!
            truep = 0
            truen = 0
            falsep = 0
            falsen = 0
            for i in range(0, len(y)):
                if y[i] == yhat[i] and y[i] == 1:
                    truep += 1
                elif y[i] == yhat[i] and y[i] == 0:
                    truen += 1
                elif y[i] != yhat[i] and yhat[i] == 1:
                    falsep += 1
                else:
                    falsen += 1
            print "True Positive - ", truep
            print "True Negative - ", truen
            print "False Positive - ", falsep
            print "False Negative - ", falsen
            precision = truep * 1.0 / (truep + falsep)
            recall = truep * 1.0 / (truep + falsen)
            f_score = (precision * recall * 2) / (precision + recall)
            accuracy = (truep + truen) * 1.0 / len(y)
            print "Precision is", precision
            print "Recall is", recall
            print "F Score is", f_score
            print "Accuracy is ", accuracy

# Launch Codes

# Step 1 - Data is read from file "duplicate_sample.in" in same directory
ld = LoadData()
tfidf = TfIdf()
# Step 2 - Verify correct loading
ld.load_statistics()
# Step 3 - Parse questions (stop word removal, stemming)
ld.parse_questions()
# Step 4 - Create tf-idf matrix for all documents
tfidf.create_tfidf_matrix(ld.get_rawsamples())
tfidf.create_tfidf_topics(ld.get_rawsamples())
# Step 5 - Call cosine similarity for test pairs and accuracy
# ld.test_data(tfidf)
# ld.test_accuracy()

# ML Method
# Create training data and train a model using logistic regression.
print "Forming Training Data set"
ld.create_training_data(tfidf)
print "Creating Model"
ld.create_model()
print "Forming Testing Data set"
ld.create_testing_data(tfidf)
print "Testing Model!"
ld.test_model()


