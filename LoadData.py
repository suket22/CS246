import json
import numpy
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import text
from TfIdf import TfIdf


class LoadData:

    rawSampleCount = 0
    rawSamples = {}  # Map of Question Key -> Question Text

    trainingCount = 0
    trainingData = None  # Define numpy array once size is known

    testCount = 0
    testData = None  # Define numpy array once size is known

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
                    list_data = []  # Add extra data here.
                    list_data.append(j[u'question_text'])
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
                    pass
                line_count += 1

    # Create Vocabulary from Raw Samples
    def parse_questions(self):
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')
        for questions_key in self.rawSamples:
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

    # For Debugging
    def load_statistics(self):
        print "Number of Raw Samples - ", len(self.rawSamples)
        print "Number of Training Samples - ", len(self.trainingData)
        print "Number of Test Samples - ", len(self.testData)

    def get_rawsamples(self):
        return self.rawSamples

    # Modify the threshold for defining whether two documents are similar or not.
    def boolean_similarity(self, similarity):
        if similarity > 0.80:
            return 1
        else:
            return 0

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

            similarity = tfidf.calc_cosine_similarity(index1, index2)[0][0]  # Cosine similarity returns a list of a list.
            f.write(self.testData[i][0] + " " + self.testData[i][1] + " " + str(self.boolean_similarity(similarity)) + "\n")
        f.close()

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
# Step 5 - Call cosine similarity for test pairs.
ld.test_data(tfidf)