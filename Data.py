import numpy
import json
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import text


class Data:
    rawSampleCount = 0
    rawSamples = []
    keyToIndex = {}  # Map of Question Key -> List Index
    indexToKey = {}  # Map of List Index -> Question Key

    trainingCount = 0
    trainingData = None  # Define numpy array once size is known

    testCount = 0
    testData = None  # Define numpy array once size is known

    def __init__(self):
        self.read_data()
        pass

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

                    self.keyToIndex[j[u'question_key']] = line_count - 1
                    self.rawSamples.insert(line_count - 1, list_data)
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
        self.indexToKey = {v: k for k, v in self.keyToIndex.items()}

    # Create Vocabulary from Raw Samples
    def parse_questions(self):
        stemmer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')
        for index in range(0,len(self.rawSamples)):
            words_array = tokenizer.tokenize(self.rawSamples[index][0].lower())
            question_text = ""
            for word in words_array:
                if word.isnumeric():
                    continue
                if word not in text.ENGLISH_STOP_WORDS:
                    word = stemmer.stem(word)
                    question_text += (word + " ")
            self.rawSamples[index][0] = question_text

    # For Debugging
    def load_statistics(self):
        print "Number of Raw Samples - ", len(self.rawSamples)
        print "Number of Training Samples - ", len(self.trainingData)
        print "Number of Test Samples - ", len(self.testData)

    # Returning Raw Samples - List of Lists
    def get_rawsamples(self):
        return self.rawSamples

#data = Data()
#data.load_statistics()
#data.parse_questions()
#rawData = data.get_rawsamples()
#print rawData[0][0]
