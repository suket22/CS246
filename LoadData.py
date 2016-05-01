import json
import numpy
import string
from Vocabulary import Vocabulary


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
    def parse_questions(self, v):
        for questions_key in self.rawSamples:
            # This returns a list of data associated with question key.
            # Currently, question_text is stored as the first element of the list. (Can change to dict later)
            question_text = self.rawSamples[questions_key][0]
            exclude = set(string.punctuation)
            word_array = (''.join(ch for ch in question_text if ch not in exclude)).split()
            for word in word_array:
                v.add_word(word)
        pass

    # For Debugging
    def load_statistics(self):
        print "Number of Raw Samples - ", len(self.rawSamples)
        print "Number of Training Samples - ", len(self.trainingData)
        print "Number of Test Samples - ", len(self.testData)


# Launch File
ld = LoadData()
ld.load_statistics()
v = Vocabulary()
ld.parse_questions(v)
print "Length of TF-IDF vector -", v.get_vocabulary_size()
