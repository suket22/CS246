from gensim import models
from Data import Data
import numpy as np
from scipy import spatial
from calcAccuracy import test_accuracy


def isSimilar(model, questionText1, questionText2):
    question1_vector = np.zeros((300), dtype=np.float32)
    question2_vector = np.zeros((300), dtype=np.float32)
    for word in questionText1.split():
        try:
            question1_vector += model[word]
        except:
            pass
    for word in questionText2.split():
        try:
            question2_vector += model[word]
        except:
            pass
    cosine_sim = 1 - spatial.distance.cosine(question1_vector, question2_vector)
    print cosine_sim
    if cosine_sim > 0.90:
        return 1
    return 0


def fire_word2vec(data, model, filename):
    f = open(filename, "w")
    for i in range(0, len(data.testData)):
        print ">>> Data Sample " + str(i + 1)
        question1Index = data.keyToIndex[data.testData[i][0]]
        question2Index = data.keyToIndex[data.testData[i][1]]
        questionText1 = data.rawSamples[question1Index][0]
        questionText2 = data.rawSamples[question2Index][0]
        duplicate = isSimilar(model, questionText1, questionText2)
        f.write(data.testData[i][0] + " " + data.testData[i][1] + " " + str(duplicate) + "\n")
    f.close()


data = Data()
data.load_statistics()
data.parse_questions_no_stemming()

model = models.Word2Vec.load_word2vec_format('googleWord2Vec.bin', binary=True)

filename = 'sample_output_word2vec.out'
fire_word2vec(data, model, filename)

print('---------------------------')
test_accuracy(filename)
