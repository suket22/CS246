from gensim import models
from Data import Data
import numpy as np
from numpy import linalg as LA
from scipy import spatial
from calcAccuracy import test_accuracy

x = []


def isSimilar(model, questionText1, questionText2, useCosine):
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

    if useCosine:
        cosine_sim = 1 - spatial.distance.cosine(question1_vector, question2_vector)
        x.append(cosine_sim)
        print cosine_sim
        if cosine_sim > 0.90:
            return 1
        return 0

    else:
        difference = abs(question1_vector - question2_vector)
        norm = LA.norm(difference)
        print norm
        if norm < 0.1:
            return 1
        return 0


def get_cosinefile(filename):
    f = open(filename, "w")
    for item in x:
        f.write(str(item) + "\n")
    f.close()


def fire_word2vec(data, model, filename, useCosine):
    f = open(filename, "w")
    for i in range(0, len(data.testData)):
        print ">>> Data Sample " + str(i + 1)
        question1Index = data.keyToIndex[data.testData[i][0]]
        question2Index = data.keyToIndex[data.testData[i][1]]
        questionText1 = data.rawSamples[question1Index][0]
        questionText2 = data.rawSamples[question2Index][0]
        duplicate = isSimilar(model, questionText1, questionText2, useCosine)
        f.write(data.testData[i][0] + " " + data.testData[i][1] + " " + str(duplicate) + "\n")
    f.close()

data = Data()
data.load_statistics()
data.parse_questions_no_stemming()

model = models.Word2Vec.load_word2vec_format('googleWord2Vec.bin', binary=True)

filename = 'sample_output_word2vec.out'
useCosine = True
fire_word2vec(data, model, filename, useCosine)
get_cosinefile('word2vec_cosine.txt')
print('---------------------------')
test_accuracy(filename)
