import os.path
from Data import Data
import GensimFunctions as gen
from calcAccuracy import test_accuracy


def isSimilar(corpus, hdpModel, hdpIndex, questionKey, testKey, keyToIndex, indexToKey):
    i = keyToIndex[questionKey]
    vecHdp = hdpModel[corpus[i]]
    similarQuestions = hdpIndex[vecHdp]
    similarQuestions = sorted(enumerate(similarQuestions), key=lambda item: -item[1])
    testIndex = keyToIndex[testKey]
    for top10 in range(0, 10):
        if similarQuestions[top10][0] == testIndex:
            return 1
    return 0


def fire_hdp(data, corpus, hdpModel, hdpIndex, filename):
    f = open(filename, "w")
    for i in range(0, len(data.testData)):
        print ">>> Data Sample " + str(i + 1)
        questionKey = data.testData[i][0]
        testKey = data.testData[i][1]
        duplicate = isSimilar(corpus, hdpModel, hdpIndex, questionKey, testKey, data.keyToIndex, data.indexToKey)
        f.write(data.testData[i][0] + " " + data.testData[i][1] + " " + str(duplicate) + "\n")
    f.close()


data = Data()
data.load_statistics()
data.parse_questions()

fname_hdpModel = 'gensim_hdpmodel.hdp'
fname_hdpIndex = 'gensim_hdpindex.index'
fname_corpus = 'gensim_corpus.mm'
fname_dictionary = 'gensim_dictionary.dict'

if (not (os.path.isfile(fname_corpus) and os.path.isfile(fname_dictionary))):
    (dictionary, corpus) = gen.create_dictionary(data.rawSamples)
else:
    (dictionary, corpus) = gen.load_dictcorpus()
if (not (os.path.isfile(fname_hdpIndex) and os.path.isfile(fname_hdpModel))):
    (hdpModel, hdpIndex) = gen.create_hdpmodelindex(dictionary, corpus)
else:
    (hdpModel, hdpIndex) = gen.load_hdpmodelindex()

filename = 'sample_output_hdp.out'
fire_hdp(data, corpus, hdpModel, hdpIndex, filename)

print('---------------------------')
test_accuracy(filename)
