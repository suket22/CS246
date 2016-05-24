import os.path
from Data import Data
import GensimFunctions as gen
from calcAccuracy import test_accuracy


def isSimilar(corpus, rpModel, rpIndex, questionKey, testKey, keyToIndex, indexToKey):
    i = keyToIndex[questionKey]
    vecRp = rpModel[corpus[i]]
    similarQuestions = rpIndex[vecRp]
    similarQuestions = sorted(enumerate(similarQuestions), key=lambda item: -item[1])
    testIndex = keyToIndex[testKey]
    for top10 in range(0, 10):
        if similarQuestions[top10][0] == testIndex:
            return 1
    return 0


def fire_rp(data, corpus, rpModel, rpIndex, filename):
    f = open(filename, "w")
    for i in range(0, len(data.testData)):
        print ">>> Data Sample " + str(i + 1)
        questionKey = data.testData[i][0]
        testKey = data.testData[i][1]
        duplicate = isSimilar(corpus, rpModel, rpIndex, questionKey, testKey, data.keyToIndex, data.indexToKey)
        f.write(data.testData[i][0] + " " + data.testData[i][1] + " " + str(duplicate) + "\n")
    f.close()


data = Data()
data.load_statistics()
data.parse_questions()

fname_rpModel = 'gensim_rpmodel.rp'
fname_rpIndex = 'gensim_rpindex.index'
fname_corpus = 'gensim_corpus.mm'
fname_dictionary = 'gensim_dictionary.dict'

if (not (os.path.isfile(fname_corpus) and os.path.isfile(fname_dictionary))):
    (dictionary, corpus) = gen.create_dictionary(data.rawSamples)
else:
    (dictionary, corpus) = gen.load_dictcorpus()
if (not (os.path.isfile(fname_rpIndex) and os.path.isfile(fname_rpModel))):
    (rpModel, rpIndex) = gen.create_rpmodelindex(dictionary, corpus, 100)
else:
    (rpModel, rpIndex) = gen.load_rpmodelindex()

filename = 'sample_output_rp.out'
fire_rp(data, corpus, rpModel, rpIndex, filename)

print('---------------------------')
test_accuracy(filename)
