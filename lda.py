from Data import Data
from sklearn.feature_extraction import text
from gensim import corpora, models, similarities


def create_dictionary(rawSamples):
    texts = [[word for word in question[0].lower().split()] for question in rawSamples]
    dictionary = corpora.Dictionary(texts)
    dictionary.save('gensim_dictionary.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('gensim_corpus.mm', corpus)


def create_ldamodelindex():
    dictionary = corpora.Dictionary.load('gensim_dictionary.dict')
    corpus = corpora.MmCorpus('gensim_corpus.mm')
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
    lda.save('gensim_ldamodel.lda')
    index = similarities.MatrixSimilarity(lda[corpus])
    index.save('gensim_ldaindex.index')


def get_similarquestions(questionKey, keyToIndex, indexToKey, testKey):
    corpus = corpora.MmCorpus('gensim_corpus.mm')
    lda = models.LdaModel.load('gensim_ldamodel.lda')
    index = similarities.MatrixSimilarity(lda[corpus])
    i = keyToIndex[questionKey]
    vecLda = lda[corpus[i]]
    similarQuestions = index[vecLda]
    similarQuestions = sorted(enumerate(similarQuestions), key=lambda item: -item[1])
    testIndex = keyToIndex[testKey]

    for ind in similarQuestions:
        if ind[0] == testIndex:
            return 1 if ind[1]>0.98 else 0

    # for top10 in range(0,10):
    #    # print similarQuestions[top10]
    #     if similarQuestions[top10][0] == testIndex:
    #         return 1
    # return 0



    # print similarQuestions[0:5]
    # print similarQuestions[1][1]
    # for iter in range(0,5):
    #     print indexToKey[similarQuestions[iter][0]]
    # findIndex = keyToIndex["AAEAAOZreqlKsP0uctqEK8b58qdLGuPG3LQQ4Hr2dRiSy7KF"]
    # print findIndex
    # for item in similarQuestions:
    #     if item[0] == findIndex:
    #         print(item)


def fire_lda(data,filename):
    rawData = data.get_rawsamples()
    create_dictionary(rawData)
    create_ldamodelindex()
    # questionKey = "AAEAAJU9VfJqzjKYP0FFFuYD4Y5dNxuqwFqYxzfLGTL9wZi2"
    f = open(filename, "w")
    for i in range(0, len(data.testData)):
        print ">>> Data Sample " + str(i + 1)
        questionKey = data.testData[i][0]
        print questionKey
        testKey = data.testData[i][1]
        print testKey
        duplicate = get_similarquestions(questionKey, data.keyToIndex, data.indexToKey, testKey)
        f.write(data.testData[i][0] + " " + data.testData[i][1] + " " + str(duplicate) + "\n")
    f.close()


data = Data()
data.load_statistics()
data.parse_questions()
filename = 'sample_output_lda.out'
fire_lda(data,filename)
