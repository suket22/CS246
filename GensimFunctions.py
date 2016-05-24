from gensim import corpora, models, similarities

def create_dictionary(rawSamples):
    texts = [[word for word in question[0].lower().split()] for question in rawSamples]
    dictionary = corpora.Dictionary(texts)
    dictionary.save('gensim_dictionary.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('gensim_corpus.mm', corpus)
    return (dictionary,corpus)

def load_dictcorpus():
    dictionary = corpora.Dictionary.load('gensim_dictionary.dict')
    corpus = corpora.MmCorpus('gensim_corpus.mm')
    return (dictionary,corpus)

# LSI MODEL
def create_lsimodelindex(dictionary, corpus, numOfTopics):
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=numOfTopics)
    lsi.save('gensim_lsimodel.lsi')
    index = similarities.MatrixSimilarity(lsi[corpus])
    index.save('gensim_lsiindex.index')
    return (lsi,index)

def load_lsimodelindex():
    lsiModel = models.LsiModel.load('gensim_lsimodel.lsi')
    lsiIndex = similarities.MatrixSimilarity.load('gensim_lsiindex.index')
    return (lsiModel,lsiIndex)

# LDA MODEL
def create_ldamodelindex(dictionary,corpus,numOfTopics):
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=numOfTopics)
    lda.save('gensim_ldamodel.lda')
    index = similarities.MatrixSimilarity(lda[corpus])
    index.save('gensim_ldaindex.index')
    return (lda,index)

def load_ldamodelindex():
    ldaModel = models.LdaModel.load('gensim_ldamodel.lda')
    ldaIndex = similarities.MatrixSimilarity.load('gensim_ldaindex.index')
    return (ldaModel,ldaIndex)

# RP MODEL
def create_rpmodelindex(dictionary,corpus,numOfTopics):
    rp = models.RpModel(corpus, num_topics=numOfTopics)
    rp.save('gensim_rpmodel.rp')
    index = similarities.MatrixSimilarity(rp[corpus])
    index.save('gensim_rpindex.index')
    return (rp,index)

def load_rpmodelindex():
    rpModel = models.RpModel.load('gensim_rpmodel.rp')
    rpIndex = similarities.MatrixSimilarity.load('gensim_rpindex.index')
    return (rpModel,rpIndex)

# HDP MODEL
def create_hdpmodelindex(dictionary,corpus):
    hdp = models.HdpModel(corpus, id2word=dictionary)
    hdp.save('gensim_hdpmodel.hdp')
    index = similarities.MatrixSimilarity(hdp[corpus])
    index.save('gensim_hdpndex.index')
    return (hdp,index)

def load_hdpmodelindex():
    hdpModel = models.HdpModel.load('gensim_hdpmodel.hdp')
    hdpIndex = similarities.MatrixSimilarity.load('gensim_hdpindex.index')
    return (hdpModel,hdpIndex)