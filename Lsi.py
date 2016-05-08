from Data import Data
from sklearn.feature_extraction import text
from gensim import corpora, models, similarities


def create_dictionary(rawSamples):
    texts = [[word for word in question[0].lower().split()] for question in rawSamples]
    dictionary = corpora.Dictionary(texts)
    dictionary.save('gensim_dictionary.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('gensim_corpus.mm', corpus)


data = Data()
data.load_statistics()
data.parse_questions()
rawData = data.get_rawsamples()
print rawData[19445][0]
create_dictionary(rawData)
