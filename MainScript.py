from Data import Data
from Lsi import fire_lsi
from Lda import fire_lda
from calcAccuracy import test_accuracy
import GensimFunctions as gen

# Load Data and parse it
data = Data()
data.load_statistics()
data.parse_questions()

# TFIDF --> Generate output file

# Testing TFIDF

# TFIDF with Synonymns --> Generate output file

# Testing TFIDF with Synonymns

# Context Topics --> Generate output file

# Testing Context Topics

# Generating Dictionary and Corpus for gensim 
(dictionary, corpus) = gen.create_dictionary(data.rawSamples)

# --------------------------LSI-------------------------------
# LSI Model Building
numOfTopics = 100
(lsiModel, lsiIndex) = gen.create_lsimodelindex(dictionary, corpus, numOfTopics)

# LSI --> Generate output file
filename = 'sample_output_lsi.out'
fire_lsi(data, corpus, lsiModel, lsiIndex, filename)

# Testing LSI
test_accuracy(filename)

# --------------------------LDA-------------------------------
# LDA Model Building
numOfTopics = 100
(ldaModel, ldaIndex) = gen.create_ldamodelindex(dictionary, corpus, numOfTopics)

# LDA --> Generate output file
filename = 'sample_output_lda.out'
fire_lda(data, corpus, ldaModel, ldaIndex, filename)

# Testing LDA
test_accuracy(filename)

# --------------------------LDA-------------------------------
# LDA Model Building