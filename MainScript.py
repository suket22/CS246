from Data import Data
from Lsi import fire_lsi
from calcAccuracy import test_accuracy

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

# LSI --> Generate output file
filename = 'sample_output_lsi.out'
fire_lsi(data,filename)

# Testing LSI
test_accuracy(filename)

# LDA --> Generate output file

# Testing LDA