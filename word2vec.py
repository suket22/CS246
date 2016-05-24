from gensim import models


def testing():
    model = models.Word2Vec.load_word2vec_format('googleWord2Vec.bin', binary=True)
    yolo = model.most_similar(positive=['woman', 'king'], negative=['man'])
    print yolo


testing()
