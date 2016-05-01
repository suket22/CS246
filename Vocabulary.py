class Vocabulary:

    words = {}

    def __init__(self):
        pass

    def add_word(self, word):
        if word in self.words:
            self.words[word] += 1
        else:
            self.words[word] = 1

    # Vocabulary Size = Size of TF-IDF vector
    def get_vocabulary_size(self):
        return len(self.words)

    def get_vocabulary(self):
        return self.words
