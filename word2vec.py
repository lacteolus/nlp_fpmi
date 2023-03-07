import re
import string

import numpy
import numpy as np
import torch


default_window_size = 5
default_vector_size = 100


def clean(inp: str) -> str:
    """
    Cleans text by removing all punctuation and special characters. "Your string!" -> "your string"
    :param inp: Input text as string
    :return: Cleaned text as string
    """
    inp = inp.translate(str.maketrans(string.punctuation, " "*len(string.punctuation)))
    inp = re.sub(r'\s+', ' ', inp.lower())
    return inp


def train(data: str) -> dict:
    """
    Function ss used for assignment compatibility
    return: w2v_dict: dict
            - key: string (word)
            - value: np.array (embedding)
    """
    w2v = Word2Vec(txt=data)
    return w2v.word_vectors


class Word2Vec:
    def __init__(self, txt: str, window_size: int = 5, vector_size: int = 100, epochs: int = 1):
        self.txt = txt.split(' ')
        self.words = list(set(self.txt))  # list of all unique words provided
        self.words_num = len(self.words)
        self.window_size = window_size
        self.vector_size = vector_size
        self.epochs = epochs

        # Word vectors
        self.word_vectors = {}
        # Word vectors as dict{'word': numpy.array()} are initialized as one-hot vectors
        for word in self.words:
            self.word_vectors[word] = self.word2onehot(word)

        # Co-occurrence matrix initialized with zeroes
        self.occ_mtrx = np.zeros(shape=(self.words_num, self.words_num))

        # Embedding matrix
        self.embd_mtrx = np.random.uniform(low=-0.8, high=0.8, size=(self.words_num, self.vector_size))

        # Context matrix
        self.cntx_mtrx = np.random.uniform(low=-0.8, high=0.8, size=(self.vector_size, self.words_num))

    @staticmethod
    def softmax(x) -> np.ndarray:
        """
        Numerically stable softmax function which works better in case of large numbers
        :param x: Vector in array-like form: list, set or numpy.ndarray
        :return: Normilized values as numpy.ndarray
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def word2onehot(self, word) -> numpy.ndarray:
        """
        Builds one-hot vector for word: [0, 1, 0, 0, ... 0]
        :param word: Word from self.words
        :return: One-hot vector for the given word with '1' at index equal to word place in self.words
        """
        word_vec = np.zeros(self.words_num)
        word_vec[self.words.index(word)] = 1
        return word_vec

    def create_co_occurrence_matrix(self, w_size: int = 1) -> None:
        """
        Creates co-occurence matrix
        :param w_size: Window size
        :return: None
        """
        for focus_word_indx, focus_word in enumerate(self.words):
            for i in range(self.words_num - w_size + 1):
                window = self.txt[i: i + w_size + 1]
                if focus_word in window:
                    window.remove(focus_word)
                    for word in window:
                        cntx_word_idx = self.words.index(word)
                        self.occ_mtrx[focus_word_indx][cntx_word_idx] += 1

    def wv(self, word):
        """
        Returns vector representation for a given word
        :param word: Word to return a related vector
        :return: Word vector as numpy.array
        """
        return self.word_vectors[word]

    def word_indx(self, word) -> int:
        """
        Returns word index in array of unique words
        :param word: Word to return a related index
        :return: Index as int
        """
        return self.words.index(word)

    def word_cooccurrence_vector(self, word) -> numpy.ndarray:
        """
        Returns word index in array of unique words
        :param word: Word to return a related index
        :return: Index as int
        """
        idx = self.word_indx(word)
        return self.occ_mtrx[idx]

    def train(self, training_data):
        """
        Trains model
        :param training_data:
        :return:
        """
        pass


if __name__ == '__main__':
    with open('input.txt', 'r') as f:
        text = f.read()

    clean_text = clean(text)

    w2v = Word2Vec(clean_text)
    w2v.create_co_occurrence_matrix()
    print(w2v.occ_mtrx)
    print(w2v.word_vectors)
    print(w2v.word_cooccurrence_vector('the'))
