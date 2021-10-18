import numpy


def preprocess(text):

    text = text.lower()
    text = text.replace(".", " .")

    words = text.split(" ")

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = [word_to_id[word] for word in words]
    corpus = numpy.array(corpus)

    return corpus, word_to_id, id_to_word

def corpus_to_matrix(corpus, vocab_size, window_size=1):

    corpus_size = len(corpus)
    co_matrix = numpy.zeros((vocab_size, vocab_size), dtype=numpy.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - 1
            right_idx = idx + 1
        
            if left_idx >= 0:
                co_matrix[word_id, corpus[left_idx]] += 1

            if right_idx < corpus_size:
                co_matrix[word_id, corpus[right_idx]] += 1

    return co_matrix 

def cosine_similarity(x, y, eps=1e-8):
    nx = x / (numpy.sqrt(numpy.sum(x**2)) + eps)
    ny = y / (numpy.sqrt(numpy.sum(y**2)) + eps)
    return numpy.dot(nx, ny)

text = "You say goodbye and I say hello."

corpus, word_to_id, id_to_word = preprocess(text)
co_matrix = corpus_to_matrix(corpus, len(word_to_id), 1)

print(co_matrix)
print(cosine_similarity(co_matrix[word_to_id["you"]], co_matrix[word_to_id["i"]]))

