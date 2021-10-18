import numpy


text = "You say goodbye and I say hello."

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

print(preprocess(text))