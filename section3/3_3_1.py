import numpy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

import common
import common.util


def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)

    return numpy.array(contexts), numpy.array(target)


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = common.util.preprocess(text)

print(create_contexts_target(corpus))