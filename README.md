These are implementations of ML models starting from bigrams all the way till transformers along with Andrej Karpathy's youtube series.

There are two major differences:
1. In videos word prediction is done, but here we try to predict the entire sentence
2. The training and predictions are done for marathi language instead of english

For the task this is the dataset used: https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/mr.txt

Notebooks and their essence:
1. bigram.ipynb => Simple bigram model but at word level. In the video next character is predicted with one character as input, here we predict next word with one word as input. Note that the vocabulary here is words, so much larger than 27 character vocabulary in videos, so produces terrible results as expected.
2. simple_bigram_network.ipynb => NN implementation of the bag_of_words.ipyng. Note that vocabulary here is still words instead of characters. Because of this the vocab size is very very large, 50k+, which is even more than gpt4 :p. Note that this also predicts the entire sentence, one word at a time. As expected, gives terrible results.
3. one_char_context_network.ipynb => Same as simple_bigram_network.ipynb, but remedies using the word as vocab instead problem. This uses character as a vocab. So vocabulary now is manageable ~400. As the theme of this repo, this also tries to predict the sentences.
4. k_char_context_network.ipynb => Same as one_char_context_network.ipynb, but main difference that it uses k character context as input instead of just last character