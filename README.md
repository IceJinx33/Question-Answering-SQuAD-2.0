# Question-Answering-SQuAD-2.0

This repository contains a Python notebook where 2 models are trained and evaluated on the SQuAD 2.0 Question Answering Task using Google Colab. For both models, the question answering task is treated as a span extraction task, with the models predicting the starting and ending positions of the answer in the document. If the starting position of the answer is greater than the ending position, the question has no answer.

Models:
1. A simplification of the Document Reader model detailed in <a href="https://arxiv.org/abs/1704.00051">Reading Wikipedia to Answer Open-Domain Questions</a> by Chen et. al., 2017. 
   - Paragraph Encoding 
    - Hidden representation for each token by passing the following features through a Bi-LSTM
    - (Features for Each Paragraph Token)
      - GloVe embeddings - glove-wiki-gigaword-50 downloaded using the gensim.downloader package.
      - Part of Speech tags from nltk's pos_tag (averaged perceptron tagger).
      - Binary indicator Case-insensitive exact match - 1 if token in tokenized question, 0 if not.
   - Aligned Question Embedding (implemented as in paper)
   - Question Encoding
      - Mean of the hidden representation of the question tokens from a Bi-LSTM.
2. DistilBERT (a deep contextualized pre-trained language model) with 2 linear layers on top - one for predicting the starting position of the answer, and another predicting the ending position of the answer in the document. 

