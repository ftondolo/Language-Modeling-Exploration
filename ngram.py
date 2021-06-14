import nltk
from nltk import word_tokenize, bigrams, trigrams
from nltk.corpus import brown

# Corpus - could be later modified to read from file
with open('corpus.txt', 'r') as file:
    corpus = file.read().replace('\n', ' ')

# Removing punctuation from corpus and transforming into unigram, bigram, and trigram lists
tokenizer = nltk.RegexpTokenizer(r"\w+")
shake_unigrams = tokenizer.tokenize(corpus)
shake_bigrams = nltk.bigrams(shake_unigrams)

brown_bigrams = nltk.bigrams(brown.words())

# Trandsorming into context-target pairs
shake_bigram_pairs = (((w0), w1) for w0, w1 in shake_bigrams)

brown_bigram_pairs = (((w0), w1) for w0, w1 in brown_bigrams)

# Calculating word frequency and consquently probability distributions
shake_bi_cfd = nltk.ConditionalFreqDist(shake_bigram_pairs)
shake_bi_cpd = nltk.ConditionalProbDist(shake_bi_cfd, nltk.MLEProbDist)

brown_bi_cfd = nltk.ConditionalFreqDist(brown_bigram_pairs)
brown_bi_cpd = nltk.ConditionalProbDist(brown_bi_cfd, nltk.MLEProbDist)

# Sample functions
print(shake_bi_cpd[('I')].samples())
print(brown_bi_cpd[('I', 'am')].samples())
print(shake_bi_cpd[('was')].prob('and'))
print(brown_bi_cpd[('this', 'was')].prob('and'))
