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
shake_trigrams = nltk.trigrams(shake_unigrams)

brown_bigrams = nltk.bigrams(brown.words())
brown_trigrams = nltk.trigrams(brown.words())

# Trandsorming into context-target pairs
shake_bigram_pairs = (((w0), w1) for w0, w1 in shake_bigrams)
shake_trigram_triplets = (((w0, w1), w2) for w0, w1, w2 in shake_trigrams)

brown_bigram_pairs = (((w0), w1) for w0, w1 in brown_bigrams)
brown_trigram_triplets = (((w0, w1), w2) for w0, w1, w2 in brown_trigrams)

# Calculating word frequency and consquently probability distributions
shake_bi_cfd = nltk.ConditionalFreqDist(shake_bigram_pairs)
shake_bi_cpd = nltk.ConditionalProbDist(shake_bi_cfd, nltk.MLEProbDist)

brown_bi_cfd = nltk.ConditionalFreqDist(brown_bigram_pairs)
brown_bi_cpd = nltk.ConditionalProbDist(brown_bi_cfd, nltk.MLEProbDist)

shake_tri_cfd = nltk.ConditionalFreqDist(shake_trigram_triplets)
shake_tri_cpd = nltk.ConditionalProbDist(shake_tri_cfd, nltk.MLEProbDist)

brown_tri_cfd = nltk.ConditionalFreqDist(brown_trigram_triplets)
brown_tri_cpd = nltk.ConditionalProbDist(brown_tri_cfd, nltk.MLEProbDist)

# Sample functions
print(shake_tri_cpd[('I')].samples())
print(brown_tri_cpd[('I', 'am')].samples())
print(shake_tri_cpd[('was')].prob('and'))
print(brown_tri_cpd[('this', 'was')].prob('and'))
