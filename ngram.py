import nltk, string
from nltk import word_tokenize, bigrams, trigrams
from nltk.corpus import brown, reuters
from nltk.util import ngrams
from collections import Counter

unk_frequency_boundary = 10

# Corpus
with open('/Volumes/LOG/LMs/corpus.txt', 'r') as file:
    corpus = file.read().replace('\n', ' ')

# Removing punctuation from corpus and transforming into unigram, bigram, and trigram lists
tokenizer = nltk.RegexpTokenizer(r"\w+")
shake_unigrams = list(map(lambda x:x.lower(),tokenizer.tokenize(corpus)))
shake_train = shake_unigrams[:441568]
shake_test = shake_unigrams[441568:]
shake_bigrams = nltk.bigrams(shake_train)
shake_trigrams = nltk.trigrams(shake_train)
shake_fourgrams = ngrams(shake_train, 4)
shake_fivegrams = ngrams(shake_train, 5)

###############################################
brown_corpus = []
for each in brown.words():
    if each not in string.punctuation:
        chars = set('0123456789$.,"')
        if any((c not in chars) for c in each):
            brown_corpus.append(each.lower())

brown_train = brown_corpus[:821498]
brown_test = brown_corpus[821498:]
brown_bigrams = nltk.bigrams(brown_train)
brown_trigrams = nltk.trigrams(brown_train)
brown_fourgrams = ngrams(brown_train, 4)
brown_fivegrams = ngrams(brown_train, 5)

###############################################
reuter_corpus = []
for each in reuters.words():
    if each not in string.punctuation:
        chars = set('0123456789$.,"')
        if any((c not in chars) for c in each):
            reuter_corpus.append(each.lower())

reuter_bigrams = nltk.bigrams(reuter_corpus)
reuter_trigrams = nltk.trigrams(reuter_corpus)
reuter_fourgrams = ngrams(reuter_corpus, 4)
reuter_fivegrams = ngrams(reuter_corpus, 5)

# Trandsorming into context-target pairs
shake_bigram_pairs = (((w0), w1) for w0, w1 in shake_bigrams)
shake_trigram_triplets = (((w0, w1), w2) for w0, w1, w2 in shake_trigrams)
shake_fourgram_quadruplets = (((w0, w1, w2), w3) for w0, w1, w2, w3 in shake_fourgrams)
shake_fivegram_quintuplets = (((w0, w1, w2, w3), w4) for w0, w1, w2, w3, w4 in shake_fivegrams)

###############################################
brown_bigram_pairs = (((w0), w1) for w0, w1 in brown_bigrams)
brown_trigram_triplets = (((w0, w1), w2) for w0, w1, w2 in brown_trigrams)
brown_fourgram_quadruplets = (((w0, w1, w2), w3) for w0, w1, w2, w3 in brown_fourgrams)
brown_fivegram_quintuplets = (((w0, w1, w2, w3), w4) for w0, w1, w2, w3, w4 in brown_fivegrams)

#Creatiing data structures for comparative testing
sbigrams = list((((w0), w1) for w0, w1 in nltk.bigrams(shake_test)))
strigrams = list((((w0, w1), w2) for w0, w1, w2 in nltk.trigrams(shake_test)))
squadruplets = list((((w0, w1, w2), w3) for w0, w1, w2, w3 in ngrams(shake_test, 4)))
squintuplets = list(((w0, w1, w2, w3), w4) for w0, w1, w2, w3, w4 in ngrams(shake_test, 5))

###############################################
bbigrams = list((((w0), w1) for w0, w1 in nltk.bigrams(brown_test)))
btrigrams = list((((w0, w1), w2) for w0, w1, w2 in nltk.trigrams(brown_test)))
bquadruplets = list((((w0, w1, w2), w3) for w0, w1, w2, w3 in ngrams(brown_test, 4)))
bquintuplets = list(((w0, w1, w2, w3), w4) for w0, w1, w2, w3, w4 in ngrams(brown_test, 5))

###############################################
rbigrams = list((((w0), w1) for w0, w1 in nltk.bigrams(reuter_corpus)))
rtrigrams = list((((w0, w1), w2) for w0, w1, w2 in nltk.trigrams(reuter_corpus)))
rquadruplets = list((((w0, w1, w2), w3) for w0, w1, w2, w3 in ngrams(reuter_corpus, 4)))
rquintuplets = list(((w0, w1, w2, w3), w4) for w0, w1, w2, w3, w4 in ngrams(reuter_corpus, 5))

# Calculating word frequency and consquently probability distributions
shake_bi_cfd = nltk.ConditionalFreqDist(shake_bigram_pairs)
shake_bi_cpd = nltk.ConditionalProbDist(shake_bi_cfd, nltk.MLEProbDist)

shake_tri_cfd = nltk.ConditionalFreqDist(shake_trigram_triplets)
shake_tri_cpd = nltk.ConditionalProbDist(shake_tri_cfd, nltk.MLEProbDist)

shake_quad_cfd = nltk.ConditionalFreqDist(shake_fourgram_quadruplets)
shake_quad_cpd = nltk.ConditionalProbDist(shake_quad_cfd, nltk.MLEProbDist)

shake_quin_cfd = nltk.ConditionalFreqDist(shake_fivegram_quintuplets)
shake_quin_cpd = nltk.ConditionalProbDist(shake_quin_cfd, nltk.MLEProbDist)

###############################################
brown_bi_cfd = nltk.ConditionalFreqDist(brown_bigram_pairs)
brown_bi_cpd = nltk.ConditionalProbDist(brown_bi_cfd, nltk.MLEProbDist)

brown_tri_cfd = nltk.ConditionalFreqDist(brown_trigram_triplets)
brown_tri_cpd = nltk.ConditionalProbDist(brown_tri_cfd, nltk.MLEProbDist)

brown_quad_cfd = nltk.ConditionalFreqDist(brown_fourgram_quadruplets)
brown_quad_cpd = nltk.ConditionalProbDist(brown_quad_cfd, nltk.MLEProbDist)

brown_quin_cfd = nltk.ConditionalFreqDist(brown_fivegram_quintuplets)
brown_quin_cpd = nltk.ConditionalProbDist(brown_quin_cfd, nltk.MLEProbDist)

# Calculating results and printing analysis data
prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in sbigrams:
    counter1 += 1.0
    prob1 += shake_bi_cpd[(each[0])].prob(each[1])
    if each[1] in shake_bi_cpd[(each[0])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += shake_bi_cpd[(each[0])].prob(each[1])
print("###############################################")
print("")
print("Bigram probability v. Own (Shakespeare): "+str(prob1*100/counter1))
print("% target in predictive pool: NA")#+str(insamples*100/counter1))
print("Bigram probability in previously observed pairs: "+str(prob2*100/counter2))

prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in rbigrams:
    counter1 += 1.0
    prob1 += shake_bi_cpd[(each[0])].prob(each[1])
    if each[1] in shake_bi_cpd[(each[0])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += shake_bi_cpd[(each[0])].prob(each[1])
print("")
print("Bigram probability v. Reuters (Shakespeare): "+str(prob1*100/counter1))
print("% target in predictive pool: "+str(insamples*100/counter1))
print("Bigram probability in previously observed pairs: "+str(prob2*100/counter2))

prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in bbigrams:
    counter1 += 1.0
    prob1 += brown_bi_cpd[(each[0])].prob(each[1])
    if each[1] in brown_bi_cpd[(each[0])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += brown_bi_cpd[(each[0])].prob(each[1])
print("")
print("Bigram probability v. Own (Brown): "+str(prob1*100/counter1))
print("% target in predictive pool: NA")#+str(insamples*100/counter1))
print("Bigram probability in previously observed pairs: "+str(prob2*100/counter2))

prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in rbigrams:
    counter1 += 1.0
    prob1 += brown_bi_cpd[(each[0])].prob(each[1])
    if each[1] in brown_bi_cpd[(each[0])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += brown_bi_cpd[(each[0])].prob(each[1])
print("")
print("Bigram probability v. Reuters (Brown): "+str(prob1*100/counter1))
print("% target in predictive pool: "+str(insamples*100/counter1))
print("Bigram probability in previously observed pairs: "+str(prob2*100/counter2))
print("")
print("###############################################")

###############################################
prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in strigrams:
    counter1 += 1.0
    prob1 += shake_tri_cpd[(each[0][0], each[0][1])].prob(each[1])
    if each[1] in shake_tri_cpd[(each[0][0], each[0][1])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += shake_tri_cpd[(each[0][0], each[0][1])].prob(each[1])
print("")
print("Trigram probability v. Own (Shakespeare): "+str(prob1*100/counter1))
print("% target in predictive pool: NA")#+str(insamples*100/counter1))
print("Trigram probability in previously observed triplets: "+str(prob2*100/counter2))
print("")

prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in rtrigrams:
    counter1 += 1.0
    prob1 += shake_tri_cpd[(each[0][0], each[0][1])].prob(each[1])
    if each[1] in shake_tri_cpd[(each[0][0], each[0][1])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += shake_tri_cpd[(each[0][0], each[0][1])].prob(each[1])
print("Trigram probability v. Reuters (Shakespeare): "+str(prob1*100/counter1))
print("% target in predictive pool: "+str(insamples*100/counter1))
print("Trigram probability in previously observed triplets: "+str(prob2*100/counter2))
print("")

prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in btrigrams:
    counter1 += 1.0
    prob1 += brown_tri_cpd[(each[0][0], each[0][1])].prob(each[1])
    if each[1] in brown_tri_cpd[(each[0][0], each[0][1])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += brown_tri_cpd[(each[0][0], each[0][1])].prob(each[1])
print("Trigram probability v. Own (Brown): "+str(prob1*100/counter1))
print("% target in predictive pool: NA")#+str(insamples*100/counter1))
print("Trigram probability in previously observed triplets: "+str(prob2*100/counter2))
print("")

prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in rtrigrams:
    counter1 += 1.0
    prob1 += brown_tri_cpd[(each[0][0], each[0][1])].prob(each[1])
    if each[1] in brown_tri_cpd[(each[0][0], each[0][1])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += brown_tri_cpd[(each[0][0], each[0][1])].prob(each[1])
print("Trigram probability v. Reuters (Brown): "+str(prob1*100/counter1))
print("% target in predictive pool: "+str(insamples*100/counter1))
print("Trigram probability in previously observed triplets: "+str(prob2*100/counter2))
print("")
print("###############################################")

###############################################
prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in squadruplets:
    counter1 += 1.0
    prob1 += shake_quad_cpd[(each[0][0], each[0][1], each[0][2])].prob(each[1])
    if each[1] in shake_quad_cpd[(each[0][0], each[0][1], each[0][2])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += shake_quad_cpd[(each[0][0], each[0][1], each[0][2])].prob(each[1])
print("")
print("Fourgram probability v. Own (Shakespeare): "+str(prob1*100/counter1))
print("% target in predictive pool: NA")#+str(insamples*100/counter1))
print("Fourgram probability in previously observed quadruplets: "+str(prob2*100/counter2))
print("")

prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in rquadruplets:
    counter1 += 1.0
    prob1 += shake_quad_cpd[(each[0][0], each[0][1], each[0][2])].prob(each[1])
    if each[1] in shake_quad_cpd[(each[0][0], each[0][1], each[0][2])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += shake_quad_cpd[(each[0][0], each[0][1], each[0][2])].prob(each[1])
print("Fourgram probability v. Reuters (Shakespeare): "+str(prob1*100/counter1))
print("% target in predictive pool: "+str(insamples*100/counter1))
print("Fourgram probability in previously observed quadruplets: "+str(prob2*100/counter2))
print("")

prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in bquadruplets:
    counter1 += 1.0
    prob1 += brown_quad_cpd[(each[0][0], each[0][1], each[0][2])].prob(each[1])
    if each[1] in brown_quad_cpd[(each[0][0], each[0][1], each[0][2])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += brown_quad_cpd[(each[0][0], each[0][1], each[0][2])].prob(each[1])
print("Fourgram probability v. Own (Brown): "+str(prob1*100/counter1))
print("% target in predictive pool: NA")#+str(insamples*100/counter1))
print("Fourgram probability in previously observed quadruplets: "+str(prob2*100/counter2))
print("")

prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in rquadruplets:
    counter1 += 1.0
    prob1 += brown_quad_cpd[(each[0][0], each[0][1], each[0][2])].prob(each[1])
    if each[1] in brown_quad_cpd[(each[0][0], each[0][1], each[0][2])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += brown_quad_cpd[(each[0][0], each[0][1], each[0][2])].prob(each[1])
print("Fourgram probability v. Reuters (Brown): "+str(prob1*100/counter1))
print("% target in predictive pool: "+str(insamples*100/counter1))
print("Fourgram probability in previously observed quadruplets: "+str(prob2*100/counter2))
print("")
print("###############################################")

###############################################
prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in squintuplets:
    counter1 += 1.0
    prob1 += shake_quin_cpd[(each[0][0], each[0][1], each[0][2], each[0][3])].prob(each[1])
    if each[1] in shake_quin_cpd[(each[0][0], each[0][1], each[0][2], each[0][3])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += shake_quin_cpd[(each[0][0], each[0][1], each[0][2], each[0][3])].prob(each[1])
print("")
print("Fivegram probability v. Own (Shakespeare): "+str(prob1*100/counter1))
print("% target in predictive pool: NA")#+str(insamples*100/counter1))
print("Fivegram probability in previously observed quintuplets: "+str(prob2*100/counter2))
print("")

prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in rquintuplets:
    counter1 += 1.0
    prob1 += shake_quin_cpd[(each[0][0], each[0][1], each[0][2], each[0][3])].prob(each[1])
    if each[1] in shake_quin_cpd[(each[0][0], each[0][1], each[0][2], each[0][3])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += shake_quin_cpd[(each[0][0], each[0][1], each[0][2], each[0][3])].prob(each[1])
print("Fivegram probability v. Reuters (Shakespeare): "+str(prob1*100/counter1))
print("% target in predictive pool: "+str(insamples*100/counter1))
print("Fivegram probability in previously observed quintuplets: "+str(prob2*100/counter2))
print("")

prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in bquintuplets:
    counter1 += 1.0
    prob1 += brown_quin_cpd[(each[0][0], each[0][1], each[0][2], each[0][3])].prob(each[1])
    if each[1] in brown_quin_cpd[(each[0][0], each[0][1], each[0][2], each[0][3])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += brown_quin_cpd[(each[0][0], each[0][1], each[0][2], each[0][3])].prob(each[1])
print("Fivegram probability v. Own (Brown): "+str(prob1*100/counter1))
print("% target in predictive pool: NA")#+str(insamples*100/counter1))
print("Fivegram probability in previously observed quintuplets: "+str(prob2*100/counter2))
print("")

prob1 = prob2 = insamples = counter1 = counter2 = 0.0
for each in rquintuplets:
    counter1 += 1.0
    prob1 += brown_quin_cpd[(each[0][0], each[0][1], each[0][2], each[0][3])].prob(each[1])
    if each[1] in brown_quin_cpd[(each[0][0], each[0][1], each[0][2], each[0][3])].samples():
        insamples += 1.0
        counter2 += 1.0
        prob2 += brown_quin_cpd[(each[0][0], each[0][1], each[0][2], each[0][3])].prob(each[1])
print("Fivegram probability v. Reuters (Brown): "+str(prob1*100/counter1))
print("% target in predictive pool: "+str(insamples*100/counter1))
print("Fivegram probability in previously observed quintuplets: "+str(prob2*100/counter2))
print("")
