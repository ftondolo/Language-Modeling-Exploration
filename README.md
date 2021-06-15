# Language-Modeling-Exploration
After training NGRAM models on Shakespeare, how up with the times can they be?

## Prerequisites
This ngram model reguires python's Natural Language Toolkit (NLTK)for its modeling. Not only does this library provide numerous contemporary corpora against which to test the models but it also serves to organize singular word lists into sorted n-grams and runs frequency distribution calculations to derive associateed probabilities for every available vocabulary combination.

### Install 
To install this library one can use pip on UNIX systems:
```pip install --user -U nltk```

Or the [Binary](http://pypi.python.org/pypi/nltk) for Windows... :/

## Objective
As a final project for LING 28610 Computational Linguistics, I (Federico Tondolo) and fellow students Masaki Makitani and Krisya Louie set out to ask the question: If natural language models were trained on an antiquated corpus could the statistical ties and undercurrents established during training translate to contemporary corpora? That is, if a LM were trained on Shakespeare could it perform *acceptably* on modern language texts?  In this repository are my contributions to the project consisting of a series of n-gram models including bigrams, trigrams, 4-grams, and 5-grams and analysis of their performance.

Now, it is evident from the get go that in such a scenario a LM would substantially underperform. Not only is Shakespeare particularly well known for his tortuous writing, the bane of middle school students everywhere, which is quite different from today's writing and should therefore not allow for predictive rules to easily translate, but also, by choosing a playwright, in addition to providing LMs with common parlance one is also feeding them character names and very specific plot devices which are most likely unique to the training set.  These obstacles are unfortunately rather unavoidable per the nature of LMs and the massive training sets they require to establish accurate statistical representations of the predictive patterns in the data. How do the two relate? Shakespeare is quite simply one of the largest reliable corpus of text from the period providing an ample yet verisimile window into the language of Elizabethan Eengland.  <<But Shakespeare's vocabulary was not represeentative of the time!>> I hear you cry, and indeed, Shakespeare had the infuriating literary habit of inventing words when none wuit fit his fancy: hee used approximately 20,000 unique words in his works and circa 1,700 were his own creation. However, these 1,700 words exist and are in use to this day, one of th few perks of being considerd perhaps the greatest writer of all time, and as such his plays work to bridge the linguistic gap between then and now. Any older and ttextual evidence grows more sparse and unreadable, any newer and the vestiges of Old English fade away allowing for more representative training sets and higher performance but less insightful and interesting results. As such, preprared for the performance hit to come, we brave forwards for the sake of...what exactly?

The motivtion bhind this endeavour was to inestigate the linguistic evolution of the English language through the eyes of a language model. Could a predictive algorithm follow the changes which underwent the English language in the span of 350 years or would the interim have proven too vast for such models to handle? This is the question we set out to answer.

## Plan of Attack
While we could train our models on some of Shakespeare's plays, run them on a modern corpus of some sort and call it a day what we instead decide to do in respect of scientific integrity was the following: 
(1) Train separate 2, 3, 4, & 5-grams on Shakespeare and two different contemprary corpora,
(2) Record the predictive accuracy of the models on the validation sets of their own corpus,
(3) Record the predictive accuracy of the Shakespearee trained models on both modern corpora both
* On the entirety of the modern corpora
* Only in the predictive instances were the correct target word was amoong those considered by the LM (explained later)
(4) Record the predictive accuracy of one of the contemporary-trained LMs on the other modern corpus for juxtaposition with the classically trained model

### The Data
Given this plan, this inquiry we required 3 different datasets:
1. A Shakespeare training corpus, as extensive as possible to train our datasets on
* MIT's freely accessible hosting of [The Complete Works of William Shakespeare](http://shakespeare.mit.edu/) was a life-saver as Krisya was able to scrape the raw text data into a single txt file which was then used for the LMs
2. Two corpora consisting of modern vernacular
* For the sake of simplicity and repeatability the Brown University Standard Corpus of Present-Day American English (aka. Brown) and Reuters corpora included in NLTK were used. Not only are these corpora established as both extensive and well-balanced but they allowed for the easy filtering out of numbers or similarly confounding inputs. 
