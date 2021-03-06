# Language-Modeling Exploration
After training NGRAM models on Shakespeare, how up with the times can they be?

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#prerequisites">Prerequisites</a>
      <ul>
        <li><a href="#install">Install</a></li>
      </ul>
    </li>
    <li><a href="#objective">Objective</a></li>
    <li>
      <a href="#plan-of-attack">Plan of Attack</a>
      <ul>
        <li><a href="#the-data">The Data</a></li>
      </ul>
    </li>
    <li>
      <a href="#results">Results</a>
      <ul>
        <li><a href="#preface">Preface</a></li>
      </ul>
      <ul>
        <li><a href="#trends">Trends</a></li>
      </ul>
    </li>
    <li><a href="#moving-forwards">Moving Forwards</a></li>
  </ol>
</details>

## Prerequisites
This ngram model reguires python's Natural Language Toolkit (NLTK)for its modeling. Not only does this library provide numerous contemporary corpora against which to test the models but it also serves to organize singular word lists into sorted n-grams and runs frequency distribution calculations to derive associateed probabilities for every available vocabulary combination.

### Install 
To install this library one can use pip on UNIX systems:

```pip install --user -U nltk```

Or the [Binary](http://pypi.python.org/pypi/nltk) for Windows... :/

## Objective
As a final project for LING 28610 Computational Linguistics, I (Federico Tondolo) and fellow students Masaki Makitani and Krisya Louie set out to ask the question: If natural language models were trained on an antiquated corpus could the statistical ties and undercurrents established during training translate to contemporary corpora? That is, if a LM were trained on Shakespeare could it perform *acceptably* on modern language texts?  In this repository are my contributions to the project consisting of a series of n-gram models including bigrams, trigrams, 4-grams, and 5-grams and analysis of their performance which I worked on standalone.

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

## Results
### Preface
Please consult the full writeup for an in-depth analysis not only of the n-gram results themselves but also for the results of other models applied to the data by Masaki and the significance of the variances in performance to the underlying approach these models take to extrapolating meaning. An output.txt file is also provided containing solely the predictive percentages of the various n-gram models.

For instance, LMs such as these draw meaning from the relative likelihood of specific words to follow a given sequence. For instance, the bigram ("There", "once") will most commonly return the word "was" as the most likely target word given the relatively likely occurrence of running into the trigram ("There", "once", "was") as opposed to ("There", "once", "sang"). Given this however these models cannot account for in-word variation, that is to say the evolution of spelling or form a single piece of vocabulary, as such the two trigrams ("What", "have", "you") and ("What", "hast", "thou") will be treated as vastly different when in reality they are literally synonymous.  Similarly, as mentioned earlier, much vocabulary will be unique to a corpus making all the word sequences that contain it statistically intranslatable to a different corpus such as character names, outdated vocabulary (i.e. wench, codswallop), or even just something as mundane as outdated usage (ass being used as a synonym for donkey in Midsummer Night's Dream will not translate well to its use as a descriptor of one's posterior in an anatomical text). This is why instead of focusing on the absolute accuracy of the models I chose to focus on the accuracy demonstrateed in instances where the correct word was in the pool of statistical possibilities considered by the model as it allowed to measure the model's accuracy outside of these idiomatic quandaries.

### Trends
What becomes immediately clear when observing the results is the sacrifice of specificity. As the n-grams analyzed more and more antecedent words for context the predictive accuracy of the models increased drastically but solely in the instances were the target word was even considered a possibility, and those decreased exponentially the more context was taken into consideration. As an example, while a bigram-based model trained on Shakespearee which had shwon 2.66% accuracy on its own validatin set only had an absolute accuracy of 1.2% on the Reuters coprus and 5.6% in instances where the word was in the predictive pool, a 5-gram had  0.2% accuracy on its validation set, an absurdly low 0.002% on the entirety of the Reuters corpus, but over 85% accuracy when the target word was among the possibilities. These low predictive values may seem disheartening but indeed the use of the Brown corpus as a sanity check showed similarly low values of 3.4%, 8.14%, 0.17%, and 67.9% respectively for the aforementioned conditions. 

The low predictive values are in and of themselves caused by the combination of the inability of these models to account for variations in vocabulary, the variations in vocabulary inherent to these datasets due to their differing topics (an unfortunate issue which is only compounded by the temporal distance between them), and the relatively small size and narrow thematic scope of the Shakespeare dataset as compared to the Reuters and Brown. Howver, an intereesting trend which arises is that despite these numerous obstacles the Shakespearee 5-gram was more accurate in predicting the target word in the Reuters corpus than its contemproray counterpart (Brown) in situations where the target word had been in consideration (that is, the ngram had been observed at some previous point during training). The full writeup goes in depth into the significance of this but in essence it underlines how though the two LMs capture different aspects of the linguistic simalirties between these corpora. Haaving broad application of only reasonably accurate n-grams (in comparison to the Shakespare-trained model) the Brown-traind n-grams 'learned' many sequences of of interchangeeable pronouns and articles or similarly interchangeabll words which, though lacking in specific meaning (hence their mrely occasional correctness), weree commonly used enough to appear a notable number of times in both modern texts. The Shakespeare-trained modeel on the other hand did not have access to these commonly used 'fillers' if you will, as by the very nature of its training set such linguistic expressions were less common in Shakespeare's time and even the few which did used alternative speling and formulation (has v. hath). Nevertheless, the less common word sequences which it did internalize were far better conserved, resulting in predictive accuracy close to 20% higher! What we see is that though these models rely on the conservation of vocabulary which is evidently lacking they were able to still take note of linguistic expressions which remained highly conserved in the English language despite an interlude f centuries and across immensely diverse topics, from plays to dry academic texts.

## Moving Forwards

* Unfortunately, none of the models in this writeup could account for variations in vocabulary due to their reliance on statistics. It would be fascinating to instead implement word vectors to derive meaning, using the context surrounding words to derive the linguistic purpose of words in text to perhaps be able to tie together synonymous or evolved vocabulary as a unit to provide a more truthful representation of meaning to the models and better extrapolate past such divergences.
* Ideally one would want to find corpora which are more thematically conjoined than those we selected, perhaps the usage of contemporary rimaginings of Shakespeare's plays such as those which are so in style in Broadway these days could work to minimize the difference in subject betweent he corpora, having similarly unique character names and stage directions, though the question arises of whether a large enough corpus could be constructed.
