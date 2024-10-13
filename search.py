import numpy as np
import pandas as pd
import spacy
import string
import gensim
import operator
import re

from spacy.lang.en.stop_words import STOP_WORDS
from gensim import corpora
from gensim.similarities import MatrixSimilarity
from operator import itemgetter

class Search:
    def __init__(self):
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self.punctuations = string.punctuation
        self.tokenized_list = list()
        self.corpus = None
        self.dicitonary = None
        self.tf_idf_model = None
        self.lsi_model = None
        self.index = None

    def tokenizer(self, sentence):
        if not isinstance(sentence, str):  
            return []  

        spacy_nlp = spacy.load('en_core_web_sm')

        sentence = re.sub('\'', '', sentence)
        sentence = re.sub('\w*\d\w*', '', sentence)
        sentence = re.sub(' +', ' ', sentence)
        sentence = re.sub(r'\n: \'\'.*', '', sentence)
        sentence = re.sub(r'\n!.*', '', sentence)
        sentence = re.sub(r'^:\'\'.*', '', sentence)
        sentence = re.sub(r'\n', ' ', sentence)
        sentence = re.sub(r'[^\w\s]', ' ', sentence)

        tokens = spacy_nlp(sentence)

        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
        tokens = [word for word in tokens if word not in self.stop_words and word not in self.punctuations and len(word) > 2]

        return tokens

    def preprocessing(self, description_list):
        self.tokenized_list = description_list.map(lambda x: self.tokenizer(x))

    def corpus_definition(self):
        self.dictionary = corpora.Dictionary(self.tokenized_list)
        #dictionary.filter_extremes(no_below=4, no_above=0.2)

        stoplist = set('hello and if this can would should could tell ask stop come go')
        stop_ids = [self.dictionary.token2id[stopword] for stopword in stoplist if stopword in self.dictionary.token2id]
        self.dictionary.filter_tokens(stop_ids)

        self.corpus = [self.dictionary.doc2bow(desc) for desc in self.tokenized_list]

    def similarity_search(self, search_term):
        query_bow = self.dictionary.doc2bow(self.tokenizer(search_term))
        query_tfidf = self.tfidf_model[query_bow]
        query_lsi = self.lsi_model[query_tfidf]

        self.index.num_best = 5

        document_list = self.index[query_lsi]

        document_list.sort(key=itemgetter(1), reverse=True)
        result = []

        for j, item in enumerate(document_list):
            result.append (
                {
                    'Relevance': round((item[1] * 100),2),
                    'Value' : item[0]
                }

            )
            if j == (self.index.num_best-1):
                break

        return pd.DataFrame(result, columns=['Relevance','Value'])
    
    def model_init(self):
        self.tfidf_model = gensim.models.TfidfModel(self.corpus, id2word=self.dictionary)
        self.lsi_model = gensim.models.LsiModel(self.tfidf_model[self.corpus], id2word=self.dictionary, num_topics=300)

        gensim.corpora.MmCorpus.serialize('tfidf_model_mm', self.tfidf_model[self.corpus])
        gensim.corpora.MmCorpus.serialize('lsi_model_mm', self.lsi_model[self.tfidf_model[self.corpus]])
    
    def load_model(self):
        self.tfidf_corpus = gensim.corpora.MmCorpus('tfidf_model_mm')
        self.lsi_corpus = gensim.corpora.MmCorpus('lsi_model_mm')    
        self.index = MatrixSimilarity(self.lsi_corpus, num_features = self.lsi_corpus.num_terms)
    
    def run(self, description_list, query):
        self.preprocessing(description_list)
        self.corpus_definition()
        self.model_init()
        self.load_model()

        return self.similarity_search(query)
        