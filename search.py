import numpy as np
import pandas as pd
import spacy
import string
import gensim
import operator
import re
import os

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
        self.dictionary = None
        self.tf_idf_model = None
        self.lsi_model = None
        self.index = None

        self.dictionary_path = "dictionary.dict"
        self.corpus_path = "corpus.mm"
        self.tfidf_model_path = "tfidf_model.mm"
        self.lsi_model_path = "lsi_model.mm"
        self.index_path = "index.index"

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
        
        stoplist = set('hello and if this can would should could tell ask stop come go')
        stop_ids = [self.dictionary.token2id[stopword] for stopword in stoplist if stopword in self.dictionary.token2id]
        self.dictionary.filter_tokens(stop_ids)

        self.corpus = [self.dictionary.doc2bow(desc) for desc in self.tokenized_list]

    def model_init(self):
        self.tf_idf_model = gensim.models.TfidfModel(self.corpus, id2word=self.dictionary)
        self.lsi_model = gensim.models.LsiModel(self.tf_idf_model[self.corpus], id2word=self.dictionary, num_topics=300)
        
        self.dictionary.save(self.dictionary_path)
        corpora.MmCorpus.serialize(self.corpus_path, self.corpus)
        self.tf_idf_model.save(self.tfidf_model_path)
        self.lsi_model.save(self.lsi_model_path)

    def load_model(self):
        self.dictionary = corpora.Dictionary.load(self.dictionary_path)
        self.corpus = corpora.MmCorpus(self.corpus_path)
        self.tf_idf_model = gensim.models.TfidfModel.load(self.tfidf_model_path)
        self.lsi_model = gensim.models.LsiModel.load(self.lsi_model_path)
        
        self.index = MatrixSimilarity(self.lsi_model[self.corpus], num_features=self.lsi_model.num_topics)
    
    def similarity_search(self, search_term):
        query_bow = self.dictionary.doc2bow(self.tokenizer(search_term))
        query_tfidf = self.tf_idf_model[query_bow]
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
            if j == (self.index.num_best - 1):
                break

        return pd.DataFrame(result, columns=['Relevance', 'Value'])
    
    def run(self, description_list, query):
        if not (os.path.exists(self.dictionary_path) and os.path.exists(self.corpus_path)
                and os.path.exists(self.tfidf_model_path) and os.path.exists(self.lsi_model_path)):
            self.preprocessing(description_list)
            self.corpus_definition()
            self.model_init()
        
        self.load_model()

        return self.similarity_search(query)