
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors,keyedvectors
import numpy as np
from .util import *
import re
from annoy import AnnoyIndex
from collections import defaultdict
import unidecode

DEFAULT_REGEX = "[-,.\\/!@#$%^&*))_+=]"

def tokenize(text, regex=DEFAULT_REGEX):
    if text is None or (not isinstance(text,str) and np.isnan(text)):
        return []
    return [unidecode(x) for y in word_tokenize(str(text)) for x in re.split(regex, y)]

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def lemmatize(word):
    return  lemmatizer.lemmatize(str(word))

def stem(word):
    return  lemmatizer.lemmatize(str(word))

def lemmatize_data(obj):
    return apply2values(obj, lemmatize)

def stem_data(obj):
    return apply2values(obj, stem)

def reduce_w3v_size(data, model_file, tokenization_regex = DEFAULT_REGEX, use_lemmatizer = True, user_stemmer = False):
    words_model = None
    data = set(flattern_values(data, lambda x: tokenize(x, tokenization_regex)))
    
    with open(model_file, 'rb') as fin:
        words_model = KeyedVectors.load_word2vec_format(fin, binary=True)
    def get_vec(word):
        if word in words_model:
            return words_model.word_vec(word) 
        if use_lemmatizer:
            word = lemmatize(word) 
            if word in words_model:
                return words_model.word_vec(word) 
        if user_stemmer:
            word = stem(word)
            if word in words_model:
                return words_model.word_vec(word) 
        return words_model.word_vec("unk")

    vecs = {w: get_vec(w) for w in data}
    m = keyedvectors.Word2VecKeyedVectors(vector_size=words_model.vector_size)
    m.add(list(vecs.keys()), list(vecs.values()))
    return m

class AnnoySearch:
    def __init__(self, w2v_model, columns, use_lemmatizer = True, use_stemmer= False):
        self.words_model = w2v_model
        self.use_lemmatizer = use_lemmatizer
        self.use_stemmer = use_stemmer
        self.index = {c:AnnoyIndex(w2v_model.vector_size, 'dot') for c in columns}
        
    def get_vector(self, text):
        def get_vec(word):
            if word in self.words_model:
                return self.words_model.word_vec(word) 
            if self.use_lemmatizer:
                word = lemmatize(word) 
                if word in self.words_model:
                    return self.ords_model.word_vec(word) 
            if self.user_stemmer:
                word = stem(word)
                if word in self.words_model:
                    return self.words_model.word_vec(word) 
            return self.words_model.word_vec("unk")
        
        text = tokenize(text)
        if len(text) == 0:
            text = ["unk"]
        vector = np.mean([get_vec(w) for w in text] ,axis=0)
        return vector
    
    def build(self, df, n_trees=1000):
        self.ids_index = df.index.to_list()
        self.ids = dict(zip(self.ids_index, range(len(df))))
        for c, idx in self.index.items():
            for i, row in df.iterrows():
                idx.add_item(self.ids[i], self.get_vector(row[c]))
            idx.build(n_trees)
        
    def save(self):
        write_object("annoy_search.gz", self)
        
    def load(self):
        newObj =  read_object("annoy_search.gz")
        self.__dict__.update(newObj.__dict__)
        
    def query(self, q, n_items, w = None, and_weight= 0.7):
        q_vec = self.get_vector(q)
        res_sum = dict()
        res_min = dict()
        count_dict = defaultdict(int)
        for c, idx in self.index.items():
            ids =  idx.get_nns_by_vector(q_vec, n_items, include_distances=True)
            for i, s in list(zip(list(ids[0]),list(ids[1]))):
                s = s if w is None or c not in w else s * w[c]
                if i in res_sum:
                    res_sum[i] += s
                    res_min[i] = max(res_min[i], s)
                else:
                    res_sum[i] = s
                    res_min[i] = s
                count_dict[i] += 1
        res = dict()
        for i in res_sum.keys():
            res_sum[i] /= count_dict[i]
            res[i] =  res_sum[i]*(1- and_weight) + res_min[i]*(and_weight)
        res = sorted(res.keys(), key= lambda x: res[x])
        res = [self.ids_index[i] for i in res]
        return res

        

        
