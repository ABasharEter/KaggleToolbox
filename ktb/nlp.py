
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors,keyedvectors
import numpy as np
from .util import *
import re
from annoy import AnnoyIndex
from collections import defaultdict
import unidecode
from gensim.summarization.bm25 import BM25
from nltk.corpus import stopwords

DEFAULT_REGEX = "[-,.\\/!@#$%^&*))_+=\(|\)|:|,|;|\.|’|”|“|\?|%|>|<]+"

stopwords_en = set(stopwords.words('english'))

def tokenize(text, regex=DEFAULT_REGEX):
    if text is None or (not isinstance(text,str) and np.isnan(text)):
        return []
    return [x for y in word_tokenize(str(text)) for x in re.split(regex, y)]

def analyze(tokesn):
    tokens = [unidecode.unidecode(t).lower() for t in tokens 
        if len(t) > 1 and t not in stopwords_en and (not t.isnumeric() or t.isalpha())]
    return tokens
   

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

def reduce_w3v_size(data, model_file, tokenization_regex = DEFAULT_REGEX, use_lemmatizer = True, use_stemmer = False):
    words_model = None
    data = set(flattern_values(data, lambda x: analyze(x, tokenization_regex)))
    
    with open(model_file, 'rb') as fin:
        words_model = KeyedVectors.load_word2vec_format(fin, binary=True)
    def get_vec(word):
        if word in words_model:
            return words_model.word_vec(word) 
        if use_lemmatizer:
            word = lemmatize(word) 
            if word in words_model:
                return words_model.word_vec(word) 
        if use_stemmer:
            word = stem(word)
            if word in words_model:
                return words_model.word_vec(word) 
        return words_model.word_vec("unk")

    vecs = {w: get_vec(w) for w in data}
    m = keyedvectors.Word2VecKeyedVectors(vector_size=words_model.vector_size)
    m.add(list(vecs.keys()), list(vecs.values()))
    return m

def rank_items(scores, n_indexs, w = None, and_weight= 0.7):
    items = []
    for item, sc  in scores.items():
        sc = [w[k] * v if w is not None and k in w else v for k,v in sc.items()]
        score = min(sc) * (and_weight) + (1-and_weight) * (np.sum(sc) + (n_indexs -len(sc)) * max(sc))/n_indexs
        items.append((item, score))
    items = sorted(items, key = lambda x: x[1])
    return items
        


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
                    return self.words_model.word_vec(word) 
            if self.use_stemmer:
                word = stem(word)
                if word in self.words_model:
                    return self.words_model.word_vec(word) 
            return self.words_model.word_vec("unk")
        
        text = analyze(text)
        if len(text) == 0:
            text = ["unk"]
        vector = np.mean([get_vec(w) for w in text] ,axis=0)
        return vector
    
    def build(self, df, n_trees=1000):
        self.ids_index = df.index.to_list()
        self.ids = dict(zip(self.ids_index, range(len(df))))
        self.df = df
        for c, idx in self.index.items():
            for i, row in df.iterrows():
                idx.add_item(self.ids[i], self.get_vector(row[c]))
            idx.build(n_trees)
        
    def save(self):
        write_object("annoy_search.gz", self)
        
    def load(self):
        newObj =  read_object("annoy_search.gz")
        self.__dict__.update(newObj.__dict__)
    
    def query(self, q, n_items, w = None, and_weight= 0.7, include_distances= False):
        q_vec = self.get_vector(q)
        score_dict = defaultdict(dict)
        for c, idx in self.index.items():
            ids =  idx.get_nns_by_vector(q_vec, n_items, include_distances= True)
            for i, s in list(zip(list(ids[0]),list(ids[1]))):
                score_dict[i].update({c:s})
        sorted_res = rank_items(score_dict, len(self.index), w, and_weight)
        
        if not include_distances:
            sorted_res = [k for k,v in sorted_res]
        return sorted_res

        
class HybridSearch:
    def __init__(self, w2v_model, columns, use_lemmatizer = True, use_stemmer= True, use_lemmatizer_annoy = True, use_stemmer_annoy= False):
        self.use_lemmatizer = use_lemmatizer
        self.use_stemmer = use_stemmer
        self.annoy_search = AnnoySearch(w2v_model, columns, use_lemmatizer_annoy, use_lemmatizer_annoy)
        self.bm25 = {c:None for c in columns}

    def process_text(self, text):
        text = analyze(text)
        if self.use_lemmatizer:
            text = lemmatize_data(text)
        if self.use_stemmer:
            text = stem_data(text)
        return text

    def build(self, df, n_trees=1000):
        for c in self.bm25.keys():
            data = []
            for i, row in df.iterrows():
                data.append(self.process_text(row[c]))
            self.bm25[c] = BM25(data)
        self.annoy_search.build(df, n_trees)
    
    def save(self):
        write_object("hybrid_search.gz", self)
        
    def load(self):
        newObj =  read_object("hybrid_search.gz")
        self.__dict__.update(newObj.__dict__) 
    
    def query(self, q, n_items, w = None, and_weight= 0.7, include_distances=False, additive_semantic_weight = 0.1):
        score_dict = defaultdict(dict)
        q_tokens = self.process_text(q)
        for c, idx in self.bm25.items():
            scores =  idx.get_scores(q_tokens)
            for i, s in zip(range(len(scores)), scores):
                score_dict[i].update({c: -s})
        sorted_res = rank_items(score_dict, len(self.bm25), w, and_weight)
        bm25_dict = dict(sorted_res)
        annoy_res = self.annoy_search.query(q, n_items, w, and_weight, True)
        for item,score in annoy_res:
             bm25_dict[item] = score*additive_semantic_weight + bm25_dict[item]*score
        sorted_res = sorted(bm25_dict.items(), key = lambda x: x[1])
        results = self.annoy_search.df.iloc[[idx for idx,_ in sorted_res]][list(self.bm25.keys())]
        if include_distances:
            results["Score"] = [score for _,score in sorted_res]
        return results[:n_items]