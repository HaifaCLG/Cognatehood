#!/usr/bin/env python
# coding: utf-8

# In[1]:


ROOT = '/home/yuli_zeira/Cognatehood/'
ENG = 'eng'
SPA = 'spa'


# In[2]:


BI_INSERTION_0 = ROOT + 'char_weights/alphabet/new/insertion_0'
BI_DELETION_0 = ROOT + 'char_weights/alphabet/new/deletion_0'
BI_SUBSTITUTION_0 = ROOT + 'char_weights/alphabet/new/substitution_0'

BI_INSERTION_1 = ROOT + 'char_weights/alphabet/new/insertion_1'
BI_DELETION_1 = ROOT + 'char_weights/alphabet/new/deletion_1'
BI_SUBSTITUTION_1 = ROOT + 'char_weights/alphabet/new/substitution_1'

TRI_INSERTION = ROOT + 'char_weights/alphabet/new/insertion_tri'
TRI_DELETION = ROOT + 'char_weights/alphabet/new/deletion_tri'
TRI_SUBSTITUTION = ROOT + 'char_weights/alphabet/new/substitution_tri'

IPA_BI_INSERTION_0 = ROOT + 'char_weights/ipa/new/ipa_insertion_0'
IPA_BI_DELETION_0 = ROOT + 'char_weights/ipa/new/ipa_deletion_0'
IPA_BI_SUBSTITUTION_0 = ROOT + 'char_weights/ipa/new/ipa_substitution_0'

IPA_BI_INSERTION_1 = ROOT + 'char_weights/ipa/new/ipa_insertion_1'
IPA_BI_DELETION_1 = ROOT + 'char_weights/ipa/new/ipa_deletion_1'
IPA_BI_SUBSTITUTION_1 = ROOT + 'char_weights/ipa/new/ipa_substitution_1'

IPA_TRI_INSERTION = ROOT + 'char_weights/ipa/new/ipa_insertion_tri'
IPA_TRI_DELETION = ROOT + 'char_weights/ipa/new/ipa_deletion_tri'
IPA_TRI_SUBSTITUTION = ROOT + 'char_weights/ipa/new/ipa_substitution_tri'


# In[3]:


from numpy import dot
from numpy.linalg import norm
import numpy
import json
from os import listdir


class SimilarityMeasure:

    def __init__(self, en_vecs_path: str, es_vecs_path: str, en_words: str, es_words: str):
        self._en_words, self._en_vecs, self._en_words_rev = self.load_vectors(en_vecs_path)
        self._es_words, self._es_vecs, self._es_words_rev = self.load_vectors(es_vecs_path)
        self._cos_sim = lambda a, b: dot(a, b) / (norm(a) * norm(b))
        self._en_lemmas_corpus = self.load_words(en_words)
        self._es_lemmas_corpus = self.load_words(es_words)
        self._en_similarities, self._es_similarities = dict(), dict()

    @staticmethod
    def load_vectors(path: str):
        words, vectors, words_rev = dict(), list(), dict()
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                line = line.split()
                word = line[0]
                words[word] = i
                words_rev[i] = word
                vectors.append(line[1:])
                if i % 50000 == 0:
                    print(i)
                if i == 750000:
                    break
        print('Done!')
        return words, numpy.array(vectors, dtype=float), words_rev
    
    def load_words(self, words: str):
        lemmas_corpus = list()
        with open(words, 'r') as f:
            for line in f.readlines():
                if len(line[:-1]) > 2:
                    lemmas_corpus.append(line[:-1].lower())
        print('Done loading words!')
        return lemmas_corpus
    
    def get_similarities(self, word: str, lang: str, filename: str):
        similarities = dict()
        if lang == 'eng':
            try:
                word_ind = self._en_words[word]
                word_vec = self._en_vecs[word_ind]
            except KeyError:
                  print(f'No representation for {word} ({lang})')
                  return
            for ind in self._es_words_rev.keys():
                if len(self._es_words_rev[ind]) > 2:
                    similarities[self._es_words_rev[ind]] = [self._cos_sim(word_vec, self._es_vecs[ind]), 0.0]
                else:
                    print(self._es_words_rev[ind])
        else:
            try:
                word_ind = self._es_words[word]
                word_vec = self._es_vecs[word_ind]
            except KeyError:
                  print(f'No representation for {word} ({lang})')
                  return
            for ind in self._en_words_rev.keys():
                if len(self._en_words_rev[ind]) > 2:
                    similarities[self._en_words_rev[ind]] = [self._cos_sim(word_vec, self._en_vecs[ind]), 0.0]
                else:
                    print(self._en_words_rev[ind])
        with open(filename, 'w') as f:
            json.dump(similarities, f)
    
    def get_all_similarities(self, en_output: str, es_output: str):
        print(len(listdir(en_output)), len(listdir(es_output)))
        en_words = [w for w in self._en_lemmas_corpus if w + '.json' not in listdir(en_output)]
        es_words = [w for w in self._es_lemmas_corpus if w + '.json' not in listdir(es_output)]
        for word in en_words:
            self.get_similarities(word, 'eng', en_output + word + '.json')
        print('Finished English words!')
        for word in es_words:
            self.get_similarities(word, 'spa', es_output + word + '.json')
        print('Finished Spanish words!')


# In[4]:


# sm = SimilarityMeasure(ROOT + 'embeddings/final_en_vecs.vec',
#                        ROOT + 'embeddings/final_es_vecs.vec',
#                        ROOT + 'word_files/bm_lemmas_en.txt',
#                        ROOT + 'word_files/bm_lemmas_es.txt')                                                  
# sm.get_all_similarities(ROOT + 'measures/en_lemmas_similarities/', ROOT + 'measures/es_lemmas_similarities/')


# sm = SimilarityMeasure(ROOT + 'embeddings/final_en_vecs.vec',
#                        ROOT + 'embeddings/final_es_vecs.vec',
#                        ROOT + 'word_files/lince_lemmas_en.txt',
#                        ROOT + 'word_files/lince_lemmas_es.txt')                                                  
# sm.get_all_similarities(ROOT + 'measures/en_lemmas_similarities/', ROOT + 'measures/es_lemmas_similarities/')


# # In[6]:



import pickle


class FormMeasure:

    def __init__(self, insertion, deletion, substitution, ind: int, ipa=False):
        self._is_ipa = ipa
        self._ind = ind
        self._insertion = self.unpickle_weights(insertion)
        self._deletion = self.unpickle_weights(deletion)
        self._substitution = self.unpickle_weights(substitution)
        self.make_n_gram = [self.make_n_gram_first, self.make_n_gram_second,
                            self.make_trigram][ind]

    @staticmethod
    def unpickle_weights(weight_file):
        with open(weight_file, 'rb') as f:
            weight_dict = pickle.load(f)
        return weight_dict

    def make_n_gram_first(self, word, k: int) -> tuple[str]:
        if self._is_ipa:
            tup = [''] + word
        else:
            tup = [''] + list(word)
        new_tup = tup[k: k + 2]
        return tuple(new_tup)

    def make_n_gram_second(self, word, k: int) -> tuple[str]:
        if self._is_ipa:
            tup = word + ['']
        else:
            tup = list(word) + ['']
        new_tup = tup[k: k + 2]
        return tuple(new_tup)

    def make_trigram(self, word, k: int) -> tuple[str]:
        if self._is_ipa:
            tup = [''] + word + ['']
        else:
            tup = [''] + list(word) + ['']
        new_tup = tup[k: k + 3]
        return tuple(new_tup)

    def operations(self, ngram_1, ngram_2):
        if self._ind == 2:
            if ngram_1 in self._deletion.keys():
                delete = self._deletion[ngram_1]
            else:
                delete = 1.0
            if ngram_2 in self._insertion.keys():
                insert = self._insertion[ngram_2]
            else:
                insert = 1.0
            if (ngram_1, ngram_2) in self._substitution.keys():
                substitute = self._substitution[(ngram_1, ngram_2)]
            else:
                substitute = 1.0
        else:
            delete = self._deletion[ngram_1]
            insert = self._insertion[ngram_2]
            substitute = self._substitution[(ngram_1, ngram_2)]
        return delete, insert, substitute
    
    def initialize(self, word_1: str, word_2: str) -> tuple[dict, dict]:
        # Set up:
        distances = {(-1, -1): 0}
        # Initialize the arrays:
        for i in range(len(word_1)):
            w1_ngram = self.make_n_gram(word_1, i)
            if self._ind == 2 and w1_ngram not in self._deletion.keys():
                distances[(i, -1)] = 1.0 + distances[(i - 1, -1)]
            else:
                distances[(i, -1)] = self._deletion[w1_ngram] + distances[(i - 1, -1)]
        for j in range(len(word_2)):
            w2_ngram = self.make_n_gram(word_2, j)
            if self._ind == 2 and w2_ngram not in self._insertion.keys():
                distances[(-1, j)] = 1.0 + distances[(-1, j - 1)]
            else:
                distances[(-1, j)] = self._insertion[w2_ngram] + distances[(-1, j - 1)]
        return distances

    def edit_distance(self, word_1: str, word_2: str) -> tuple[float, float, float]:
        # Set up:
        if not self._is_ipa:
            word_1, word_2 = word_1.lower(), word_2.lower()
        len_1, len_2 = len(word_1), len(word_2)
        length = max(len_1, len_2)
        distances = self.initialize(word_1, word_2)
        # Fill distances and operations:
        for i in range(len_1):
            for j in range(len_2):
                w1_ngram, w2_ngram = self.make_n_gram(word_1, i), self.make_n_gram(word_2, j)
                delete, insert, substitute = self.operations(w1_ngram, w2_ngram)
                if word_1[i] == word_2[j]:
                    cost = 0
                else:
                    cost = substitute
                curr_distances = [distances[(i, j - 1)] + insert,
                                  distances[(i - 1, j)] + delete,
                                  distances[(i - 1, j - 1)] + cost]
                distances[(i, j)] = min(curr_distances)
        return distances[(len_1 - 1, len_2 - 1)] / length


# # In[8]:


import json
from os import listdir
import time


class WordDictEditor:

    def __init__(self, alphabet_weights, ipa_weights, en_translit: str, es_translit: str):
        self._bigram_measure_0 = FormMeasure(alphabet_weights[0][0], alphabet_weights[0][1],
                                             alphabet_weights[0][2], 0)
        self._bigram_measure_1 = FormMeasure(alphabet_weights[1][0], alphabet_weights[1][1],
                                             alphabet_weights[1][2], 1)
        self._trigram_measure = FormMeasure(alphabet_weights[2][0], alphabet_weights[2][1],
                                            alphabet_weights[2][2], 2)
        self._ipa_bigram_measure_0 = FormMeasure(ipa_weights[0][0], ipa_weights[0][1],
                                                 ipa_weights[0][2], 0, True)
        self._ipa_bigram_measure_1 = FormMeasure(ipa_weights[1][0], ipa_weights[1][1],
                                                 ipa_weights[1][2], 1, True)
        self._ipa_trigram_measure = FormMeasure(ipa_weights[2][0], ipa_weights[2][1],
                                                ipa_weights[2][2], 2, True)
        self._en_translit_dict = self.load_dict(en_translit)
        self._es_translit_dict = self.load_dict(es_translit)
        self._avg = lambda a, b, c: (0.4 * a) + (0.4 * b) + (0.2 * c)
    
    @staticmethod
    def load_dict(dict_file: str) -> dict:
        with open(dict_file, 'r') as f:
            d = json.load(f)
        return d

    def measure_dist(self, eng_word: str, spa_word: str,
                     eng_ipa: list, spa_ipa: list) -> list[float]:
        x = self._avg(self._bigram_measure_0.edit_distance(eng_word, spa_word),
                      self._bigram_measure_1.edit_distance(eng_word, spa_word),
                      self._trigram_measure.edit_distance(eng_word, spa_word))
        y = self._avg(self._ipa_bigram_measure_0.edit_distance(eng_ipa, spa_ipa),
                      self._ipa_bigram_measure_1.edit_distance(eng_ipa, spa_ipa),
                      self._ipa_trigram_measure.edit_distance(eng_ipa, spa_ipa))
        return 1 - (0.5 * x + 0.5 * y)
                                                  
    def add_form_similarity(self, word: str, word_ipa: list[str], words_file: str, lang: str):
        with open(words_file, 'r') as f:
            words_dict = json.load(f)
        if lang == 'eng':
            for w in words_dict.keys():
                if words_dict[w][1]:
#                     continue
                    return
                w_ipa = self._es_translit_dict[w]
                words_dict[w][1] = self.measure_dist(word, w, word_ipa, w_ipa)
        else:
            count = 0
            for w in words_dict.keys():
                if words_dict[w][1]:
#                     continue
                    return
                count += 1
                w_ipa = self._en_translit_dict[w]
                words_dict[w][1] = self.measure_dist(w, word, w_ipa, word_ipa)
        print(word)
        new = dict([(k, words_dict[k]) for k in words_dict.keys()])
        with open(words_file, 'w') as f:
            json.dump(new, f)
    
    def add_all_form(self, dir_name: str, lang: str):
        for wf in listdir(dir_name):
            if wf[-5:] != '.json':
                print('not json', wf)
                continue
            word = wf[:-5]
            if word == 'pam':
                continue
            wf = dir_name + wf
            if lang == 'eng':
                ipa = self._en_translit_dict[word]
            else:
                ipa = self._es_translit_dict[word]
#             print(word, wf, ipa)
            self.add_form_similarity(word, ipa, wf, lang)


# word_dict_editor = WordDictEditor(((BI_INSERTION_0, BI_DELETION_0, BI_SUBSTITUTION_0),
#                                    (BI_INSERTION_1, BI_DELETION_1, BI_SUBSTITUTION_1),
#                                    (TRI_INSERTION, TRI_DELETION, TRI_SUBSTITUTION)),
#                                    ((IPA_BI_INSERTION_0, IPA_BI_DELETION_0, IPA_BI_SUBSTITUTION_0),
#                                    (IPA_BI_INSERTION_1, IPA_BI_DELETION_1, IPA_BI_SUBSTITUTION_1),
#                                    (IPA_TRI_INSERTION, IPA_TRI_DELETION, IPA_TRI_SUBSTITUTION)),
#                                  ROOT + '/word_files/en_translit.json',
#                                  ROOT + '/word_files/es_translit.json')
                                               
# word_dict_editor.add_all_form(ROOT + 'measures/en_lemmas_similarities/', 'eng')
# word_dict_editor.add_all_form(ROOT + 'measures/es_lemmas_similarities/', 'spa')                                                  

import numpy as np
import json
from os import listdir
from math import sqrt

class Cognatehood:
    
    def __init__(self, en_words: str, es_words: str, bad_words_en: str, bad_words_es: str):
        self._en_dir, self._es_dir = en_words, es_words
        self._bad_words_en, self._bad_words_es = bad_words_en, bad_words_es
        self._temp_mean, self._temp_std = list(), list()
#         self._mean = [-0.07194480432566823, 0.5644210684396023]
#         self._std = [0.07350102476879125, 0.11018131478978205]
#         self._min =  [-4.986093878393775, -5.122656863520613]
#         self._max = [10.530336636851322, 3.9532921928863414]
        self._mean = [0, 0]
#         self.calculate_mean()
        self._std = [0, 0]
        self.calculate_std()
        self._min = [1, 1]
        self._max = [-1, -1]
        self.get_min_max()
        self._diff = [self._max[i] - self._min[i] for i in [0,1]]
#         self.calculate_mean()
#         self.calculate_std()
#         self.get_min_max()
    
    @staticmethod
#     def load_similarities(dirname: str, file_name: str, bad_words: str, prev_list=[]) -> (list[list], int):
    def load_similarities(dirname: str, file_name: str, prev_list=[]) -> (list[list], int):
        similarities, num_words = [[], []], 0
        if len(file_name[:-5]) < 3:
            print(file_name, file_name[-5:])
        if file_name[-5:] == '.json':
            with open(dirname + file_name, 'r') as f:
                try:
                    w = json.load(f)
                except:
#                     print(file_name)
                    w = dict()
                for k in w.keys():
                    if 0 <= w[k][1] <= 1 and len(k) > 2:   # Changed only words of length 3 and above
                        if k not in prev_list:
                            num_words += 1
                            similarities[0].append(w[k][0])   # X axis - semantic sim
                            similarities[1].append(w[k][1])   # Y axis - form sim
#                     elif len(k) > 2:
#                         with open(bad_words, 'a') as ff:
#                             ff.writelines(f'{file_name[:-5]} {k} {w[k][0]} {w[k][1]}\n')
        return similarities, num_words
    
    def calculate_mean(self):
        done_en = []
        total, count = [0, 0], 0
        c = 0
        for dir_name in [self._en_dir, self._es_dir]:
            print(dir_name)
            print(total, count)
            for wf in listdir(dir_name):
                if dir_name == self._en_dir:
                    done_en.append(wf[:-5])
                    similarities, num = self.load_similarities(dir_name, wf, self._bad_words_en)
                else:
                    similarities, num = self.load_similarities(dir_name, wf, self._bad_words_es, done_en)
                for i in [0,1]:
                    total[i] += sum(similarities[i])
                count += num
                c += 1
                if c % 1000 == 0:
                    print(c)
#                 if c < 100:
#                     if dir_name == self._en_dir:
#                         done_en.append(wf[:-5])
#                         similarities, num = self.load_similarities(dir_name, wf)
#                     else:
#                         similarities, num = self.load_similarities(dir_name, wf, done_en)
#                     for i in [0,1]:
#                         total[i] += sum(similarities[i])
#                     count += num
#                     c += 1
#                     if c % 10 == 0:
#                         print(c)
        self._mean = [total[i] / count for i in [0, 1]]
        print('Mean:', self._mean)
    
    def calculate_std(self):
        done_en = []
        total, count = [0, 0], 0
        c = 0
        for dir_name in [self._en_dir, self._es_dir]:
            print(dir_name)
            print(total, count)
            for wf in listdir(dir_name):
                if dir_name == self._en_dir:
                    done_en.append(wf[:-5])
                    similarities, num = self.load_similarities(dir_name, wf)
                else:
                    similarities, num = self.load_similarities(dir_name, wf, done_en)
                if num:
                    for i in [0, 1]:
                        for j in range(num):
                            similarities[i][j] = (similarities[i][j] - self._mean[i]) ** 2
                        total[i] += sum(similarities[i])
                count += num
                c += 1
                if c % 1000 == 0:
                    print(c)
        self._std = [sqrt(total[i] / count) for i in [0, 1]]
        print('STD:', self._std)
    
    def standard_score(self, score: list[float]) -> list[float]:
        return [(score[i] -  self._mean[i]) * (1 / self._std[i]) for i in [0, 1]]
    
    def normalize_score(self, score: list[float]):
        return [(score[i] - self._min[i]) / self._diff[i] for i in [0, 1]]
    
    def get_min_max(self):
        maxi, mini = [-1, -1], [1, 1]
        count = 0
        for words_dir in [self._en_dir, self._es_dir]:
            for wf in listdir(words_dir):
                if wf[-5:] != '.json':
#                 if wf[-5:] != '.json' or len(wf[:-5]) < 3:
                    continue
                with open(words_dir + wf, 'r') as f:
                    try:
                        word_dict = json.load(f)
                        count += 1
                    except:
#                         print('Not opened :(', wf)
                        word_dict = dict()
                    if count % 1000 == 0:
                        print(wf[:-5], count)
                    for elem in word_dict.keys():
                        if 0 <= word_dict[elem][1] <= 1:
                            updated = self.standard_score(word_dict[elem])
                            for i in [0, 1]:
                                if updated[i] > maxi[i]:
                                    maxi[i] = updated[i]
                                if updated[i] < mini[i]:
                                    mini[i] = updated[i]
#                         else:
#                             print(wf[:-5], elem, word_dict[elem][1])
        self._min = mini
        self._max = maxi
        print(f'Min: {mini}\tMax: {maxi}')
    
    def save_new_scores(self, raw_dir: str, new_dir: str):
        for wf in listdir(raw_dir):
            if wf[-5:] != '.json' or len(wf[:-5:]) < 3:
                continue
            with open(raw_dir + wf, 'r') as f:
                try:
                    word_dict = json.load(f)
                except:
#                     print(wf)
                    word_dict = dict()
                new = dict()
                for elem in word_dict.keys():
                    if len(elem) > 2 and 0 <= word_dict[elem][1] <= 1:
                        updated = self.standard_score(word_dict[elem])
                        updated = self.normalize_score(updated)
                        new[elem] = updated
                with open(new_dir + wf, 'w') as f:
                    json.dump(new, f)

                    
# cog = Cognatehood(ROOT + 'measures/en_lemmas_similarities/', ROOT + 'measures/es_lemmas_similarities/',
#                  ROOT + 'measures/bad_word_pairs_en.txt', ROOT + 'measures/bad_word_pairs_es.txt')
# cog.save_new_scores(ROOT + 'measures/en_lemmas_similarities/', ROOT + 'measures/en_lemmas_new/')
# cog.save_new_scores(ROOT + 'measures/es_lemmas_similarities/', ROOT + 'measures/es_lemmas_new/')


# from os import listdir
# import json

# def get_cog(word: str, full_path_file: str, cognates_file: str) -> (str, float):
#     cognatehood, cognate = 0, ''
#     with open(full_path_file, 'r') as f:
#         similarities = json.load(f)
#     for elem in similarities.keys():
#         coghood = (similarities[elem][0] + similarities[elem][1]) / 2
#         if coghood > cognatehood:
#             cognatehood = coghood
#             cognate = elem
#     with open(cognates_file, 'a') as f:
#         f.writelines(f'{word} {cognate} {cognatehood}\n')


# def all_cogs(words_path: str, cognates_file: str):
#     for filename in listdir(words_path):
#         get_cog(filename[:-5], words_path + filename, cognates_file)
        
# all_cogs(ROOT + 'measures/en_lemmas_new/', ROOT + 'measures/cognatehood_final_lists/en_cognates.txt')
# all_cogs(ROOT + 'measures/es_lemmas_new/', ROOT + 'measures/cognatehood_final_lists/es_cognates.txt')

# by_degrees = dict()

# with open(ROOT + 'measures/en_cognates.txt', 'r') as f:
#     for line in f.readlines():
#         line = line.split()
#         word, deg, cog = line[0], float(line[1]), line[2]
#         if deg in by_degrees.keys():
#             by_degrees[deg].append((word, cog))
#         else:
#             by_degrees[deg] = [(word, cog)]

# for k in by_degrees.keys():
#     if len(by_degrees[k]) > 1:
#         print(k, by_degrees[k])

# print()
# all_degs = list(by_degrees.keys())
# all_degs.sort(reverse=True)


# with open(ROOT + 'measures/cognatehood_final_lists/en_cognates_sorted.txt', 'w') as f:
#     for d in all_degs:
#         for cogs in by_degrees[d]:
#             if cogs[0]:
#                 f.writelines(f'{cogs[0]} - {cogs[1]}: {round(d, 4)}\n')

from os import listdir
import json
from random import sample

en_words = ROOT + 'measures/en_lemmas_new/'
es_words = ROOT + 'measures/es_lemmas_new/'

en_list = listdir(en_words)
es_list = [word[:-5] for word in listdir(es_words)]

pairs = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}

for elem in en_list:
    with open(en_words + elem, 'r') as f:
        sim_dict = json.load(f)
        for elem_es in es_list:
            if elem_es in sim_dict.keys():
                sim = round((sim_dict[elem_es][0] + sim_dict[elem_es][1]) * 0.5, 4)
                pairs[round(sim * 10)].append((elem, elem_es, sim))

with open(ROOT + 'measures/all_dataset_pairs.json', 'w') as f:
    json.dump(pairs, f)

# samples = []
            
# for sim in pairs.keys():
#     if len(pairs[sim]) > 15:
#         samp = sample(pairs[sim], 15)
#     else:
#         samp = pairs[sim]
#     samples += samp
            
# with open(ROOT + 'pairs_for_survey.txt', 'w') as f:
#     for elem in samples:
#         f.writelines(elem[0] + ' ' + elem[1] + ' ' + str(elem[2]) + '\n')