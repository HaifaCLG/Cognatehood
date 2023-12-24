from lingua import Language, LanguageDetectorBuilder
import stanza
import epitran

ROOT = '/home/yuli_zeira/Cognatehood/'
IPA_CHARS = ['p', 'e', 'ɾ', 'o', 'i', 's', 'x', 'a', 'l', 'k', 'd', 'n', 'we',
             'b', 't', 'm', 'ɡ', 'f', 'u', 'jo', 't͡ʃ', 'ja', 'ʝ', 'je', '',
             'si', 'se', 'xe', 'wa', 'w', 'ɡi', 'r', 'ks', 'ɡw', 'wi', 'ɡe',
             'xi', 'ju', 'wo', 'ʃ', 'ɲ', 't͡ɬ', 't͡s', 'æ', 'ɹ̩', 'ð', 'ɪ', 'j',
             'd͡ʒ', 'ʌ', 'ɑ', 'ə', 'h', 'ɹ', 'ŋ', 'ɔ', 'ʊ', 'z', 'v', 'ɛ', 'θ',
             'ʒ']
ABC = 'abcdefghijklmnopqrstuvwxyz'
SPANISH_ABC = 'abcdefghijklmnopqrstuvwxyzñúóéáíü'

LANG = {'en': 'eng-Latn', 'es': 'spa-Latn'}

LANGS = {Language.ENGLISH: 'en', Language.SPANISH: 'es', None: 'none'}
detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.SPANISH).build()
EN_WORD_FILES = [ROOT + 'embeddings/lexicons/english_words.txt',
                ROOT + '/word_files/bm_lemmas_en.txt',
                ROOT + '/word_files/lince_lemmas_en.txt']
ES_WORD_FILES = [ROOT + 'embeddings/lexicons/spanish_words_new.txt',
                ROOT + '/word_files/bm_lemmas_es.txt',
                ROOT + '/word_files/lince_lemmas_es.txt']


class EmbeddingEliminator:
    
    def __init__(self, language: str, words_files: list[str], alphabet: str):
        self._lang = language
        self._words_files = words_files
        self._words = list()
        self.read_words(words_files)
        self._alphabet = alphabet
        self._lemmatize = stanza.Pipeline(lang=language, processors='tokenize,lemma')
        self._trans = epitran.Epitran(LANG[language])
        self._detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.SPANISH).build()
    
    def read_words(self, files: list[str]):
        for file in files:
            with open(file, 'r') as f:
                for line in f.readlines():
                    self._words.append(line[:-1])
    
    def is_lemma(self, word: str) -> str:
        lemma = self._lemmatize(word).sentences[0].words[0].lemma
        if not lemma:
            return ''
        lemma =  lemma.lower()
        if word != lemma:
            return ''
        return word
    
    def is_in_lang(self, word: str) -> str:
        for char in word:
            if char not in self._alphabet:
                return ''
        for char in self._trans.trans_list(word):
            if char not in IPA_CHARS:
                return ''
        if word in self._words:
            return word
        if LANGS[self._detector.detect_language_of(word)] == self._lang:
            return word
        return ''
    
    def write_vecs(self, original_vecs: str, new_vecs: str):
        count = 0
        with open(original_vecs, 'r') as f:
            with open(new_vecs, 'w') as ff:
                for i, line in enumerate(f):
                    split_line = line.split()
                    word, vec = split_line[0], split_line[1:]
                    if len(vec) != 300:
                        continue
                    if self.is_in_lang(word) and self.is_lemma(word):
                        ff.write(line)
                        count += 1
                    if i % 5000 == 0:
                        print(i, word, self.is_lemma(word))
                    if count == 70000:
                        break


# english_vecs = EmbeddingEliminator('en', EN_WORD_FILES, ABC)
# english_vecs.write_vecs(ROOT + 'embeddings/wiki.en.align.vec', ROOT + 'embeddings/final_en_vecs.vec')

# english_vecs = EmbeddingEliminator('es', ES_WORD_FILES, ABC)
# english_vecs.write_vecs(ROOT + 'embeddings/wiki.es.align.vec', ROOT + 'embeddings/final_es_vecs.vec')

import epitran
import json

english = epitran.Epitran('eng-Latn')
spanish = epitran.Epitran('spa-Latn')

def create_transliteration_dict(vecs_file: str, trans_file: str, lang: int, first=False):
    if first:
        transliteration_dict = dict()
    else:
        with open(trans_file, 'r') as f:
            transliteration_dict = json.load(f)
    print(len(transliteration_dict.keys()))
    with open(vecs_file, 'r') as f:
        for line in f.readlines():
            word = line.split()[0]
            if lang == 0:
                trans = english.trans_list(word)
            else:
                trans = spanish.trans_list(word)
            transliteration_dict[word] = trans
    print('Loaded!')
    with open(trans_file, 'w') as f:    
        json.dump(transliteration_dict, f)
    print('Done!')

# create_transliteration_dict(ROOT + '/embeddings/final_en_vecs.vec',
#                            ROOT + '/word_files/en_translit.json', 0)
# create_transliteration_dict(ROOT + '/embeddings/final_es_vecs.vec',
#                            ROOT + '/word_files/es_translit.json', 1)

create_transliteration_dict(ROOT + '/embeddings/final_en_vecs.vec',
                           ROOT + '/word_files/en_translit.json', 0, True)
create_transliteration_dict(ROOT + '/word_files/bm_lemmas_en.txt',
                           ROOT + '/word_files/en_translit.json', 0)
create_transliteration_dict(ROOT + '/word_files/bm_lemmas_es.txt',
                           ROOT + '/word_files/es_translit.json', 1)
create_transliteration_dict(ROOT + '/word_files/lince_lemmas_en.txt',
                           ROOT + '/word_files/en_translit.json', 0)
create_transliteration_dict(ROOT + '/word_files/lince_lemmas_es.txt',
                           ROOT + '/word_files/es_translit.json', 1)

