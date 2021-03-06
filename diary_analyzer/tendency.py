from collections import defaultdict
from nltk.corpus import wordnet as wn
from pprint import pprint
import csv
import os
import numpy as np
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
from math import log

from diary_analyzer import tagger
from diary_analyzer.tagger import TAG_WORD, TAG_WORD_POS, \
    TAG_DEPENDENCY, TAG_WORD_ROLE, TAG_NAMED_ENTITY


# abstract class for WordSet Corpus Retriever
class WordSetCorpusRetriever(object):
    IDX_SYNSET = 0
    IDX_LEVEL = 1
    IDX_LEMMA_WORDS = 2
    IDX_LEMMA_COUNT = 3

    def __init__(self, categoricals=[]):
        self.synset_list = list()
        self.categoricals = categoricals
        self.categoricals_lemma = self._get_lemmas(categoricals)

    @classmethod
    def _get_lemmas(cls, synset_names):
        lemmas = list()
        for synset_name in synset_names:
            lemmas.extend(_lemmas_to_name_list(wn.synset(synset_name).lemmas()))
        return lemmas

    def get_list(self):
        return self.synset_list

    def find_synset(self, synset, check_cat=False):
        if synset.name() in self.categoricals and not check_cat:
            return None
        for item in self.synset_list:
            if synset.name() == item[self.IDX_SYNSET].name():
                return item[self.IDX_SYNSET]
        return None

    def check_synset_in(self, synset, check_cat=False):
        if self.find_synset(synset, check_cat) is not None:
            return True
        else:
            return False

    def find_word(self, word, check_cat=False, pos='n'):
        if word in self.categoricals_lemma and not check_cat:
            return None
        for item in self.synset_list:
            if pos in item[self.IDX_LEMMA_WORDS].keys():
                if word in item[self.IDX_LEMMA_WORDS][pos]:  # lemma list
                    return item[self.IDX_SYNSET]  # synset
        return None

    def check_word_in(self, word, check_cat=False, pos='n'):
        if self.find_word(word, check_cat, pos) is not None:
            return True
        else:
            return False

    def get_item_word(self, word, check_cat=False, pos='n'):
        if word in self.categoricals_lemma and not check_cat:
            return None
        for item in self.synset_list:
            if pos in item[self.IDX_LEMMA_WORDS].keys():
                if word in item[self.IDX_LEMMA_WORDS][pos]:  # lemma list
                    return item
        return None

    def get_item_synset(self, synset, check_cat=False):
        if synset.name() in self.categoricals and not check_cat:
            return None
        for item in self.synset_list:
            if synset.name() == item[self.IDX_SYNSET].name():
                return item
        return None


# class for retrieving word set based on WordNet hyponyms
class HyponymCorpusRetriever(WordSetCorpusRetriever):
    def __init__(self, *root_synsets, max_level=10, excepts=[], categoricals=[]):
        if type(root_synsets[0]) is list or type(root_synsets[0]) is tuple:
            root_synsets = root_synsets[0]
        self.synset_list = list()
        self.excepts = excepts
        self.categoricals = categoricals
        self.categoricals_lemma = self._get_lemmas(categoricals)
        for synset in root_synsets:
            self.synset_list += self._collect_hyponyms(synset, max_level)

    def _collect_hyponyms(self, root_synset, max_level=1, now_level=0):
        if now_level == max_level: return []
        now_level += 1
        hyponym_list = list()
        for hyponym in root_synset.hyponyms():
            # names, counts = _lemmas_to_name_list(hyponym.lemmas(), True)
            # hyponym_list.append((hyponym, max_level, names, counts))

            if hyponym.name() in self.excepts:
                continue

            # tuple (sysnet, level, lemma_words)
            lemma_words = {}
            lemma_words[hyponym.pos()] = _lemmas_to_name_list(hyponym.lemmas())
            hyponym_list.append((hyponym, now_level, lemma_words))
            hyponym_list = hyponym_list + self._collect_hyponyms(hyponym, max_level, now_level)
        return hyponym_list


# class for retrieving word set based on WordNet hypernyms
class HypernymCorpusRetriever(WordSetCorpusRetriever):
    def __init__(self, *root_synsets, max_level=10, excepts=[], categoricals=[]):
        if type(root_synsets[0]) is list or type(root_synsets[0]) is tuple:
            root_synsets = root_synsets[0]
        self.synset_list = list()
        self.excepts = excepts
        self.categoricals = categoricals
        self.categoricals_lemma = self._get_lemmas(categoricals)
        for synset in root_synsets:
            self.synset_list += self._collect_hyponyms(synset, max_level)

    def _collect_hyponyms(self, root_synset, max_level=1, now_level=0):
        if now_level == max_level: return []
        now_level += 1
        hypernym_list = list()
        for hypernym in root_synset.hypernyms():
            # names, counts = _lemmas_to_name_list(hyponym.lemmas(), True)
            # hyponym_list.append((hyponym, max_level, names, counts))
            if hypernym.name() in self.excepts:
                continue

            # tuple (sysnet, level, lemma_words)
            lemma_words = {}
            lemma_words[hypernym.pos()] = _lemmas_to_name_list(hypernym.lemmas())
            hypernym_list.append((hypernym, now_level, lemma_words))
            hypernym_list = hypernym_list + self._collect_hyponyms(hypernym, max_level, now_level)
        return hypernym_list

    def get_list(self):
        return self.synset_list


# class for retrieving word set by words list file
class ListFileCorpusRetriever(WordSetCorpusRetriever):
    def __init__(self, file_path, pos_fliter=('n','a','v','r','s'),
                 excepts=[], categoricals=[]):
        self.synset_list = list()
        self.pos_fliter = pos_fliter
        if file_path:
            self.synset_list = self._load_list_file(file_path)
        self.excepts = excepts
        self.categoricals = categoricals
        self.categoricals_lemma = self._get_lemmas(categoricals)

    def _load_list_file(self, file_path):
        synset_list = list()
        with open(file_path, "r") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                if line[0] == '#':
                    continue
                line = _text_to_lemma_format(line.strip())
                for synset in wn.synsets(line):
                    if synset.pos() not in self.pos_fliter:
                        continue
                    lemma_words = {}
                    lemma_words[synset.pos()] = _lemmas_to_name_list(synset.lemmas())
                    synset_list.append((synset, 0, lemma_words,
                                        synset.lexname(), synset.definition()))
            file.close()
        return synset_list


# class for retrieving word set by synset list file
class SynsetListFileCorpusRetriever(WordSetCorpusRetriever):
    def __init__(self, file_path, excepts=[], categoricals=[]):
        self.synset_list = list()
        if file_path:
            self.synset_list = self._load_list_file(file_path)
        self.excepts = excepts
        self.categoricals = categoricals
        self.categoricals_lemma = self._get_lemmas(categoricals)

    def _load_list_file(self, file_path):
        synset_list = list()
        with open(file_path, "r") as file:
            synset_group = []
            while True:
                line = file.readline()
                if not line:
                    break
                if line == '\n':
                    synsets = []
                    # lemmas = []
                    lemmas = {}
                    for synset_name in synset_group:
                        synset = wn.synset(synset_name)
                        if synset:
                            synsets.append(wn.synset(synset_name))
                            if synset.pos() not in lemmas.keys():
                                lemmas[synset.pos()] = list()
                            lemmas[synset.pos()].extend(_lemmas_to_name_list(synset.lemmas()))
                    synset_list.append((synsets, 0, lemmas))

                    synset_group = [] # init
                    continue
                else:
                    synset_name = line.split(' [')[0]
                    synset_group.append(synset_name)
            file.close()
        return synset_list

    def get_list(self):
        return self.synset_list

    def find_synset(self, synset, check_cat=False):
        if synset.name() in self.categoricals and not check_cat:
            return None
        for item in self.synset_list:
            synset_group = item[self.IDX_SYNSET]
            for s in synset_group:
                if synset.name() == s.name():
                    return synset_group[0]
        return None

    def find_word(self, word, check_cat=False, pos='n'):
        if word in self.categoricals_lemma and not check_cat:
            return None
        for item in self.synset_list:
            synset_group = item[self.IDX_SYNSET]
            for s in synset_group:
                if pos in item[self.IDX_LEMMA_WORDS].keys():
                    if word in item[self.IDX_LEMMA_WORDS][pos]:
                        return synset_group[0]
        return None

    def get_item_word(self, word, check_cat=False, pos='n'):
        if word in self.categoricals_lemma and not check_cat:
            return None
        for item in self.synset_list:
            synset_group = item[self.IDX_SYNSET]
            for s in synset_group:
                if pos in item[self.IDX_LEMMA_WORDS].keys():
                    if word in item[self.IDX_LEMMA_WORDS][pos]:
                        return item
        return None

    def get_item_synset(self, synset, check_cat=False):
        if synset.name() in self.categoricals and not check_cat:
            return None
        for item in self.synset_list:
            synset_group = item[self.IDX_SYNSET]
            for s in synset_group:
                if synset.name() == s.name():
                    return item
        return None


# class for retrieving word set by preference verb list file
class PreferenceVerbCorpusRetriever(ListFileCorpusRetriever):
    def __init__(self, like_file_path, dislike_file_path, excepts=[]):
        self.synset_list = list()
        self.excepts = excepts
        if like_file_path and dislike_file_path:
            self.synset_list = self._load_list_file(like_file_path)
            self.synset_list.extend(self._load_list_file(dislike_file_path))

    def _load_list_file(self, file_path):
        synset_list = list()
        with open(file_path, "r") as file:
            now_score = 0
            while True:
                line = file.readline()
                if not line:
                    break
                if line[0] == '#' or line == '\n':
                    continue

                if line[0] == '@':
                    now_score = float(line.split(' ')[1])

                line = _text_to_lemma_format(line.strip())
                for synset in wn.synsets(line):
                    if synset.pos() != 'v':
                        continue
                    if synset.name() in self.excepts:
                        continue
                    lemma_words = {}
                    lemma_words[synset.pos()] = _lemmas_to_name_list(synset.lemmas())
                    synset_list.append((synset, 0, lemma_words,
                                        synset.lexname(), synset.definition(), now_score))
            file.close()
        return synset_list

    def get_score(self, lemma_word):
        for pref_synset in self.synset_list:
            if lemma_word in pref_synset[2]['v']:
                return pref_synset[5]
        return 0


# class for SentiWordNet Corpus Retriever
class SentiWordNetRetriever(object):
    IDX_POS = 0
    IDX_OFFSET_ID = 1
    IDX_SCORE_POS = 2
    IDX_SCORE_NEG = 3
    IDX_SYNSETS = 4
    IDX_GLOSS = 5

    def __init__(self, filepath):
        self._load_senti_wordnet(filepath)

    def _load_senti_wordnet(self, filepath):
        self.adjective_scores = dict()
        self.noun_scores = dict()
        self.adverb_scores = dict()
        self.verb_scores = dict()
        self.default_score = {
            'positivity': 0,
            'negativity': 0,
            'objectivity': 1,
            'corpus_count': 0
        }

        with open(filepath, 'r') as sent_file:
            csvReader = csv.reader(sent_file, delimiter='\t')
            for row in csvReader:
                if len(row) == 0 or row[0].startswith('#'):
                    continue

                score_pos = float(row[self.IDX_SCORE_POS])
                score_neg = float(row[self.IDX_SCORE_NEG])
                score_obj = 1 - (score_pos + score_neg)

                # pass objective scored words
                if score_pos == 0 and score_neg == 0:
                    continue

                row_synset = wn._synset_from_pos_and_offset(row[self.IDX_POS],
                                                            int(row[self.IDX_OFFSET_ID]))
                corpus_count = 0
                for lemma in row_synset.lemmas():
                    corpus_count += lemma.count()

                score_dict = {
                    'positivity': score_pos,
                    'negativity': score_neg,
                    'objectivity': score_obj,
                    'corpus_count': corpus_count
                }

                if row[self.IDX_POS] == 'a':
                    self.adjective_scores[int(row[self.IDX_OFFSET_ID])] = score_dict
                elif row[self.IDX_POS] == 'n':
                    self.noun_scores[int(row[self.IDX_OFFSET_ID])] = score_dict
                elif row[self.IDX_POS] == 'r':
                    self.adverb_scores[int(row[self.IDX_OFFSET_ID])] = score_dict
                elif row[self.IDX_POS] == 'v':
                    self.verb_scores[int(row[self.IDX_OFFSET_ID])] = score_dict
            sent_file.close()

    def get_score(self, offset_id, pos):
        try:
            if pos == 'a':
                return self.adjective_scores[offset_id]
            elif pos == 'n':
                return self.noun_scores[offset_id]
            elif pos == 'r':
                return self.adverb_scores[offset_id]
            elif pos == 'v':
                return self.verb_scores[offset_id]
        except:
            pass
        return self.default_score

    def get_score_value(self, offset_id, pos):
        score_dict = self.get_score(offset_id, pos)
        return score_dict['positivity'] - score_dict['negativity']


# class for Tendency Analytics
class TendencyAnalyzer(object):
    """Perform Tendency analysis"""
    DEBUG = False

    SIMILAR_PATH_MAX_HYPERNYM = 2
    SIMILAR_PATH_MAX_HYPONYM = 1

    CLUSTER_CUT_DIST = 5
    CLUSTER_PATH_DIST_MAGNIFICATION = 1

    def __init__(self, senti_wordnet=None):
        self.senti_wordnet = senti_wordnet
        self.pref_verb_corpus = None
        self.words_corpora = defaultdict(lambda: None)

    def add_word_set(self, target, target_type, words_set):
        if words_set:
            self.words_corpora[(target, target_type)] = words_set

    # input: a list of tagged(preprocessed) diary
    def analyze_diaries(self, preprocessed_diaries):
        diary_len = len(preprocessed_diaries)

        # step 2
        # print("\n##### Step 2. #####")
        extracted_sent_dict_list = list()
        for diary_idx in range(0, diary_len):
            diary_tags = preprocessed_diaries[diary_idx]
            extracted_sent_dict = self.extract_sentences(diary_tags)
            extracted_sent_dict_list.append(extracted_sent_dict)
            # print("Diary #%s" % (diary_idx+1))
            # for sent_id, extracted_words in extracted_sent_dict.items():
            #     print('Sen ' + str(sent_id+1) + ':', extracted_words[0])
            #     for idx in range(1, len(extracted_words)):
            #         print('      ', extracted_words[idx])
            # print()

        # step 3
        # print("\n##### Step 3. #####")
        scores_tend_list = list()
        for diary_idx in range(0, diary_len):
            diary_tags = preprocessed_diaries[diary_idx]
            extracted_sent_dict = extracted_sent_dict_list[diary_idx]
            scores_tend = self.compute_tend_scores(diary_tags, extracted_sent_dict, diary_idx + 1)
            scores_tend_list.append(scores_tend)
            # print("Diary #%s" % (diary_idx+1))
            # pprint(dict(scores_tend))
            # print()

        # step 4
        # convert & add scores_pref dictionary to list
        # print("\n##### Step 4. #####")
        tend_list = list()
        for diary_idx in range(0, diary_len):
            scores_tend = scores_tend_list[diary_idx]
            converted_tends = self.convert_scores_dict_to_list(scores_tend)
            tend_list += converted_tends

        # group by category and type
        tend_dict = self._group_tend_list(tend_list)
        clustering_dict = dict()
        for type, tend_group_list in tend_dict.items():
            if len(tend_group_list) < 10:
                continue
            # print("Clustering for %s->%s" % (type[1].capitalize(), type[0].capitalize()))
            clusters, pref_num = self.perform_clustering(tend_group_list)
            clustering_dict[type] = {'clusters': clusters, 'pref_num': pref_num}
            # print("Number of Clusters: %s" % len(clusters))
            # for idx in range(0, len(clusters)):
            #     cluster = clusters[idx]
            #     print("Number of Elements in Cluster #%s: %s" % ((idx+1), len(cluster)))
            # print()
            # for idx in range(0, len(clusters)):
            #     cluster = clusters[idx]
            #     score_sum = 0
            #     score_cnt = 0
            #     for ta in cluster:
            #         score_sum += ta[1]
            #         score_cnt += 1
            #     if score_cnt == 0:
            #         score_cnt = 1
            #     score_avg = score_sum / score_cnt
            #     print("Cluster #%s: %s (count: %s, avg_score: %s)" %
            #           ((idx+1), self._label_for_cluster(cluster, self.words_corpora[type]),
            #            len(cluster), score_avg))
            #     for ta in cluster:
            #         print(ta)
            #     print()

        # step 5
        # print("\n##### Step 5. #####")
        pos_tendency = dict()
        neg_tendency = dict()
        for type, clustering_info in clustering_dict.items():
            pos_results, neg_results = self.figure_out_best_ta(clustering_info['clusters'],
                                                               len(preprocessed_diaries),
                                                               clustering_info['pref_num'])
            if len(pos_results) > 0:
                pos_tendency[type[0]] = pos_results
            if len(neg_results) > 0:
                neg_tendency[type[0]] = neg_results

            # print("Tendency for %s->%s" % (type[1].capitalize(), type[0].capitalize()))
            # for pos_ta in pos_results:
            #     print(_get_default_lemma(pos_ta[0]) + ': ' + \
            #           self._get_preference_class_name(pos_ta[1]) + \
            #           ' (' + str(pos_ta[1]) + ')')
            # for neg_ta in neg_result:
            #     print(_get_default_lemma(neg_ta[0]) + ': ' + \
            #           self._get_preference_class_name(neg_ta[1]) + \
            #           ' (' + str(neg_ta[1]) + ')')
            # print()

        # self._plot_result(pos_tendency, neg_tendency)

        return pos_tendency, neg_tendency

    ###############################################################################
    # Step 1. Retrieving Corpora about Things, Activities, and Preferences        #
    ###############################################################################
    def load_word_corpora(self, type_corpora):
        sw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordset',
                              'SentiWordNet_3.0.0_20130122.txt')
        senti_wordnet = SentiWordNetRetriever(sw_path)
        pv_like_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordset',
                                    'like_word_list.txt')
        pv_dislike_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordset',
                                       'dislike_word_list.txt')
        pref_verbs = PreferenceVerbCorpusRetriever(pv_like_path, pv_dislike_path,
                                                   excepts=['sleep_together.v.01', 'visualize.v.01'])
        self.senti_wordnet = senti_wordnet
        self.pref_verb_corpus = pref_verbs

        for type in type_corpora:
            word_set_corpus = None
            if type == ('food', 'thing'):
                foods_categoricals = ['cut.n.06', 'cold_cuts.n.01', 'nutriment.n.01', 'foodstuff.n.02', 'dish.n.02',
                                      'plate.n.07', 'side_dish.n.01',
                                      'course.n.07', 'game.n.07', 'halal.n.01', 'horsemeat.n.01', 'broth.n.01',
                                      'date.n.08', 'side.n.09', 'pop.n.02', 'bird.n.02', 'carbonado.n.02', ]
                foods_categoricals.extend(list(_get_hypernyms_name(wn.synset('cut.n.06'))))
                foods = HyponymCorpusRetriever(wn.synset('food.n.02'), wn.synset('food.n.01'),
                                               max_level=10,
                                               excepts=['slop.n.04', 'loaf.n.02', 'leftovers.n.01',
                                                        'convenience_food.n.01', 'concentrate.n.02',
                                                        'miraculous_food.n.01', 'micronutrient.n.01',
                                                        'feed.n.01', 'fare.n.04',
                                                        'culture_medium.n.01', 'comestible.n.01',
                                                        'comfort_food.n.01', 'commissariat.n.01',
                                                        'alcohol.n.01', 'chyme.n.01', 'meal.n.03', 'meal.n.01',
                                                        'variety_meat.n.01', 'vitamin.n.01'],
                                               categoricals=foods_categoricals)
                word_set_corpus = foods
            elif type == ('hobby', 'activity'):
                hobby_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordset',
                                          'hobbies_wiki_wordnet.txt')
                hobbies = SynsetListFileCorpusRetriever(hobby_path)
                word_set_corpus = hobbies
            elif type == ('sport', 'activity'):
                sport_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordset',
                                          'sports_wiki_wordnet.txt')
                sports = SynsetListFileCorpusRetriever(sport_path)
                word_set_corpus = sports
            self.add_word_set(type[0], type[1], word_set_corpus)

    ###############################################################################
    # Step 2. Identifying Sentences including Things and Activities in Each Diary #
    ###############################################################################
    def extract_sentences(self, diary_tags):
        identified_sent_dict = defaultdict(lambda: list())
        for sent_idx in range(0, len(diary_tags)):
            tagged_sent = diary_tags[sent_idx]

            # for last 'for' loop in word
            if tagged_sent[len(tagged_sent)-1][TAG_WORD_POS] or \
                    tagged_sent[len(tagged_sent) - 1][TAG_WORD_ROLE] :
                tagged_sent.append(('', '', '', '', ''))

            prev_word_comp_list = [] # for compound word
            for entity_idx in range(0, len(tagged_sent)):
                entity = tagged_sent[entity_idx]

                # find word as noun
                if not entity[TAG_WORD_POS] or not entity[TAG_WORD_POS].startswith('NN'):
                    # end of single or compound noun
                    if len(prev_word_comp_list) >= 1:
                        is_found = False
                        search_idx_start = 0
                        search_idx_end = len(prev_word_comp_list)
                        last_idx_start = search_idx_start
                        last_idx_end = search_idx_end
                        # find things and activities as noun
                        # search by backward
                        while True:
                            now_found = False
                            search_word_comp_list = prev_word_comp_list[search_idx_start:search_idx_end]
                            last_word = search_word_comp_list[len(search_word_comp_list)-1]
                            prev_word_idx = last_word[1]
                            plural = last_word[2]
                            # print(search_word_comp_list) ##### REMOVE
                            found_synset_list \
                                = self._find_comp_synsets_in_wordsets(search_word_comp_list, 'n', plural)
                            if found_synset_list and prev_word_idx != -1:
                                for found_synset in found_synset_list:
                                    identified_sent_dict[sent_idx].append(found_synset + (prev_word_idx, 'n',
                                                                                          len(search_word_comp_list)))
                                    # print('A: found', found_synset)  ##### REMOVE
                                is_found = True
                                now_found = True
                            else:
                                # find in common word
                                if self._check_comp_synsets_in_common(search_word_comp_list, plural):
                                    is_found = True
                                    now_found = True
                                    # print('A: found in common')  ##### REMOVE

                            # enxt check
                            if now_found:
                                search_idx_end = search_idx_start
                                search_idx_start = 0
                                last_idx_end = search_idx_end
                                if search_idx_end - search_idx_start <= 1:
                                    break
                            else:
                                # find next compound word
                                if search_idx_end - search_idx_start <= 2:  # there is no more compound thing or activity
                                    break
                                else:
                                    search_idx_start += 1

                        # search by forward
                        if (not is_found or search_idx_end - search_idx_start >= 2) and len(prev_word_comp_list) > 2:
                            if is_found:
                                search_idx_end = last_idx_end
                            else:
                                search_idx_end = len(prev_word_comp_list) - 1
                            search_idx_start = 0

                            while True:
                                now_found = False
                                search_word_comp_list = prev_word_comp_list[search_idx_start:search_idx_end]
                                last_word = search_word_comp_list[len(search_word_comp_list) - 1]
                                prev_word_idx = last_word[1]
                                plural = last_word[2]
                                # print(search_word_comp_list) ##### REMOVE
                                found_synset_list \
                                    = self._find_comp_synsets_in_wordsets(search_word_comp_list, 'n', plural)
                                if found_synset_list and prev_word_idx != -1:
                                    for found_synset in found_synset_list:
                                        identified_sent_dict[sent_idx].append(found_synset + (prev_word_idx, 'n',
                                                                              len(search_word_comp_list)))
                                        # print('B: found', found_synset) ##### REMOVE
                                    is_found = True
                                    now_found = True
                                else:
                                    # find in common word
                                    if self._check_comp_synsets_in_common(search_word_comp_list, plural):
                                        is_found = True
                                        now_found = True
                                        # print('B: found in common')  ##### REMOVE

                                # enxt check
                                if now_found:
                                    search_idx_start = search_idx_end
                                    search_idx_end = last_idx_end
                                    last_idx_start = search_idx_start
                                    if search_idx_end - search_idx_start <= 1:
                                        break
                                else:
                                    # find next compound word
                                    if search_idx_end - search_idx_start <= 2:  # there is no more compound thing or activity
                                        break
                                    else:
                                        search_idx_end -= 1

                        # search by one by one
                        if not is_found or (last_idx_end - last_idx_start >= 1):
                            search_idx = last_idx_start
                            while True:
                                search_word_comp_list = prev_word_comp_list[search_idx:search_idx+1]
                                last_word = search_word_comp_list[0]
                                prev_word_idx = last_word[1]
                                plural = last_word[2]
                                # print(search_word_comp_list) ##### REMOVE
                                found_synset_list \
                                    = self._find_comp_synsets_in_wordsets(search_word_comp_list, 'n', plural)
                                if found_synset_list and prev_word_idx != -1:
                                    for found_synset in found_synset_list:
                                        identified_sent_dict[sent_idx].append(found_synset + (prev_word_idx, 'n',
                                                                              len(search_word_comp_list)))
                                        # print('C: found', found_synset)  ##### REMOVE
                                # find next compound word
                                search_idx += 1
                                if search_idx >= last_idx_end:  # there is no thing or activity
                                    break
                        prev_word_comp_list.clear()
                        continue

                # a word have pos and role
                if entity[TAG_WORD_POS] and entity[TAG_WORD_ROLE]:
                    if 'VB' in entity[TAG_WORD_POS]:
                        # find activities as verb
                        found_synset_list \
                            = self._find_synsets_in_wordsets([entity[TAG_WORD]], pos='v')
                        for found_synset in found_synset_list:
                            identified_sent_dict[sent_idx].append(found_synset + (entity_idx, 'v', 0))
                        prev_word_comp_list.clear()

                    elif entity[TAG_WORD_POS].startswith('NN') and \
                            ('subj' in entity[TAG_WORD_ROLE] or 'obj' in entity[TAG_WORD_ROLE] or
                                     entity[TAG_WORD_ROLE] == 'conj' or entity[TAG_WORD_ROLE] == 'nmod' or
                                     entity[TAG_WORD_ROLE] == 'compound'):
                        # check the noun is plural
                        plural = False
                        if entity[TAG_WORD_POS].endswith('S'):
                            plural = True
                        prev_word_comp_list.append((entity[TAG_WORD], entity_idx, plural, 'n'))

                        ##this

                    # For the compound words with JJ
                    elif (entity[TAG_WORD_POS].startswith('JJ') and entity[TAG_WORD_ROLE] == 'amod') or \
                            (entity[TAG_WORD_POS].startswith('NN') and entity[TAG_WORD_ROLE] == 'compound'):
                        prev_word_comp_list.append((entity[TAG_WORD], -1, False, 'a'))

                else:
                    prev_word_comp_list.clear()

        return identified_sent_dict

    ###############################################################################
    # Step 3. Computing Tendency Scores of Things and Activities in Each Diary    #
    ###############################################################################
    def compute_tend_scores(self, diary_sen_list, identified_sen_dict, diary_idx=None):
        tend_score_counts = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        ta_types = dict()
        tend_scores = defaultdict(lambda: {'score': 0.0, 'type': 0})

        for sen_idx, identified_info_list in identified_sen_dict.items():
            tagged_sen = diary_sen_list[sen_idx]

            identified_words = defaultdict(lambda: {'freq': 0.0})

            adv_mod_pref = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'score': 0})
            verb_pref = defaultdict(lambda: {'sum': 0.0, 'count': 0})
            mod_pref = defaultdict(lambda: {'sum': 0.0, 'count': 0})
            sen_pref = defaultdict(lambda: {'sum': 0.0, 'count': 0})
            main_ta_dict = defaultdict(lambda: {'is_subj': False, 'is_mine': False, 'is_neg': False,
                                           'is_we': False, 'is_subj_none': True, 'ta_list': list()})

            # indices for identified word
            identified_word_dict = dict()
            identified_word_idxs = list()
            identified_comp_word_idxs = list()
            for identified_info in identified_info_list:
                # compute word scores depending on the frequency
                identified_synset = identified_info[0]
                word_count = self._count_word_in_corpus(identified_info[1],
                                                        pos=identified_info[4])
                count_sum = 0
                count_for_synset = 0
                for synset_word, count in word_count.items():
                    count_sum += count + 1
                    if identified_synset.name() == synset_word:
                        count_for_synset = count + 1
                # if not found, the weight is 0.5...?
                if count_sum == 0 or count_for_synset == 0:
                    count_sum = 2
                    count_for_synset = 1
                word_freq = count_for_synset / count_sum
                synset_name = identified_synset.name()
                identified_words[synset_name]['freq'] = word_freq \
                    if word_freq > identified_words[synset_name]['freq'] \
                    else identified_words[synset_name]['freq']
                identified_words[synset_name]['type'] = identified_info[2]
                identified_words[synset_name]['idx'] = identified_info[3]

                identified_word_dict[identified_info[3]] = identified_info
                if synset_name not in ta_types.keys():
                    ta_types[synset_name] = identified_info[2]

                # add index of identified word (start 0)
                identified_word_idxs.append(identified_info[3])
                if identified_info[5] > 1:  # num of words is more than 2
                    for i in range (1, identified_info[5]):
                        identified_comp_word_idxs.append(identified_info[3]-i)

            # identify root, conj of root, clause
            main_entities_idxs = list()
            root_idx = -1
            for entity_idx in range(0, len(tagged_sen)):
                entity = tagged_sen[entity_idx]
                if entity[TAG_WORD_ROLE] == 'root':     # root
                    main_entities_idxs.append(entity_idx)
                    root_idx = entity_idx
                    pass
                elif entity[TAG_WORD_ROLE] == 'conj':
                    # conj for root
                    if tagged_sen[int(entity[TAG_DEPENDENCY])-1][TAG_WORD_ROLE] == 'root':
                        main_entities_idxs.append(entity_idx)
                        pass
                elif entity[TAG_WORD_ROLE] == 'ccomp':
                    # clausal complement (ex, that clause)
                    # if tagged_sen[int(entity[TAG_DEPENDENCY])-1][TAG_WORD_ROLE] == 'root':
                    #     main_entities_idxs.append(entity_idx)
                    #     pass
                    main_entities_idxs.append(entity_idx)

                elif entity[TAG_WORD_ROLE] == 'advcl':
                    # adverb clause (ex, when clause)
                    # if tagged_sen[int(entity[TAG_DEPENDENCY])-1][TAG_WORD_ROLE] == 'root':
                    #     main_entities_idxs.append(entity_idx)
                    #     pass
                    main_entities_idxs.append(entity_idx)

                elif entity[TAG_WORD_ROLE] == 'parataxis':
                    # parataxis clauses
                    main_entities_idxs.append(entity_idx)

            # find words depend on which main entities
            entities_dep_dict = defaultdict(lambda: [])
            for entity_idx in range(0, len(tagged_sen)):
                if not tagged_sen[entity_idx][TAG_WORD_POS]:
                    continue
                if entity_idx in main_entities_idxs:
                    entities_dep_dict[entity_idx].append(entity_idx)
                else:
                    now_idx = entity_idx
                    while True:
                        now_entity = tagged_sen[now_idx]
                        # print(now_entity)
                        if not now_entity[TAG_DEPENDENCY]:
                            break
                        dep_idx = int(now_entity[TAG_DEPENDENCY]) - 1
                        if now_idx == dep_idx:
                            break
                        if dep_idx in main_entities_idxs:
                            entities_dep_dict[dep_idx].append(entity_idx)
                            break
                        else:
                            now_idx = dep_idx

            # if a main entity has no child dependency, it means conj of its dependent entity
            for entity_idx in main_entities_idxs:
                if entity_idx not in entities_dep_dict.keys():
                    entity = tagged_sen[entity_idx]
                    dep_idx = int(entity[TAG_DEPENDENCY]) - 1
                    if dep_idx in entities_dep_dict.keys():
                        entities_dep_dict[dep_idx].append(entity_idx)
                    else:
                        entities_dep_dict[root_idx].append(entity_idx)

            # identify default thing/activity in the tokens
            for main_idx, entity_idxs in entities_dep_dict.items():
                for entity_idx in entity_idxs:
                    entity = tagged_sen[entity_idx]

                    # if already identified word (thing or activity)
                    if entity_idx in identified_comp_word_idxs: # for comp words
                        continue
                    elif entity_idx in identified_word_idxs:
                        if 'subj' in entity[TAG_WORD_ROLE]:
                            main_ta_dict[main_idx]['is_subj'] = True
                        main_ta_dict[main_idx]['ta_list'].append(entity_idx)
                        continue

                    if not entity[TAG_WORD_POS] or not entity[TAG_WORD_ROLE]:
                        continue

                    if 'subj' in entity[TAG_WORD_ROLE]:
                        if entity[TAG_WORD_POS] == 'PRP':
                            if 'I' == entity[TAG_WORD].upper():
                                main_ta_dict[main_idx]['is_mine'] = True
                            elif 'we' == entity[TAG_WORD].lower():
                                main_ta_dict[main_idx]['is_we'] = True
                        main_ta_dict[main_idx]['is_subj_none'] = False  # no subj -> may be subj is I
                        continue

                    elif 'neg' in entity[TAG_WORD_ROLE]:
                        main_ta_dict[main_idx]['is_neg'] = True
                        continue

            # calculate the adverb pref first
            for main_idx, entity_idxs in entities_dep_dict.items():
                for entity_idx in entity_idxs:
                    entity = tagged_sen[entity_idx]

                    # check adverb modifier
                    if entity[TAG_WORD_POS].startswith('RB') and entity[TAG_WORD_ROLE] != 'neg':
                        offsets = _find_offsets_from_word(entity[TAG_WORD].lower(), 'r')
                        pref_score = 0
                        pref_cnt = 0
                        for offset in offsets:
                            senti_score_dict = self.senti_wordnet.get_score(offset, 'r')
                            senti_score = senti_score_dict['positivity'] - senti_score_dict['negativity']
                            senti_count = senti_score_dict['corpus_count'] + 1
                            if senti_score == 0:
                                continue
                            else:
                                pref_score += (senti_score * senti_count)
                                pref_cnt += senti_count
                        if pref_cnt == 0:
                            pref_score = 0
                        else:
                            pref_score = pref_score / pref_cnt

                        if self.DEBUG:
                            print(entity, pref_score)

                        if pref_score != 0:
                            dep_idx = int(entity[TAG_DEPENDENCY]) - 1
                            if dep_idx == main_idx:
                                sen_pref[main_idx]['sum'] += pref_score
                                sen_pref[main_idx]['count'] += 1
                            else:
                                adv_mod_pref[dep_idx]['sum'] += pref_score
                                adv_mod_pref[dep_idx]['count'] += 1
            for entity_idx in adv_mod_pref.keys():
                adv_mod_pref[entity_idx]['score'] = adv_mod_pref[entity_idx]['sum'] / adv_mod_pref[entity_idx]['count']

            # calculate the tendency score for words
            for main_idx, entity_idxs in entities_dep_dict.items():
                # print(tagged_sen)
                # print(sen_idx, main_idx, entity_idxs)
                for entity_idx in entity_idxs:
                    entity = tagged_sen[entity_idx]

                    # check verb
                    if 'VB' in entity[TAG_WORD_POS]:
                        pref_score = 0
                        pref_cnt = 0
                        need_find_in_senti = True

                        if self.pref_verb_corpus:
                            lemma_form = _text_to_lemma_format(entity[TAG_WORD])
                            pref_score = self.pref_verb_corpus.get_score(lemma_form)
                            if pref_score != 0:
                                pref_cnt = 1
                                need_find_in_senti = False
                                # print('########', entity, pref_score, '########')

                        if need_find_in_senti:
                            offsets = _find_offsets_from_word(entity[TAG_WORD].lower(), 'v')

                            for offset in offsets:
                                senti_score_dict = self.senti_wordnet.get_score(offset, 'v')
                                senti_score = senti_score_dict['positivity'] - senti_score_dict['negativity']
                                senti_count = senti_score_dict['corpus_count'] + 1
                                if senti_score == 0:
                                    continue
                                else:
                                    pref_score += (senti_score * senti_count)
                                    pref_cnt += senti_count
                            if pref_cnt == 0:
                                pref_score = 0
                            else:
                                pref_score = pref_score / pref_cnt

                        if self.DEBUG:
                            print(entity, need_find_in_senti, pref_score)

                        if pref_score != 0:
                            if entity_idx == main_idx:
                                verb_pref[main_idx]['sum'] += pref_score + adv_mod_pref[entity_idx]['score']
                                verb_pref[main_idx]['count'] += 1
                            else:
                                dep_idx = int(entity[TAG_DEPENDENCY]) - 1
                                if dep_idx == main_idx and dep_idx in entity_idxs:
                                    #entity idx? or dep idx?
                                    verb_pref[main_idx]['sum'] += pref_score + adv_mod_pref[entity_idx]['score']
                                    verb_pref[main_idx]['count'] += 1
                                else:   # how i handle..?
                                    if tagged_sen[entity_idx][TAG_WORD_ROLE] == 'acl:relcl':    # 관계대명사절
                                        if dep_idx in main_ta_dict[main_idx]['ta_list']:
                                            mod_pref[dep_idx]['sum'] += pref_score + adv_mod_pref[entity_idx]['score']
                                            mod_pref[dep_idx]['count'] += 1
                                    else:
                                        # SHOULD ENHANCE
                                        if dep_idx == main_idx:
                                            verb_pref[main_idx]['sum'] += pref_score + adv_mod_pref[entity_idx]['score']
                                            verb_pref[main_idx]['count'] += 1
                                        else:
                                            # print('VERB...?', main_idx, entity_idx, dep_idx)
                                            pass

                    # check adjective modifier
                    elif entity[TAG_WORD_POS].startswith('JJ'):
                        word = entity[TAG_WORD].lower()

                        # exceptional cases
                        if word == 'few' or word == 'little':
                            # a little and a few
                            if entity_idx >= 1 and tagged_sen[entity_idx-1][TAG_WORD] == 'a':
                                continue

                        offsets = _find_offsets_from_word(word, 'a')
                        pref_score = 0
                        pref_cnt = 0
                        for offset in offsets:
                            senti_score_dict = self.senti_wordnet.get_score(offset, 'a')
                            senti_score = senti_score_dict['positivity'] - senti_score_dict['negativity']
                            senti_count = senti_score_dict['corpus_count'] + 1
                            if senti_score == 0:
                                continue
                            else:
                                pref_score += (senti_score * senti_count)
                                pref_cnt += senti_count
                        if pref_cnt == 0:
                            pref_score = 0
                        else:
                            pref_score = pref_score / pref_cnt

                        if self.DEBUG:
                            print(entity, pref_score)

                        if pref_score != 0:
                            if entity_idx == main_idx:
                                for ta_entity_idx in main_ta_dict[main_idx]['ta_list']:
                                    mod_pref[ta_entity_idx]['sum'] += pref_score + adv_mod_pref[entity_idx]['score']
                                    mod_pref[ta_entity_idx]['count'] += 1
                            else:
                                dep_idx = int(entity[TAG_DEPENDENCY]) - 1
                                if dep_idx in main_ta_dict[main_idx]['ta_list']:
                                    mod_pref[dep_idx]['sum'] += pref_score + adv_mod_pref[entity_idx]['score']
                                    mod_pref[dep_idx]['count'] += 1
                                else:   # how i handle..?
                                    # print('ADJECTIVE...?', main_idx, entity_idx, dep_idx)
                                    pass

            if self.DEBUG:
                print('dep_dict')
                for entity_idx, dep_list in entities_dep_dict.items():
                    print(str(entity_idx)+':', dep_list)
                print('main_ta_dict')
                pprint(main_ta_dict)
                print('verb_pref')
                pprint(verb_pref)
                print('mod_pref')
                pprint(mod_pref)
                print('sen_pref')
                pprint(sen_pref)
                print()

            # compute tend score for sentence
            for main_idx, ta_dict in main_ta_dict.items():
                # calculate weight of subjectivity
                weight_subj = 1 if (ta_dict['is_mine'] or ta_dict['is_we'] or ta_dict['is_subj_none']) \
                    else (0.8 if ta_dict['is_subj'] else 0)
                if weight_subj == 0:
                    continue
                weight_subj *= -1 if ta_dict['is_neg'] else 1

                # calculate tend score
                for ta_entity_idx in ta_dict['ta_list']:
                    avg_pref = 0
                    if verb_pref[main_idx]['count'] + mod_pref[ta_entity_idx]['count'] != 0:
                        avg_pref = (verb_pref[main_idx]['sum'] + mod_pref[ta_entity_idx]['sum']) / \
                                   (verb_pref[main_idx]['count'] + mod_pref[ta_entity_idx]['count'])
                    # avg tend score
                    # if 0 <= avg_pref < 0.15:
                    #     avg_pref = 0.15
                    if sen_pref[main_idx]['count'] == 0:
                        sen_tend_score = weight_subj * avg_pref
                    else:
                        sen_tend_score = weight_subj * (avg_pref + sen_pref[main_idx]['sum'] / sen_pref[main_idx]['count'])

                    # synset_name = entity[0].name()
                    synset_name = identified_word_dict[ta_entity_idx][0].name()
                    tend_score_counts[synset_name]['sum'] += sen_tend_score
                    tend_score_counts[synset_name]['count'] += 1

        # compute tendency scores for diary
        for synset_name, count_dict in tend_score_counts.items():
            tend_scores[synset_name]['score'] = count_dict['sum'] / count_dict['count'] * \
                                                (1 + ((count_dict['count']-1) / 11))    # 1+(count-1/11) weight
            if tend_scores[synset_name]['score'] > 1:
                tend_scores[synset_name]['score'] = 1
            elif tend_scores[synset_name]['score'] < -1:
                tend_scores[synset_name]['score'] = -1
            tend_scores[synset_name]['type'] = ta_types[synset_name]
            tend_scores[synset_name]['diary_idx'] = diary_idx
        return tend_scores

    ###############################################################################
    # Step 4. Clustering the Things and Activities                                #
    ###############################################################################
    def perform_clustering(self, pref_ta_list):
        def calc_distance(u, v):
            synset_u = wn.synset(u[0])
            synset_v = wn.synset(v[0])

            if synset_u.pos() != synset_v.pos():
                if synset_u.pos() != 'n':
                    synset_u_rel_list = _nounify(synset_u)
                    if len(synset_u_rel_list) > 0:
                        synset_u = synset_u_rel_list[0]
                elif synset_v.pos() != 'n':
                    synset_v_rel_list = _nounify(synset_v)
                    if len(synset_v_rel_list) > 0:
                        synset_v = synset_v_rel_list[0]

            distance = _calc_path_distance(synset_u, synset_v)

            # if distance is None or distance < 0:
            #     distance
            return distance

        # sort list for debugging
        if self.DEBUG:
            pref_ta_list = sorted(pref_ta_list, key=lambda pref_ta: pref_ta[0])

        # print list for debug
        if self.DEBUG:
            print("pref_ta_list(all): ")
            for ta in pref_ta_list:
                print(ta)
            print()

        # feature extraction
        # filtering with things and activities which there is only one item (with low score)
        # count_dict = defaultdict(int)
        # for pref_ta in pref_ta_list:
        #     count_dict[pref_ta[0]] += 1
        # for i in range(0, len(pref_ta_list))[::-1]:
        #     if count_dict[pref_ta_list[i][0]] < 2:
        #     # if count_dict[pref_ta_list[i][0]] < 2 and abs(pref_ta_list[i][1]) < 0.2:
        #         pref_ta_list.pop(i)

        # filtering with things and activities which has more than 0.01 preference score
        for i in range(0, len(pref_ta_list))[::-1]:
            if abs(pref_ta_list[i][1]) < 0.01:
                pref_ta_list.pop(i)

        # hypernyms to hyponym extraction
        pref_ta_features = list()
        check_hyper_set = set()  # check already lower
        for i in range(0, len(pref_ta_list)):
            pref_ta = pref_ta_list[i]
            hypo_pref_set = set()
            for j in range(0, len(pref_ta_list)):
                if i == j:
                    continue
                pref_ta2 = pref_ta_list[j]
                if _is_inherit_hypernym_of(wn.synset(pref_ta2[0]), wn.synset(pref_ta[0])):
                    hypo_pref_set.add(pref_ta2[0])
            # if ta is hypernym of others
            if len(hypo_pref_set) > 0:
                for pref_hypo_name in hypo_pref_set:
                    pref_feature = [pref_hypo_name, pref_ta[1], pref_ta[2], pref_ta[3], pref_ta[4], pref_ta[0]]
                    pref_ta_features.append(pref_feature)
                    check_hyper_set.add(pref_ta[0])
            else:
                pref_feature = list(pref_ta) + [None]
                pref_ta_features.append(pref_feature)
        for i in range(0, len(pref_ta_features))[::-1]:
            pref_ta = pref_ta_features[i]
            if pref_ta[0] in check_hyper_set:
                pref_ta_features.pop(i)

        # for i in range(0, len(pref_ta_list)):
        #     pref_ta_features.append(list(pref_ta_list[i]))
        pref_num = len(pref_ta_features)

        # print list for debug
        if self.DEBUG:
            print("pref_ta_list(features): ")
            for ta in pref_ta_features:
                print(ta)
            print()

        pref_len = len(pref_ta_features)
        if pref_len < 2:
            return [], 0

        # make distance matrix
        dist_matrix = np.array([list(10.0 for i in range(pref_len)) for j in range(pref_len)], np.float32)
        for u in range (0, pref_len):
            for v in range(u, pref_len):
                dist = calc_distance(pref_ta_features[u], pref_ta_features[v])
                dist_matrix[u][v] = dist
                dist_matrix[v][u] = dist

        if self.DEBUG:
            print("distance matrix: ")
            for y in dist_matrix:
                print(y)
            print()

        # perform clustering
        hac_result = hac.linkage(dist_matrix, method='single')
        # hac_result = hac.linkage(dist_matrix, method='complete')
        if self.DEBUG:
            print("linkage result: ")
            for y in hac_result:
                print(y)
            print()

        # figure out the number of clusters (determine where to cut tree)
        num_cluster = 1
        for matrix_y in hac_result:
            if matrix_y[2] > self.CLUSTER_CUT_DIST:
            # if matrix_y[2] > 0.25:
                num_cluster += 1
        if self.DEBUG:
            print("num_cluster: ", num_cluster, '\n')

        # part_cluster = hac.fcluster(hac_result, num_cluster, 'maxclust')
        part_cluster = hac.fcluster(hac_result, self.CLUSTER_CUT_DIST, 'distance')
        if self.DEBUG:
            print("part_cluster: ")
            print(part_cluster)
            print()

        # batch each thing and activity to its cluster
        clusters = [[] for i in range(num_cluster)]
        for idx_ta in range(0, len(part_cluster)):
            cluster_id = part_cluster[idx_ta] - 1
            clusters[cluster_id].append(pref_ta_features[idx_ta])
        if self.DEBUG:
            print("clusters:")
            pprint(clusters)
            print()

        # show dendrogram
        # labels = list('' for i in range(pref_len))
        # for i in range(pref_len):
        #     # labels[i] = str(i) + ' (' + str(part_cluster[i]) + ')'
        #     # labels[i] = str(i)
        #     labels[i] = '[' + str(part_cluster[i]) + '] ' + pref_ta_features[i][0] + '(' + str(i) + ')\n' + \
        #                 str(int(pref_ta_features[i][1] * 100000) / 100000.0)
        #     # labels[i] = _get_default_lemma(pref_ta_list[i][0]) + '\n' + \
        #     #             str(int(pref_ta_list[i][1] * 1000) / 1000.0)
        # ct = hac_result[-(num_cluster - 1), 2]
        # p = hac.dendrogram(hac_result, labels=labels, color_threshold=ct)
        # plt.show()

        return clusters, pref_num

    ###############################################################################
    # Step 5. Figuring out Things and Activities having the Best Preference Score #
    ###############################################################################
    def figure_out_best_ta(self, clusters, diary_num, pref_num):
        pos_ta_score_dict = dict()
        neg_ta_score_dict = dict()

        min_cluster_item = diary_num / 30.0
        for cluster in clusters:
            # if the number of the items in cluster is a few
            # if len(cluster) < min_cluster_item:
            #     continue

            target = cluster[0][2]
            target_type = cluster[0][3]

            # for i in range(0, len(cluster)):
            #     cluster[i] = list(cluster[i])

            # add weight for things or activities which are more count than others
            # weight formula can be changed
            # now: weight = 1 + log_max_count(count)
            count_dict = defaultdict(lambda: {'count': 0, 'sum': 0.0})  # dict for counting
            for ta in cluster:
                count_dict[ta[0]]['count'] += 1
                count_dict[ta[0]]['sum'] += ta[1]
                # for hypernyms
                # if ta[5] is not None:
                #     count_dict[ta[5]]['count'] += 1
                #     count_dict[ta[5]]['sum'] += ta[1]
            count_weight_dict = dict()
            for ta_name, count in count_dict.items():
                if count['count'] >= 2:
                    # average * count_weight
                    count_weight_dict[ta_name] = (count['sum'] / count['count']) * \
                                           (1 + log(count['count']-1, int(pref_num/2)+1))
                else:
                    count_weight_dict[ta_name] = (count['sum'] / count['count'])

            # remove same items and set new pref score multiplied weight
            already_exist_ta_list = list()
            for i in range(0, len(cluster))[::-1]:
                ta = cluster[i]
                # if ta[5] in already_exist_ta_list:
                #     pass
                # elif ta[5] in count_weight_dict.keys():
                #     cluster.append([ta[5], count_weight_dict[ta[5]], ta[2], ta[3], ta[4], None])
                #     already_exist_ta_list.append(ta[5])

                if ta[0] in already_exist_ta_list:
                    cluster.pop(i)
                    continue
                if ta[0] in count_weight_dict.keys():
                    # ext_ta = cluster.pop(i)
                    # new_ta = (ext_ta[0], weight_dict[ta[0]], ext_ta[2], ext_ta[3])
                    # cluster.insert(i, new_ta)
                    ta[1] = count_weight_dict[ta[0]]
                    already_exist_ta_list.append(ta[0])

            if self.DEBUG:
                print("Cluster after applying score for count")
                print(cluster)
                print()

            # plus scores to each other in same cluster (cause they are related)
            cluster_len = len(cluster)
            for i in range(0, cluster_len):
                for j in range(0, cluster_len):
                    if i != j:
                        cluster[i][1] += wn.synset(cluster[i][0]).path_similarity(wn.synset(cluster[j][0])) * \
                            cluster[j][1] / cluster_len
            if self.DEBUG:
                print("Cluster after applying score for each other")
                print(cluster)
                print()

            # find common hypernym
            synset_name_list = list()
            for ta in cluster:
                synset_name_list.append(ta[0])
            common_hypernyms = self._find_common_hypernyms(synset_name_list,
                                                           self.words_corpora[(target, target_type)])
            if self.DEBUG:
                print("common_hypernyms: ")
                print(common_hypernyms)
                print()

            # classify pos and neg
            pos_ta_list = list()
            neg_ta_list = list()
            for ta in cluster:
                if ta[1] >= 0:
                    pos_ta_list.append(ta)
                else:
                    neg_ta_list.append(ta)
            pos_ta_list = sorted(pos_ta_list, key=lambda ta: ta[1], reverse=True)
            neg_ta_list = sorted(neg_ta_list, key=lambda ta: ta[1])
            if self.DEBUG:
                print(pos_ta_list)
                print(neg_ta_list)
                print()

            if len(pos_ta_list) > 0:
                if pos_ta_list[0][0] in common_hypernyms: # if max scored ta is common hypernym
                    pos_ta_score_dict[pos_ta_list[0][0]] = pos_ta_list[0][1]
                else:
                    for k in range(0, int(len(pos_ta_list)/2)+1):
                        pos_ta = pos_ta_list[k]
                        if pos_ta[0] in common_hypernyms:
                            continue
                        if pos_ta[0] in pos_ta_score_dict.keys():
                            if pos_ta[1] > pos_ta_score_dict[pos_ta[0]]:
                                # set higher score
                                pos_ta_score_dict[pos_ta[0]] = pos_ta[1]
                        else:
                            pos_ta_score_dict[pos_ta[0]] = pos_ta[1]

            if len(neg_ta_list) > 0:
                if neg_ta_list[0][0] in common_hypernyms:
                    neg_ta_score_dict[neg_ta_list[0][0]] = neg_ta_list[0][1]
                else:
                    for k in range(0, int(len(neg_ta_list)/2)+1):
                        neg_ta = neg_ta_list[k]
                        if neg_ta[0] in common_hypernyms:
                            continue
                        if neg_ta[0] in neg_ta_score_dict.keys():
                            if neg_ta[1] > neg_ta_score_dict[neg_ta[0]]:
                                # set higher score
                                neg_ta_score_dict[neg_ta[0]] = neg_ta[1]
                        else:
                            neg_ta_score_dict[neg_ta[0]] = neg_ta[1]

        if self.DEBUG:
            print("figuring out things/activities results:")
            pprint(pos_ta_score_dict)
            pprint(neg_ta_score_dict)
            print()

        # it a thing or activity is in both side, to correct it.
        pos_ta_cluster_keys = list(pos_ta_score_dict.keys())
        for idx in range(0, len(pos_ta_cluster_keys))[::-1]:
            ta_name = pos_ta_cluster_keys[idx]
            if ta_name in neg_ta_score_dict.keys(): #if both
                ta_score = pos_ta_score_dict[ta_name] + neg_ta_score_dict[ta_name]
                if ta_score == 0:
                    del pos_ta_score_dict[ta_name]
                    del neg_ta_score_dict[ta_name]
                elif ta_score < 0:
                    del pos_ta_score_dict[ta_name]
                    neg_ta_score_dict[ta_name] = ta_score
                else:   # elif ta_score > 0:
                    del neg_ta_score_dict[ta_name]
                    pos_ta_score_dict[ta_name] = ta_score

        # scores list filter
        pos_scores_ta = list()
        for ta_name, score in pos_ta_score_dict.items():
            if score >= 0.2:   # preference score is more than
                pos_scores_ta.append((wn.synset(ta_name).lemmas()[0].name(), score))
        neg_scores_ta = list()
        for ta_name, score in neg_ta_score_dict.items():
            if score <= -0.2:
                neg_scores_ta.append((wn.synset(ta_name).lemmas()[0].name(), score))

        # best tend things and activities
        best_pos_scores = sorted(pos_scores_ta, key=lambda ta: ta[1], reverse=True)[:5]
        best_neg_scores = sorted(neg_scores_ta, key=lambda ta: ta[1])[:5]
        return best_pos_scores, best_neg_scores

    def _plot_result(self, pos_tendency, neg_tendency):
        color_list = ['r', 'b', 'g', 'y', 'c', 'm']
        type_set = set(pos_tendency.keys()) | set(neg_tendency.keys())

        data_x = []
        data_y = []
        label_y = []
        labels = []
        colors = []
        ticks = []
        ticklables = []
        type_cnt = 0
        for type in type_set:
            label_y_dist = 20
            if type in pos_tendency.keys():
                tend_list = pos_tendency[type]
                for tend in tend_list:
                    data_x.append(tend[1])
                    data_y.append(type_cnt+1)
                    labels.append('%s (%.3f)' % (tend[0], tend[1]))
                    label_y.append(label_y_dist)
                    label_y_dist += 20
                    colors.append(color_list[type_cnt])
            label_y_dist = 20
            if type in neg_tendency.keys():
                tend_list = neg_tendency[type]
                for tend in tend_list:
                    data_x.append(tend[1])
                    data_y.append(type_cnt+1)
                    labels.append('%s (%.3f)' % (tend[0], tend[1]))
                    label_y.append(label_y_dist)
                    label_y_dist += 20
                    colors.append(color_list[type_cnt])
            ticks.append(type_cnt+1)
            ticklables.append('{0}'.format(type[0]))
            # ticklables.append('{0}->{1}'.format(type[1],type[0]))
            type_cnt += 1

        plt.scatter(data_x, data_y, marker='o', c=colors, s=200)

        for label, ytext, x, y in zip(labels, label_y, data_x, data_y):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, ytext),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.axis([-1.1, 1.1, 0, len(type_set)+1])
        plt.xlabel('Tendency Score')
        # plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_ticks(ticks)
        plt.gca().axes.get_yaxis().set_ticklabels(ticklables)
        plt.gca().axes.get_xaxis().set_ticks([-1, 0, 1])
        plt.gca().axes.get_xaxis().set_ticklabels(['Worst\n(Disliked the most)', 'Neutural', 'Best\n(Liked the most)'])
        plt.show()
        pass

    # return a list of (found_synset, lemma_word, words_set_type)
    def _find_synsets_in_wordsets(self, finding_word_list, pos='n', plural=False):
        synset_list = list()
        for words_set_type, words_set in self.words_corpora.items():
            found_synset, lemma_word = \
                self._find_synset_by_word_list(words_set, finding_word_list, pos, plural)
            if found_synset:
                synset_list.append((found_synset, lemma_word, words_set_type))
        return synset_list

    def _find_comp_synsets_in_wordsets(self, finding_comp_word_list, pos='n', plural=False):
        finding_word_list = list()
        for finding_comp_word in finding_comp_word_list:
            finding_word_list.append(finding_comp_word[0])
        return self._find_synsets_in_wordsets(finding_word_list, pos, plural)

    def _check_synsets_in_common(self, finding_word_list, plural=False):
        lemma_word = _word_list_to_lemma_form(finding_word_list)
        synsets = wn.synsets(lemma_word)
        if synsets:
            return True
        else:
            length = len(finding_word_list)
            if plural and length >= 2:
                try:
                    # plural to singular
                    morphys = wn._morphy(finding_word_list[length - 1], wn.NOUN)
                    plural_noun = morphys[len(morphys) - 1]
                    finding_word_list[length - 1] = plural_noun
                    return self._check_synsets_in_common(finding_word_list, False)
                except Exception as e:  # list index out of range exception -> no matching synset
                    return False
        return False

    def _check_comp_synsets_in_common(self, finding_comp_word_list, plural=False):
        finding_word_list = list()
        for finding_comp_word in finding_comp_word_list:
            finding_word_list.append(finding_comp_word[0])
        return self._check_synsets_in_common(finding_word_list, plural)

    @classmethod
    def _score_to_hyponyms(cls, hypernym_synset, hypernym_score, score_sentiments, continue_hyponyms=False):
        hyponyms = hypernym_synset.hyponyms()
        length = len(hyponyms)
        if hypernym_score == 0 or length == 0:
            return
        subscore = hypernym_score / length
        for hyponym in hyponyms:
            synset_name = hyponym.name()
            score_sentiments[synset_name] += subscore
            if continue_hyponyms:
                cls._score_to_hyponyms(hyponym, subscore, score_sentiments)

    @classmethod
    def _find_synset_by_word_list(cls, words_set, finding_word_list, pos='n', plural=False):
        length = len(finding_word_list)
        lemma_word = _word_list_to_lemma_form(finding_word_list).lower()
        synset = words_set.find_word(lemma_word, pos=pos)

        if synset is not None:
            return synset, lemma_word

        elif synset is None and pos == 'v':
            # find original form of the verb
            synsets = wn.synsets(lemma_word, pos='v')
            for s in synsets:
                synset = words_set.find_synset(s)
                if synset is not None:
                    return synset, lemma_word

        # if lemma is not found, but the noun in lemma is plural
        # if plural and length >= 2:
        if plural:
            try:
                # plural_noun = wn.synsets(finding_word_list[length - 1])[0].lemmas()[0].name()
                morphys = wn._morphy(finding_word_list[length - 1], wn.NOUN)
                plural_noun = morphys[len(morphys)-1]
                finding_word_list[length - 1] = plural_noun
                return cls._find_synset_by_word_list(words_set, finding_word_list, pos, False)
            except Exception as e:  # list index out of range exception -> no matching synset
                return None, None
        return None, None

    @classmethod
    def _count_word_in_corpus(cls, word, pos=None):
        word_count = dict()
        for synset in wn.synsets(word):
            if pos is not None and pos != synset.pos():
                continue
            for lemma in synset.lemmas():
                if word == lemma.name():
                    # word_count.append((synset.name(), lemma.count()))
                    word_count[synset.name()] = lemma.count()
                    break
        return word_count

    @classmethod
    def _calc_min_sim_cluster(cls, leaf_ids, data_list, dist_func):
        leaf_count = len(leaf_ids)
        if leaf_count < 2:
            return 0
        min_dist = 100
        for u in range(0, leaf_count-1):
            u_id = leaf_ids[u]
            for v in range(u+1, leaf_count):
                v_id = leaf_ids[v]
                dist = dist_func(data_list[u_id], data_list[v_id])
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    @classmethod
    def convert_scores_dict_to_list(cls, scores_pref):
        pref_ta_list = list()
        for synset_name, score_dict in scores_pref.items():
            if 'diary_idx' in score_dict:
                pref_ta = (synset_name, score_dict['score'],
                           score_dict['type'][0], score_dict['type'][1],
                           score_dict['diary_idx'])
            else:
                pref_ta = (synset_name, score_dict['score'],
                           score_dict['type'][0], score_dict['type'][1])
            pref_ta_list.append(pref_ta)
        return pref_ta_list

    @classmethod
    def _group_tend_list(cls, tend_list):
        tend_dict = defaultdict(lambda: list())
        for tend_item in tend_list:
            tend_dict[(tend_item[2], tend_item[3])].append(tend_item)
        return tend_dict

    @classmethod
    def _find_common_hypernyms(cls, ta_name_list, corpus_retriever, search_level=3, check_cat=False):
        if len(ta_name_list) == 0:
            return []
        is_all_same = True
        first_ta_name = ta_name_list[0]
        for ta_name in ta_name_list:
            if first_ta_name != ta_name:
                is_all_same = False
                break
        if is_all_same:
            return [first_ta_name]

        hypernym_count_dict = defaultdict(lambda: {'count': 0, 'level': -1})
        for ta_name in ta_name_list:
            synset = wn.synset(ta_name)
            hypernym_item = corpus_retriever.get_item_synset(synset)
            # add self
            hypernym_count_dict[ta_name]['count'] += 1
            hypernym_count_dict[ta_name]['level'] = hypernym_item[WordSetCorpusRetriever.IDX_LEVEL]

            # add hypernyms
            if synset:
                hypernyms = _get_hypernyms(synset, search_level)
                for hypernym in hypernyms:
                    hypernym_item = corpus_retriever.get_item_synset(hypernym, check_cat=check_cat)
                    if hypernym_item:
                        hypernym_count_dict[hypernym.name()]['count'] += 1
                        hypernym_count_dict[hypernym.name()]['level'] \
                            = hypernym_item[WordSetCorpusRetriever.IDX_LEVEL]
        # pprint(hypernym_count_dict)

        # hypernym_count_dict_common = dict()
        # for common_synset_name in hypernym_count_dict.keys():
        #     if common_synset_name in ta_name_list:
        #         hypernym_count_dict_common[common_synset_name] = hypernym_count_dict[common_synset_name]
        #
        # if len(hypernym_count_dict_common):
        #     hypernym_count_dict = hypernym_count_dict_common
        #
        # print(hypernym_count_dict)

        max_hypernym_list = list()
        max_hypernym_count = 0
        max_hypernym_level = -1
        for hypernym_name, count_dict in hypernym_count_dict.items():
            if max_hypernym_count < count_dict['count']:
                max_hypernym_list.clear()
                max_hypernym_count = count_dict['count']
                max_hypernym_level = count_dict['level']
                max_hypernym_list.append(hypernym_name)
            elif max_hypernym_count == count_dict['count']:
                if max_hypernym_level < count_dict['level']:
                    max_hypernym_list.clear()
                    max_hypernym_level = count_dict['level']
                    max_hypernym_list.append(hypernym_name)
                elif max_hypernym_level == count_dict['level']:
                    max_hypernym_list.append(hypernym_name)
        # pprint(max_hypernym_list)
        if max_hypernym_count >= len(ta_name_list):
            max_hypernym_list_in_origin = list()
            for hypernym_name in max_hypernym_list:
                if hypernym_name in ta_name_list:
                    max_hypernym_list_in_origin.append(hypernym_name)
            if len(max_hypernym_list_in_origin) > 0:
                return max_hypernym_list_in_origin
            else:
                return max_hypernym_list
        return []

    @classmethod
    def _label_for_cluster(cls, cluster, corpus_retriever=None):
        rep_ta_name = ''

        synset_name_list = list()
        for ta in cluster:
            synset_name_list.append(ta[0])
        common_hypernyms = cls._find_common_hypernyms(synset_name_list,
                                                      corpus_retriever,
                                                      check_cat=True)
        if common_hypernyms:
            for idx in range(0, len(common_hypernyms)):
                hypernym_name = common_hypernyms[idx]
                if idx > 0:
                    rep_ta_name += ', '
                rep_ta_name += _get_default_lemma(hypernym_name)
        else:
            sum_score = 0
            cnt_score = 0

            count_item_dict = defaultdict(int)
            for ta in cluster:
                count_item_dict[ta[0]] += 1
                sum_score += ta[1]
                cnt_score += 1
            if cnt_score == 0:
                cnt_score = 1

            max_ta_count = 0
            max_ta_list = list()
            for ta_name, count in count_item_dict.items():
                if max_ta_count < count:
                    max_ta_list.clear()
                    max_ta_count = count
                    max_ta_list.append(ta_name)
                elif max_ta_count == count:
                    max_ta_list.append(ta_name)

            for idx in range(0, len(max_ta_list)):
                ta_name = max_ta_list[idx]
                if idx > 0:
                    rep_ta_name += ', '
                rep_ta_name = _get_default_lemma(ta_name)
        return rep_ta_name

    @classmethod
    def _get_preference_class_name(cls, score):
        if score == 1:
            return "Absolutely Like"
        elif score >= 0.8:
            return "Very Like"
        elif score >= 0.6:
            return "More Like"
        elif score >= 0.4:
            return "Like"
        elif score >= 0.2:
            return "Slightly Like"
        elif score == -1:
            return "Absolutely Dislike"
        elif score <= -0.8:
            return "Very Dislike"
        elif score <= -0.6:
            return "More Dislike"
        elif score <= -0.4:
            return "Dislike"
        elif score <= -0.2:
            return "Slightly Dislike"
        else:
            return ""


def _lemmas_to_name_list(lemmas, include_count=False):
    names = list()
    if include_count:
        counts = list()
        for lemma in lemmas:
            names.append(lemma.name())
            counts.append(lemma.count())
        return names, counts
    else:
        for lemma in lemmas:
            names.append(lemma.name())
        return names


def _get_synsets(word, pos_list=('n', 'v', 'a', 's', 'r')):
    # s means adjective satellite (a word that describes a person, animal, place, thing or idea)
    # r meas adverb
    synsets = wn.synsets(word)
    synsets_filtered = list()
    for synset in synsets:
        if synset.pos() in pos_list:
            synsets_filtered.append(synset)
    return synsets_filtered


def _find_original_form(word, pos):
    for synset in wn.synsets(word):
        if synset.pos() == pos:
            return synset.lemmas()[0].name()
    return word


def _find_offsets_from_word(word, pos):
    offsets = []
    for synset in wn.synsets(word):
        if synset.pos() == pos or (pos == 'a' and synset.pos() == 's'):
            offsets.append(synset.offset())
    return offsets


def _text_to_lemma_format(text):
    lemma_form = ""
    text = text.lower()
    texts = text.split(' ')
    for i in range(0, len(texts)):
        if i > 0:
            lemma_form += '_'
        lemma_form += texts[i]
    return lemma_form


def _get_hypernyms(synset, level=99999):
    hypernym_set = set()
    for hypernym in synset.hypernyms():
        hypernym_set.add(hypernym)
        if level > 1:
            hypernym_set |= _get_hypernyms(hypernym, level-1)
    return hypernym_set


def _get_hypernyms_name(synset, level=99999):
    hypernym_name_set = set()
    for hypernym in synset.hypernyms():
        hypernym_name_set.add(hypernym.name())
        if level > 1:
            hypernym_name_set |= _get_hypernyms_name(hypernym, level-1)
    return hypernym_name_set


def _get_default_lemma(synset_name):
    synset = wn.synset(synset_name)
    if synset:
        lemmas = synset.lemmas()
        if lemmas:
            return lemmas[0].name()
    return None


def _get_hypernym_lemmas(synset_name=None, word=None, level=5):
    lemma_list = []
    if synset_name:
        synset = wn.synset(synset_name)
        if synset:
            hypernyms = _get_hypernyms(synset, level)
            for hypernym in hypernyms:
                lemma_list.extend(hypernym.lemmas())
        return lemma_list
    elif word:
        for synset in wn.synsets(word):
            lemma_list = []
            if synset:
                hypernyms = _get_hypernyms(synset, level)
                for hypernym in hypernyms:
                    lemma_list.extend(hypernym.lemmas())
    return lemma_list


def _word_list_to_lemma_form(word_list):
    length = len(word_list)
    lemma_word = ''
    for i in range(0, length):
        if i > 0:
            lemma_word += '_'
        lemma_word += word_list[i]
    return lemma_word


def _is_inherit_hypernym_of(hyponym, hypernym):
    r_hypernym_names = _get_hypernyms_name(hyponym, level=20)
    if hypernym.name() in r_hypernym_names:
        return True
    else:
        return False


def _calc_path_distance(synset_a, synset_b):
    distance = synset_a.shortest_path_distance(synset_b,
       simulate_root=True and synset_a._needs_root())
    return distance


def _nounify(verb_synset):
    set_of_related_nouns = list()
    for lemma in verb_synset.lemmas():
        for related_form in lemma.derivationally_related_forms():
            for synset in wn.synsets(related_form.name(), pos=wn.NOUN):
                set_of_related_nouns.append(synset)
    return set_of_related_nouns


# create instance for TendencyAnalyzer
tend_analyzer = TendencyAnalyzer()
tend_analyzer.load_word_corpora([('food', 'thing'), ('hobby', 'activity'), ('sport', 'activity')])


if __name__ == "__main__":
    # JENIIFER_DIARY = [
    # ]
    # for i in range(0, len(JENIIFER_DIARY)):
    #     diary = JENIIFER_DIARY[i]
    #     print("start tagging diary #%s" % i)
    #     diary_tags = tagger.tag_pos_doc(diary, True)
    #     print("create piclke for tags of diary #%s" % i)
    #     tagger.tags_to_pickle(diary_tags, "pickles/jennifer" + str(i) + ".pkl")
    #     print()

    # joanne_diaries = list()
    # for i in range(0, 30):
    #     diary_tags = tagger.pickle_to_tags("pickles/joanne" + str(i) + ".pkl")
    #     joanne_diaries.append(diary_tags[1])
    # print("load joanne diaries done.")
    # tend_analyzer.analyze_diary(joanne_diaries,
    #         [('food', 'thing'), ('hobby', 'activity'), ('sport', 'activity')])

    # jeniffer_diaries = list()
    # for i in range(1, 37):
    #     diary_tags = tagger.pickle_to_tags("diary_pickles/jennifer_" + str(i) + ".pkl")
    #     jeniffer_diaries.append(diary_tags[1])
    # print("load jeniffer diaries done.")
    # tend_analyzer.analyze_diary(jeniffer_diaries,
    #         [('food', 'thing'), ('hobby', 'activity'), ('sport', 'activity')])
    # tend_analyzer.analyze_diary(jeniffer_diaries, [('food', 'thing')])
    # tend_analyzer.analyze_diary(jeniffer_diaries, [('exercise', 'activity'),
    #                                                ('food', 'thing')])
    # tend_analyzer.analyze_diary(jeniffer_diaries, [('exercise', 'activity')])
    # tend_analyzer.analyze_diary(jeniffer_diaries, [('hobby', 'activity')])

    # smiley_diaries = list()
    # # for i in range(1, 116):
    # for i in range(1, 50):
    #     diary_tags = tagger.pickle_to_tags("diary_pickles/ksmiley_" + str(i) + ".pkl")
    #     smiley_diaries.append(diary_tags[1])
    # print("load smiley diaries done.")
    # tend_analyzer.analyze_diary(smiley_diaries,
    #         [('food', 'thing'), ('hobby', 'activity'), ('sport', 'activity')])
    # tend_analyzer.analyze_diary(smiley_diaries, [('food', 'thing')])

    # d_diaries = list()
    # for i in range(1, 47):
    #     diary_tags = tagger.pickle_to_tags("diary_pickles/misstick_" + str(i) + ".pkl")
    #     d_diaries.append(diary_tags[1])
    # print("load misstick diaries done.")
    # tend_analyzer.analyze_diary(d_diaries,
    #         [('food', 'thing'), ('hobby', 'activity'), ('sport', 'activity')])

    # elize_diaries = list()
    # for i in range(1, 5):
    #     diary_tags = tagger.pickle_to_tags("diary_pickles/eliz_" + str(i) + ".pkl")
    #     elize_diaries.append(diary_tags[1])
    # print("load eliz diaries done.")
    # tend_analyzer.analyze_diary(elize_diaries, [('food', 'thing')])

    as_travel_diaries = list()
    for i in range(1, 56):
        diary_tags = tagger.pickle_to_tags("diary_pickles/as_travel_" + str(i) + ".pkl")
        as_travel_diaries.append(diary_tags[1])
    print("load as travel diaries done.")
    pos_result, neg_result = tend_analyzer.analyze_diaries(as_travel_diaries)
    pprint(pos_result)
    pprint(neg_result)

    # jeniffer_2015_diaries = list()
    # for i in range(0, 228):
    #     if i == 144:
    #         continue
    #     diary_tags = tagger.pickle_to_tags("pickles/jennifer_2015" + str(i) + ".pkl")
    #     jeniffer_2015_diaries.append(diary_tags[1])
    # print("load jeniffer 2015 diaries done.")
    # tend_analyzer.analyze_diary(jeniffer_2015_diaries,
    #         [('food', 'thing'), ('hobby', 'activity'), ('sport', 'activity')])
    #         # [('exercise', 'activity')])
    #         # [('food', 'thing')])

    # TEST_DIARY5 = "I definitely am a hunter"
    # diary_tags5 = tagger.tag_pos_doc(TEST_DIARY5)
    # print(diary_tags5)
    # print()
    #
    # TEST_DIARY4 = "Pork roast never looked so good.
    # TEST_DIARY4 = "In the evening we visited the local fish restaurant for local fish, calamari, and scallops! "
    # TEST_DIARY4 = "Sailed From Bribie Island to Mooloolaba."
    # TEST_DIARY4 = "After Champagne we had barbecued pork belly with all the trimmings, plenty of wine and a home made Creme Brulee, all on the boat!!"
    # diary_tags4 = tagger.tag_pos_doc(TEST_DIARY4)
    # print(diary_tags4)
    # for i in range(0, len(diary_tags4[1][0])):
    #     print(str(i+1)+':', diary_tags4[1][0][i])
    # print()
    # tend_analyzer.analyze_diary([diary_tags4[1]], [('hobby', 'activity')])
    # tend_analyzer.analyze_diary([diary_tags4[1]], [('food', 'thing')])

    # pprint(tagger.tag_pos_doc("Sue brought a 1000 piece puzzle for us to do as a family and a good sized bottle of Columbia Crest Chardonnay for dinner."))
    # pprint(tagger.tag_pos_doc("An hour later, the 3 of us were pouring over the damn puzzle."))
    # pprint(tagger.tag_pos_doc("She hate me."))
    # pprint(tagger.tag_pos_doc("She doesn't want to cook dinner for me."))

    # likes = PreferenceVerbCorpusRetriever('wordset/like_word_list.txt', 'wordset/dislike_word_list.txt')
    # for s in likes.get_list():
    #     print(s[0].name(), [s[0].lexname(), s[0].definition(), s[5], s[2]['v']])

