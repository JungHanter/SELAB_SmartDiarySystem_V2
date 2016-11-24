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


class WordSetRetriever(object):
    IDX_SYNSET = 0
    IDX_LEVEL = 1
    IDX_LEMMA_WORDS = 2
    IDX_LEMMA_COUNT = 3

    def __init__(self):
        self.synset_list = list()

    def get_list(self):
        return self.synset_list

    def find_synset(self, synset):
        for item in self.synset_list:
            if synset.name() == item[self.IDX_SYNSET].name():
                return item[self.IDX_SYNSET]
        return None

    def check_synset_in(self, synset):
        if self.find_synset(synset) is not None:
            return True
        else:
            return False

    def find_word(self, word):
        for item in self.synset_list:
            if word in item[self.IDX_LEMMA_WORDS]:  # lemma list
                return item[self.IDX_SYNSET]  # synset
        return None

    def check_word_in(self, word):
        if self.find_word(word) is not None:
            return True
        else:
            return False


class HyponymRetriever(WordSetRetriever):
    def __init__(self, *root_synsets, max_level=1, excepts=[]):
        if type(root_synsets[0]) is list or type(root_synsets[0]) is tuple:
            root_synsets = root_synsets[0]
        self.synset_list = list()
        self.excepts = excepts
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
            hyponym_list.append((hyponym, now_level, _lemmas_to_name_list(hyponym.lemmas())))
            hyponym_list = hyponym_list + self._collect_hyponyms(hyponym, max_level, now_level)
        return hyponym_list


class HypernymRetriever(WordSetRetriever):
    def __init__(self, *root_synsets, max_level=1, excepts=[]):
        if type(root_synsets[0]) is list or type(root_synsets[0]) is tuple:
            root_synsets = root_synsets[0]
        self.synset_list = list()
        self.excepts = excepts
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
            hypernym_list.append((hypernym, now_level, _lemmas_to_name_list(hypernym.lemmas())))
            hypernym_list = hypernym_list + self._collect_hyponyms(hypernym, max_level, now_level)
        return hypernym_list

    def get_list(self):
        return self.synset_list


class ListFileRetriever(WordSetRetriever):
    def __init__(self, file_path):
        self.synset_list = list()
        if file_path:
            self.synset_list = self._load_list_file(file_path)

    def _load_list_file(self, file_path):
        synset_list = list()
        with open(file_path, "r") as file:
            while True:
                line = file.readline()
                if not line:
                    break
                line = _text_to_lemma_format(line.strip())
                for synset in wn.synsets(line):
                    synset_list.append((synset, 0, _lemmas_to_name_list(synset.lemmas()),
                                        synset.lexname(), synset.definition()))
            file.close()
        return synset_list


class SynsetListRetirever(WordSetRetriever):
    def __init__(self, file_path):
        self.synset_list = list()
        if file_path:
            self.synset_list = self._load_list_file(file_path)

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
                    lemmas = []
                    for synset_name in synset_group:
                        synset = wn.synset(synset_name)
                        if synset:
                            synsets.append(wn.synset(synset_name))
                            lemmas.extend(_lemmas_to_name_list(synset.lemmas()))
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

    def find_synset(self, synset):
        for item in self.synset_list:
            synset_group = item[0]
            for s in synset_group:
                if synset.name() == s.name():
                    return synset_group[0]
        return None

    def find_word(self, word):
        for item in self.synset_list:
            synset_group = item[0]
            if word in item[self.IDX_LEMMA_WORDS]:  # lemma list
                return synset_group[0]  # synset
        return None


class SentiWordNet(object):
    IDX_POS = 0
    IDX_OFFSET_ID = 1
    IDX_SCORE_POS = 2
    IDX_SCORE_NEG = 3
    IDX_SYNSETS = 4
    IDX_GLOSS = 5

    def __init__(self, filepath):
        self._load_seni_wordnet(filepath)

    def _load_seni_wordnet(self, filepath):
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


class TendencyAnalyzer(object):
    """Perform Tendency analysis"""
    DEBUG = False

    SIMILAR_PATH_MAX_HYPERNYM = 2
    SIMILAR_PATH_MAX_HYPONYM = 1

    CLUSTER_CUT_DIST = 1  #1
    CLUSTER_PATH_DIST_MAGNIFICATION = 1 #4

    def __init__(self, senti_wordnet=None):
        self.senti_wordnet = senti_wordnet
        self.words_sets = dict()
        # and more word set ...

    def add_word_set(self, target, target_type, words_set):
        if words_set:
            self.words_sets[(target, target_type)] = words_set

    # input: a list of tagged diary
    def analyze_diary(self, diary_tags_list, target_types):
        pref_ta_list = list()
        diary_len = len(diary_tags_list)

        # step 1
        self._load_word_corpora(target_types)

        # step 2
        print("\n##### Step 2. #####")
        extracted_sent_dict_list = list()
        for diary_idx in range(0, diary_len):
            diary_tags = diary_tags_list[diary_idx]
            extracted_sent_dict = self._extract_sentences(diary_tags)
            extracted_sent_dict_list.append(extracted_sent_dict)
            print("Diary #%s" % (diary_idx+1))
            # pprint(extracted_sent_dict)
            for sent_id, extracted_words in extracted_sent_dict.items():
                print(str(sent_id) + ':', extracted_words[0])
                for idx in range(1, len(extracted_words)):
                    print('  ', extracted_words[idx])
            print()

        # step 3
        print("\n##### Step 3. #####")
        scores_tend_list = list()
        for diary_idx in range(0, diary_len):
            diary_tags = diary_tags_list[diary_idx]
            extracted_sent_dict = extracted_sent_dict_list[diary_idx]
            scores_tend = self._compute_pref_scores(diary_tags, extracted_sent_dict, diary_idx+1)
            scores_tend_list.append(scores_tend)
            print("Diary #%s" % (diary_idx+1))
            pprint(scores_tend)
            print()

        # step 4
        # print("\n##### Step 4. #####")
        # for diary_idx in range(0, diary_len):
        #     scores_tend = scores_tend_list[diary_idx]
        #     scores_tend_sim = self._compute_pref_scores_of_similars(scores_tend)
        #     scores_tend_list[diary_idx] = scores_tend_sim
        #     print("Diary #%s" % (diary_idx+1))
        #     pprint(scores_tend_sim)
        #     print()

        # step 5
        # convert & add scores_pref dictionary to list
        print("\n##### Step 5. #####")
        tend_list = list()
        for diary_idx in range(0, diary_len):
            scores_tend = scores_tend_list[diary_idx]
            converted_tends = self._convert_scores_dict_to_list(scores_tend)
            tend_list += converted_tends

        # group by category and type
        tend_dict = self._group_tend_list(tend_list)
        clustering_dict = dict()
        for type, tend_group_list in tend_dict.items():
            if len(tend_group_list) < 10:
                continue
            print("Clustering for %s.%s" % (type[1], type[0]))
            clusters, pref_num = self._perform_clustering(tend_group_list)
            clustering_dict[type] = {'clusters': clusters, 'pref_num': pref_num}
            print("Number of Clusters: %s" % len(clusters))
            for idx in range(0, len(clusters)):
                cluster = clusters[idx]
                print("Number of Clsuter #%s: %s" % ((idx+1), len(cluster)))
            print()
            for idx in range(0, len(clusters)):
                cluster = clusters[idx]
                score_sum = 0
                score_cnt = 0
                for ta in cluster:
                    score_sum += ta[1]
                    score_cnt += 1
                if score_cnt == 0:
                    score_cnt = 1
                score_avg = score_sum / score_cnt
                print("Cluster #%s: %s (count: %s, avg_score: %s)" %
                      ((idx+1), self._lable_for_cluster(cluster), len(cluster), score_avg))
                pprint(cluster)
                print()

        # step 6
        print("\n##### Step 6. #####")
        for type, clustering_info in clustering_dict.items():
            pos_results, neg_result = self._figure_out_best_ta(clustering_info['clusters'],
                                                               len(diary_tags_list),
                                                               clustering_info['pref_num'])
            print("Tendency for %s.%s" % (type[1], type[0]))
            for pos_ta in pos_results:
                print(_get_default_lemma(pos_ta[0]) + ': ' + \
                      self._get_preference_class_name(pos_ta[1]) + \
                      ' (' + str(pos_ta[1]) + ')')
            for neg_ta in neg_result:
                print(_get_default_lemma(neg_ta[0]) + ': ' + \
                      self._get_preference_class_name(neg_ta[1]) + \
                      ' (' + str(neg_ta[1]) + ')')
            print()

        # clusters, pref_num = self._perform_clustering(tend_list)
        # # print("# of Cluster: %s" % len(clusters))
        # for idx in range(0, len(clusters)):
        #     cluster = clusters[idx]
        #     score_sum = 0
        #     score_cnt = 0
        #     for ta in cluster:
        #         score_sum += ta[1]
        #         score_cnt += 1
        #     score_avg = score_sum / score_cnt
        #     # print("Cluster #%s: %s (avg: %s)" % (idx, self._lable_for_cluster(cluster), score_avg))
        #     # pprint(cluster)
        #     # print()
        #
        # # step 6
        # print("\n##### Step 6. #####")
        # pos_results, neg_result = self._figure_out_best_ta(clusters, len(diary_tags_list), pref_num)
        # pprint(pos_results)
        # pprint(neg_result)
        # print()

        # return pos_results, neg_result

    ###############################################################################
    # Step 1. Retrieving Corpora about Things, Activities, and Preferences        #
    ###############################################################################
    def _load_word_corpora(self, type_corpora):
        sw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordset',
                              'SentiWordNet_3.0.0_20130122.txt')
        senti_wordnet = SentiWordNet(sw_path)
        foods = HyponymRetriever(wn.synset('food.n.02'), wn.synset('food.n.01'),
                                 max_level=10,
                                 excepts=['slop.n.04', 'loaf.n.02', 'leftovers.n.01',
                                          'convenience_food.n.01', 'nutriment.n.01',
                                          'miraculous_food.n.01', 'micronutrient.n.01',
                                          'feed.n.01', 'fare.n.04', 'cut.n.06',
                                          'culture_medium.n.01', 'comestible.n.01',
                                          'comfort_food.n.01', 'commissariat.n.01',
                                          'alcohol.n.01', 'chyme.n.01', 'meal.n.03',
                                          'variety_meat.n.01'])
        restaurants = HyponymRetriever(wn.synset('restaurant.n.01'), max_level=8)
        weathers = HyponymRetriever(wn.synset('weather.n.01'), max_level=8,
                                    excepts=['thaw.n.02', 'wave.n.08', 'wind.n.01',
                                             'elements.n.01', 'atmosphere.n.04'])
        exercises = HyponymRetriever(wn.synset('sport.n.01'), wn.synset('sport.n.02'),
                                     wn.synset('exercise.n.01'), wn.synset('exercise.v.03'),
                                     wn.synset('exercise.v.04'), max_level=12,
                                     excepts=['set.n.03'])
        hobbies = SynsetListRetirever("wordset/hobbies_wiki_wordnet.txt")

        self.senti_wordnet = senti_wordnet

        for type in type_corpora:
            word_set_corpus = None
            if type == ('food', 'thing'):
                word_set_corpus = foods
            elif type == ('restaurant', 'thing'):
                word_set_corpus = restaurants
            elif type == ('weather', 'thing'):
                word_set_corpus = weathers
            elif type == ('hobby', 'activity'):
                word_set_corpus = hobbies
            elif type == ('exercise', 'activity'):
                word_set_corpus = exercises
            self.add_word_set(type[0], type[1], word_set_corpus)

    ###############################################################################
    # Step 2. Identifying Sentences including Things and Activities in Each Diary #
    ###############################################################################
    def _extract_sentences(self, diary_tags):
        identified_sent_dict = defaultdict(lambda: list())
        for sent_idx in range(0, len(diary_tags)):
            tagged_sent = diary_tags[sent_idx]

            # for last 'for' loop in word
            if tagged_sent[len(tagged_sent)-1][TAG_WORD_POS] is not None or \
                    tagged_sent[len(tagged_sent) - 1][TAG_WORD_ROLE] is not None:
                tagged_sent.append((None, None, None, None, None))

            prev_word_comp_list = [] # for compound word
            for word_idx in range(0, len(tagged_sent)):
                word = tagged_sent[word_idx]

                # find word as noun
                if word[TAG_WORD_POS] is None or not word[TAG_WORD_POS].startswith('NN'):
                    # end of single or compound noun
                    if len(prev_word_comp_list) > 0:
                        is_found = False
                        search_idx = 0
                        # find things and activities as noun
                        # search by backward
                        while True:
                            search_word_comp_list = prev_word_comp_list[search_idx:]
                            last_word = search_word_comp_list[len(search_word_comp_list)-1]
                            prev_word_idx = last_word[1]
                            plural = last_word[2]
                            found_synset_list \
                                = self._find_comp_synsets_in_wordsets(search_word_comp_list, plural)
                            if found_synset_list and prev_word_idx != -1:
                                for found_synset in found_synset_list:
                                    identified_sent_dict[sent_idx].append(found_synset + (prev_word_idx, 'n'))
                                is_found = True
                                break
                            else:
                                # find next compound word
                                search_idx += 1
                                if search_idx >= len(prev_word_comp_list): # there is no thing or activity
                                    break
                        # search by forward
                        if not is_found and len(prev_word_comp_list) > 1:
                            search_idx = len(prev_word_comp_list) - 1
                            while True:
                                search_word_comp_list = prev_word_comp_list[:search_idx]
                                last_word = search_word_comp_list[len(search_word_comp_list) - 1]
                                prev_word_idx = last_word[1]
                                plural = last_word[2]
                                found_synset_list \
                                    = self._find_comp_synsets_in_wordsets(search_word_comp_list, plural)
                                if found_synset_list and prev_word_idx != -1:
                                    for found_synset in found_synset_list:
                                        identified_sent_dict[sent_idx].append(found_synset + (prev_word_idx, 'n'))
                                    is_found = True
                                    break
                                else:
                                    # find next compound word
                                    search_idx -= 1
                                    if search_idx <= 0:  # there is no thing or activity
                                        break
                        # search by one by one
                        if not is_found and len(prev_word_comp_list) > 2:
                            search_idx = 1
                            while True:
                                search_word_comp_list = prev_word_comp_list[search_idx:search_idx+1]
                                last_word = search_word_comp_list[0]
                                prev_word_idx = last_word[1]
                                plural = last_word[2]
                                found_synset_list \
                                    = self._find_comp_synsets_in_wordsets(search_word_comp_list, plural)
                                if found_synset_list and prev_word_idx != -1:
                                    for found_synset in found_synset_list:
                                        identified_sent_dict[sent_idx].append(found_synset + (prev_word_idx, 'n'))
                                    break
                                else:
                                    # find next compound word
                                    search_idx += 1
                                    if search_idx >= len(prev_word_comp_list)-1:  # there is no thing or activity
                                        break
                        prev_word_comp_list.clear()

                # a word have pos and role
                if word[TAG_WORD_POS] is not None and word[TAG_WORD_ROLE] is not None:
                    if 'VB' in word[TAG_WORD_POS]:
                        # find activities as verb
                        found_synset_list \
                            = self._find_synsets_in_wordsets([word[TAG_WORD]])
                        for found_synset in found_synset_list:
                            identified_sent_dict[sent_idx].append(found_synset + (word_idx, 'v'))
                        prev_word_comp_list.clear()

                    elif word[TAG_WORD_POS].startswith('NN') and \
                            ('subj' in word[TAG_WORD_ROLE] or 'obj' in word[TAG_WORD_ROLE] or
                                 word[TAG_WORD_ROLE is 'conj']):
                        # check the noun is plural
                        plural = False
                        if word[TAG_WORD_POS].endswith('S'):
                            plural = True
                        prev_word_comp_list.append((word[TAG_WORD], word_idx, plural))

                        ##this

                    # For the compound words
                    elif (word[TAG_WORD_POS].startswith('JJ') and word[TAG_WORD_ROLE] == 'amod') or \
                            (word[TAG_WORD_POS].startswith('NN') and word[TAG_WORD_ROLE] == 'compound'):
                        prev_word_comp_list.append((word[TAG_WORD], -1, False))
        return identified_sent_dict

    ###############################################################################
    # Step 3. Computing Preference Scores of Things and Activities in Each Diary  #
    ###############################################################################
    def _compute_pref_scores(self, diary_sent_list, identified_sent_dict, diary_idx=None):
        scores_pref = dict()
        scores_pref_sent = defaultdict(lambda: {'score': 0.0, 'count': 0, 'type': None})

        # analyze sentence to compute preference score
        for sent_idx, identified_info_list in identified_sent_dict.items():
            tagged_sent = diary_sent_list[sent_idx]

            is_sent_mine = False  # 'I' is subject of sentence
            is_sent_we = False # 'We'
            is_sent_subj = False  # the thing or activity is subject of sentence
            is_sent_neg = False  # the sentence is negative

            weights_word = defaultdict(lambda: {'type': None, 'weight': 0.0})
            weight_sent = 0  # weight of sentiment words
            count_sent = 0  # the number of sentiment words
            adverb_neg = {'score': 0.0, 'count': 0}
            adverb_pos = {'score': 0.0, 'count': 0}

            # indices for identified word
            identified_word_idxs = list()
            for identified_info in identified_info_list:
                # add index of identified word
                identified_word_idxs.append(identified_info[3])

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
                if count_sum == 0 and count_for_synset == 0:
                    count_sum = 2
                    count_for_synset = 1
                word_freq_weight = count_for_synset / count_sum
                synset_name = identified_synset.name()
                weights_word[synset_name]['weight'] = word_freq_weight \
                    if word_freq_weight > weights_word[synset_name]['weight'] \
                    else weights_word[synset_name]['weight']
                weights_word[synset_name]['type'] = identified_info[2]

            # identify roles, figure out negative, and calculate the degree of sentiment
            for word_idx in range(0, len(tagged_sent)):
                word = tagged_sent[word_idx]

                # if already identified word
                if word_idx in identified_word_idxs:
                    # check the subject of sentence is the identified word
                    if 'subj' in word[TAG_WORD_ROLE]:
                        is_sent_subj = True
                    continue

                if word[TAG_WORD_POS] is not None and word[TAG_WORD_ROLE] is not None:
                    # check the subject of sentence is I
                    if 'I' == word[TAG_WORD] and 'subj' in word[TAG_WORD_ROLE]:
                        is_sent_mine = True
                        continue

                    if 'we' == word[TAG_WORD].lower() and 'subj' in word[TAG_WORD_ROLE]:
                        is_sent_we = True
                        continue

                    # check the sentence is negative
                    elif 'neg' in word[TAG_WORD_ROLE]:
                        is_sent_neg = True
                        continue

                    # check a sentiment score of an adverb in the sentence
                    elif word[TAG_WORD_POS].startswith('RB'):
                        offsets = _find_offsets_from_word(word[TAG_WORD], 'r')
                        for offset in offsets:
                            senti_score = self.senti_wordnet.get_score(offset, 'r')
                            if senti_score['positivity'] == 1 and senti_score['negativity'] == 0:
                                continue
                            adverb_neg['score'] += senti_score['negativity']
                            adverb_neg['count'] += 1
                            adverb_pos['score'] += senti_score['positivity']
                            adverb_pos['count'] += 1

                    # check a sentiment score of a verb in the sentence
                    elif 'VB' in word[TAG_WORD_POS]:
                        verb_weight = 0.5
                        sent_score = 0
                        if word[TAG_WORD_ROLE] == 'root':  # main verb...
                            verb_weight = 1
                        offsets = _find_offsets_from_word(word[TAG_WORD], 'v')
                        word_cnt = 0
                        for offset in offsets:
                            senti_score_dict = self.senti_wordnet.get_score(offset, 'v')
                            senti_score = senti_score_dict['positivity'] - senti_score_dict['negativity']
                            senti_count = senti_score_dict['corpus_count'] + 1
                            if senti_score == 0:
                                continue
                            else:
                                sent_score += (senti_score * senti_count)
                                word_cnt += senti_count
                        if word_cnt == 0:
                            sent_score = 0
                        else:
                            # average of sent score of sentiment word considering count in corpus
                            sent_score = sent_score / word_cnt
                        if sent_score is not 0:
                            weight_sent += verb_weight * sent_score
                            count_sent += 1

                    # Check a sentiment score of an adjective in the sentence
                    elif 'JJ' in word[TAG_WORD_POS]:
                        adjective_weight = 1
                        sent_score = 0

                        if word[TAG_WORD_ROLE] == 'root':
                            adjective_weight = 1
                        elif 'amod' in word[TAG_WORD_ROLE]:
                            adjective_weight = 0.75
                        elif 'mod' in word[TAG_WORD_ROLE]:
                            adjective_weight = 0.5
                        else:
                            continue

                        offsets = _find_offsets_from_word(word[TAG_WORD], 'a')
                        word_cnt = 0
                        # for offset in offsets:
                        #     senti_score = self.senti_wordnet.get_score_value(offset, 'a')
                        #     if senti_score == 0:
                        #         continue
                        #     else:
                        #         sent_score += senti_score
                        #         word_cnt += 1
                        for offset in offsets:
                            senti_score_dict = self.senti_wordnet.get_score(offset, 'v')
                            senti_score = senti_score_dict['positivity'] - senti_score_dict['negativity']
                            senti_count = senti_score_dict['corpus_count'] + 1
                            if senti_score == 0:
                                continue
                            else:
                                sent_score += (senti_score * senti_count)
                                word_cnt += senti_count
                        if word_cnt == 0:
                            sent_score = 0
                        else:
                            # average of sent score of sentiment word considering count in corpus
                            sent_score = sent_score / word_cnt
                        if sent_score is not 0:
                            weight_sent += adjective_weight * sent_score
                            count_sent += 1

            # apply preference score of things and activities in the sentence
            for synset_name, word_weight in weights_word.items():
                if self.DEBUG:
                    print('================')
                    print('sentence_idx: %s' % sent_idx)
                    print('word_name: %s' % synset_name)
                    print('word_weight: %s' % word_weight)
                # calculate weight of subjectivity
                weight_subj = 1 if (is_sent_mine or is_sent_we) else (0.8 if is_sent_subj else 0.1)
                weight_subj *= -0.8 if is_sent_neg else 1
                if self.DEBUG: print('weight_subj: %s' % weight_subj)

                # calculate average of weight of sentiment words
                if count_sent > 0:
                    weight_sent = weight_sent / count_sent
                if weight_sent < 0.25 and weight_sent >= 0:
                    weight_sent = 0.25  # the minimum neutral weight of sent
                if self.DEBUG: print('weight_sent: %s' % weight_sent)

                sent_score = weight_subj * weight_sent
                if self.DEBUG: print('sent_score: %s (subj*sent)' % sent_score)

                adverb_score = 0
                adverb_neg_score = 0
                adverb_neg_flag = False
                if sent_score >= 0:
                    if adverb_pos['count'] > 0:
                        adverb_score = sent_score * (adverb_pos['score'] / adverb_pos['count'])
                    if adverb_neg['count'] > 0 and adverb_neg['score'] > adverb_pos['score']:
                        adverb_neg_score = (adverb_score + sent_score) * (
                        adverb_neg['score'] / adverb_neg['count']) * -1
                        adverb_neg_flag = True
                else:
                    if adverb_pos['count'] > 0:
                        adverb_score = sent_score * (adverb_pos['score'] / adverb_pos['count']) * -1
                    if adverb_neg['count'] > 0 and adverb_neg['score'] > adverb_pos['score']:
                        adverb_neg_score = (adverb_score + sent_score) * \
                                           (adverb_neg['score'] / adverb_neg['count']) * -1
                        adverb_neg_flag = True
                if self.DEBUG: print('adverb_pos_score: %s' % adverb_score)
                if self.DEBUG: print('adverb_neg_score %s' % adverb_neg_score)

                score = 0
                if sent_score >= 0:
                    # score = (word_weight['weight'] * 0.5) + (sent_score * word_weight['weight'] * 0.5)
                    score = (word_weight['weight'] * 0.25) + (sent_score * word_weight['weight'] * 0.75)
                    if self.DEBUG:
                        print('word score with sent: %s = (%s*0.5) + (%s*%s*0.5)' %
                              (score, word_weight['weight'], sent_score, word_weight['weight']))
                else:
                    # score = -(word_weight['weight'] * 0.5) + (sent_score * word_weight['weight'] * 0.5)
                    score = -(word_weight['weight'] * 0.25) + (sent_score * word_weight['weight'] * 0.75)
                    if self.DEBUG:
                        print('word score with sent: %s = -(%s*0.5) + (%s*%s*0.5)' %
                              (score, word_weight['weight'], sent_score, word_weight['weight']))
                score = (score + adverb_score) * (adverb_neg_score if adverb_neg_flag else 1)
                if self.DEBUG: print('result score applying adverb: %s' % score)

                # scores_word[synset_name] = score
                scores_pref_sent[synset_name]['score'] += score
                scores_pref_sent[synset_name]['count'] += 1
                if not scores_pref_sent[synset_name]['type']:
                    scores_pref_sent[synset_name]['type'] = word_weight['type']

        # compute preference scores for diaries
        for synset_name, sent_dict in scores_pref_sent.items():
            scores_pref[synset_name] = dict()
            if diary_idx is not None:
                scores_pref[synset_name]['diary_idx'] = diary_idx
            # not just average, if the word is refered many times, it have to multiply weight
            scores_pref[synset_name]['score'] = sent_dict['score'] / sent_dict['count'] * \
                                                (1 + ((sent_dict['count']-1) / 11)) # 1+(count-1/11) weight
            if scores_pref[synset_name]['score'] > 1:
                scores_pref[synset_name]['score'] = 1
            elif scores_pref[synset_name]['score'] < -1:
                scores_pref[synset_name]['score'] = -1
            scores_pref[synset_name]['type'] = sent_dict['type']

        return scores_pref

    ###############################################################################
    # Step 4. Computing Preference Scores of Similar Things and Activities        #
    #         of the Scored Things and Activities                                 #
    ###############################################################################
    def _compute_pref_scores_of_similars(self, scores_pref):
        # dict for sum and count of similar words
        scores_pref_rel = defaultdict(lambda: {'score': 0.0, 'count': 0, 'type': None})

        for synset_name, score_dict in scores_pref.items():
            # if score is 0, pass
            if score_dict['score'] == 0:
                continue

            lemmas_of_type_hypernyms = _get_hypernym_lemmas(word=score_dict['type'][0])
            lemmas_of_type_hypernyms.extend(_get_hypernym_lemmas(word=score_dict['type'][1]))

            # find hypernyms and hyponyms and score to them
            hypernyms = HypernymRetriever(wn.synset(synset_name), max_level=self.SIMILAR_PATH_MAX_HYPERNYM).get_list()
            for hypernym in hypernyms:
                if score_dict['type'][0] in hypernym[2] or \
                        score_dict['type'][1] in hypernym[2]:
                    continue
                root_word_found = False
                for lemma in hypernym[0].lemmas():
                    if lemma in lemmas_of_type_hypernyms:
                        root_word_found = True
                        break
                if root_word_found:
                    continue
                scores_pref_rel[hypernym[0].name()]['score'] += score_dict['score'] / (hypernym[1]+1)
                scores_pref_rel[hypernym[0].name()]['count'] += 1
                if not scores_pref_rel[hypernym[0].name()]['type']:
                    scores_pref_rel[hypernym[0].name()]['type'] = score_dict['type']
            # for i in range(0, len(hypernym))[::-1]:
            #     hypernym = hypernyms[i]
            #     if score_dict['type'][] in hypernym

            hyponyms = HyponymRetriever(wn.synset(synset_name), max_level=self.SIMILAR_PATH_MAX_HYPONYM).get_list()
            for hyponym in hyponyms:
                scores_pref_rel[hyponym[0].name()]['score'] += score_dict['score'] / ((hyponym[1]+1)*2)
                scores_pref_rel[hyponym[0].name()]['count'] += 1
                if not scores_pref_rel[hyponym[0].name()]['type']:
                    scores_pref_rel[hyponym[0].name()]['type'] = score_dict['type']

        # compute preference score of similar things and activities
        for synset_name, score_dict in scores_pref_rel.items():
            pref_score_rel = score_dict['score'] / score_dict['count']

            # if similar thing or activity is already existed thing or activity, add score
            if synset_name in scores_pref.keys():
                scores_pref[synset_name]['score'] += pref_score_rel
                if scores_pref[synset_name]['score'] > 1:
                    scores_pref[synset_name]['score'] = 1
                elif scores_pref[synset_name]['score'] < -1:
                    scores_pref[synset_name]['score'] = -1
            else:
                scores_pref[synset_name] = dict()
                scores_pref[synset_name]['score'] = pref_score_rel
                scores_pref[synset_name]['type'] = score_dict['type']

        return scores_pref

    ###############################################################################
    # Step 5. Clustering the Things and Activities                                #
    ###############################################################################
    def _perform_clustering(self, pref_ta_list):
        def calc_distance(u, v):
            # path_dist = wn.synset(u[0]).path_similarity(wn.synset(v[0]))
            # if path_dist is None:
            #     path_dist = 0
            # return 1 - (path_dist * self.CLUSTER_PATH_DIST_MAGNIFICATION)

            if u[0] == v[0]:
                return 0
            elif u[0] in _get_hypernyms_name(wn.synset(v[0]), level=3) or \
                    v[0] in _get_hypernyms_name(wn.synset(u[0]), level=3):
                path_dist = wn.synset(u[0]).path_similarity(wn.synset(v[0]))
                if path_dist is None:
                    path_dist = 0
                return 1 - path_dist
                # return 0
            else:
                return 1

        # sort list for debugging
        if self.DEBUG:
            pref_ta_list = sorted(pref_ta_list, key=lambda pref_ta: pref_ta[0])

        # print list for debug
        if self.DEBUG:
            print("pref_ta_list(all): ")
            pprint(pref_ta_list)
            print()

        # feature selection
        # filtering with things and activities which there is only one item (with low score)
        count_dict = defaultdict(int)
        for pref_ta in pref_ta_list:
            count_dict[pref_ta[0]] += 1
        for i in range(0, len(pref_ta_list))[::-1]:
            if count_dict[pref_ta_list[i][0]] < 2:
                pref_ta_list.pop(i)
        # filtering with things and activities which has more than 0.01 preference score
        for i in range(0, len(pref_ta_list))[::-1]:
            if abs(pref_ta_list[i][1]) < 0.01:
                pref_ta_list.pop(i)
        pref_num = len(pref_ta_list)

        # print list for debug
        if self.DEBUG:
            print("pref_ta_list(filtered): ")
            pprint(pref_ta_list)
            print()

        # make distance matrix
        pref_len = len(pref_ta_list)
        dist_matrix = np.array([list(10.0 for i in range(pref_len)) for j in range(pref_len)], np.float32)
        for u in range (0, pref_len):
            for v in range(u, pref_len):
                dist = calc_distance(pref_ta_list[u], pref_ta_list[v])
                dist_matrix[u][v] = dist
                dist_matrix[v][u] = dist

        if self.DEBUG:
            print("distance matrix: ")
            print(dist_matrix)
            print()

        # perform clustering
        hac_result = hac.linkage(dist_matrix, method='single')
        if self.DEBUG:
            print("linkage result: ")
            print(hac_result)
            print()

        # figure out the number of clusters (determine where to cut tree)
        num_cluster = 1
        for matrix_y in hac_result:
            if matrix_y[2] >= self.CLUSTER_CUT_DIST:
            # if matrix_y[2] > 0.25:
                num_cluster += 1
        if self.DEBUG:
            print("num_cluster: ", num_cluster, '\n')

        part_cluster = hac.fcluster(hac_result, num_cluster, 'maxclust')
        if self.DEBUG:
            print("part_cluster: ")
            print(part_cluster)
            print()

        # batch each thing and activity to its cluster
        clusters = [[] for i in range(num_cluster)]
        for idx_ta in range(0, len(part_cluster)):
            cluster_id = part_cluster[idx_ta] - 1
            clusters[cluster_id].append(pref_ta_list[idx_ta])
        if self.DEBUG:
            print("clusters:")
            pprint(clusters)
            print()

        # show dendrogram
        # labels = list('' for i in range(pref_len))
        # for i in range(pref_len):
        #     # labels[i] = str(i) + ' (' + str(part_cluster[i]) + ')'
        #     # labels[i] = str(i)
        #     labels[i] = '[' + str(part_cluster[i]) + '] ' + pref_ta_list[i][0] + '(' + str(i) + ')\n' + \
        #                 str(int(pref_ta_list[i][1] * 100000) / 100000.0)
        #     # labels[i] = _get_default_lemma(pref_ta_list[i][0]) + '\n' + \
        #     #             str(int(pref_ta_list[i][1] * 1000) / 1000.0)
        # ct = hac_result[-(num_cluster - 1), 2]
        # p = hac.dendrogram(hac_result, labels=labels, color_threshold=ct)
        # plt.show()

        return clusters, pref_num

    ###############################################################################
    # Step 6. Figuring out Things and Activities having the Best Preference Score #
    ###############################################################################
    def _figure_out_best_ta(self, clusters, diary_num, pref_num):
        pos_ta_cluster_dict = dict()
        neg_ta_cluster_dict = dict()

        min_cluter_item = diary_num / 30.0
        for cluster in clusters:
            # if the number of the items in cluster is a few
            if len(cluster) < min_cluter_item:
                continue

            target = cluster[0][2]
            target_type = cluster[0][3]

            # add weight for things or activities which are more count than others
            # weight formula can be changed
            # now: weight = 1 + log_max_count(count)
            count_dict = defaultdict(lambda: {'count':0, 'sum':0.0})  # dict for counting
            for ta in cluster:
                count_dict[ta[0]]['count'] += 1
                count_dict[ta[0]]['sum'] += ta[1]
            weight_dict = dict()
            for ta_name, count in count_dict.items():
                if count['count'] >= 2:
                    # average * count_weight
                    weight_dict[ta_name] = (count['sum'] / count['count']) * \
                                           (1 + log(count['count']-1, pref_num))

            # remove same items and set new pref score multiplied weight
            already_exist_ta_list = list()
            for i in range(0, len(cluster))[::-1]:
                ta = cluster[i]
                if ta[0] in already_exist_ta_list:
                    cluster.pop(i)
                    continue
                if ta[0] in weight_dict.keys():
                    ext_ta = cluster.pop(i)
                    new_ta = (ext_ta[0], weight_dict[ta[0]], ext_ta[2], ext_ta[3])
                    cluster.insert(i, new_ta)
                    already_exist_ta_list.append(ta[0])

            if self.DEBUG:
                print("Cluster after applying weight for count")
                print(cluster)
                print()

            # find things and activities ta having macx prefs
            # first_ta = cluster[0]
            max_pos_pref_ta_list = []
            max_pos_pref_score = -1
            max_neg_pref_ta_list = []
            max_neg_pref_score = 1
            for idx_ta in range(0, len(cluster)):
                ta = cluster[idx_ta]
                if ta[1] >= 0:
                    if max_pos_pref_score == ta[1]:
                        max_pos_pref_ta_list.append(ta)
                    elif max_pos_pref_score < ta[1]:
                        max_pos_pref_score = ta[1]
                        max_pos_pref_ta_list.clear()
                        max_pos_pref_ta_list.append(ta)
                else:
                    if max_neg_pref_score == ta[1]:
                        max_neg_pref_ta_list.append(ta)
                    elif max_neg_pref_score > ta[1]:
                        max_neg_pref_score = ta[1]
                        max_neg_pref_ta_list.clear()
                        max_neg_pref_ta_list.append(ta)

            if self.DEBUG:
                print("Max pref list")
                print(max_pos_pref_ta_list)
                print(max_neg_pref_ta_list)
                print()

            # if all values are same, find their common parent(hypernym)
            parent_ta_find_dict = defaultdict(int)
            if (len(cluster) == len(max_pos_pref_ta_list) and len(max_pos_pref_ta_list) > 1) or \
                    (len(cluster) == len(max_neg_pref_ta_list) and len(max_neg_pref_ta_list) > 1):
                for ta in cluster:
                    # find all hypernym
                    for hypernym in wn.synset(ta[0]).hypernyms():
                        parent_ta_find_dict[hypernym.name()] += 1
                # find max hypernym(s)
                max_count = -1
                max_hypyernym = ''
                for hypernym_name, count in parent_ta_find_dict.items():
                    if max_count < count:
                        max_hypyernym = hypernym_name
                    elif max_count == count:
                        if type(max_hypyernym) is list:
                            max_hypyernym.append(hypernym_name)
                        else:
                            max_hypyernym = [max_hypyernym]
                            max_hypyernym.append(hypernym_name)
                # set max hypernyms to max_pos_pref_list or max_neg_pref_list
                if len(cluster) == len(max_pos_pref_ta_list):
                    max_pos_pref_score = cluster[0][1] * 1.2
                    max_pos_pref_ta_list = list()
                    if type(max_hypyernym) is list:
                        for hypernym in max_hypyernym:
                            max_pos_pref_ta_list.append((hypernym, max_pos_pref_score, target, target_type))
                    else:
                        max_pos_pref_ta_list.append((max_hypyernym, max_pos_pref_score, target, target_type))
                else:
                    max_neg_pref_score = cluster[0][1] * 1.2
                    max_neg_pref_ta_list = list()
                    if type(max_hypyernym) is list:
                        for hypernym in max_hypyernym:
                            max_neg_pref_ta_list.append((hypernym, max_neg_pref_score, target, target_type))
                    else:
                        max_neg_pref_ta_list.append((max_hypyernym, max_neg_pref_score, target, target_type))

            if self.DEBUG:
                print("Max pref list after parent finding")
                print(max_pos_pref_ta_list)
                print(max_neg_pref_ta_list)
                print()

            # plus score to max value from others
            pos_pref_ta_list_cvt = list()
            for pos_ta in max_pos_pref_ta_list:
                pos_pref_ta_list_cvt.append(list(pos_ta))
            for pos_ta in pos_pref_ta_list_cvt:
                for ta in cluster:
                    if ta is not pos_ta:
                        # add a little score which is from other items in the same cluster
                        path_sim = wn.synset(pos_ta[0]).path_similarity(wn.synset(ta[0]))
                        if path_sim is None: path_sim = 0
                        pos_ta[1] += 0.1 * path_sim * ta[1]
            neg_pref_ta_list_cvt = list()
            for neg_ta in max_neg_pref_ta_list:
                neg_pref_ta_list_cvt.append(list(neg_ta))
            for neg_ta in neg_pref_ta_list_cvt:
                for ta in cluster:
                    if ta is not neg_ta:
                        # add a little score which is from other items in the same cluster
                        path_sim = wn.synset(neg_ta[0]).path_similarity(wn.synset(ta[0]))
                        if path_sim is None: path_sim = 0
                        neg_ta[1] += 0.1 * path_sim * ta[1]
            if self.DEBUG:
                print('cvt list')
                print(pos_pref_ta_list_cvt)
                print(neg_pref_ta_list_cvt)
                print()

            # sometimes, final preference score is more than 1 because
            # there are many related ta or the ta is referred in diaries as much times..
            for pos_ta in pos_pref_ta_list_cvt:
                if pos_ta[1] > 1:
                    pos_ta[1] = 1
            for neg_ta in neg_pref_ta_list_cvt:
                if neg_ta[1] < -1:
                    neg_ta[1] = -1

            # add to final dict if the ta is higher
            for pos_ta in pos_pref_ta_list_cvt:
                if pos_ta[0] in pos_ta_cluster_dict.keys():
                    if pos_ta[1] > pos_ta_cluster_dict[pos_ta[0]][1]:
                        # set higher score
                        pos_ta_cluster_dict[pos_ta[0]][1] = pos_ta[1]
                else:
                    pos_ta_cluster_dict[pos_ta[0]] = pos_ta
            for neg_ta in neg_pref_ta_list_cvt:
                if neg_ta[0] in neg_ta_cluster_dict.keys():
                    if neg_ta[1] < neg_ta_cluster_dict[neg_ta[0]][1]:
                        # set higher score
                        neg_ta_cluster_dict[neg_ta[0]][1] = neg_ta[1]
                else:
                    neg_ta_cluster_dict[neg_ta[0]] = neg_ta

        if self.DEBUG:
            pprint(pos_ta_cluster_dict)
            print()
            pprint(neg_ta_cluster_dict)
            print()

        # it a thing or activity is in both side, to correct it.
        pos_ta_cluster_keys = list(pos_ta_cluster_dict.keys())
        for idx in range(0, len(pos_ta_cluster_keys))[::-1]:
            ta_name = pos_ta_cluster_keys[idx]
            if ta_name in neg_ta_cluster_dict.keys():
                ta_score = pos_ta_cluster_dict[ta_name][1] + neg_ta_cluster_dict[ta_name][1]
                if ta_score == 0:
                    del pos_ta_cluster_dict[ta_name]
                    del neg_ta_cluster_dict[ta_name]
                elif ta_score < 0:
                    del pos_ta_cluster_dict[ta_name]
                    neg_ta_cluster_dict[ta_name][1] = ta_score
                else:   # elif ta_score > 0:
                    del neg_ta_cluster_dict[ta_name]
                    pos_ta_cluster_dict[ta_name][1] = ta_score

        # scores list filter
        pos_scores_ta = list()
        for ta_name, ta in pos_ta_cluster_dict.items():
            if ta[1] >= 0.2:   # preference score is more than
                pos_scores_ta.append((ta[0], ta[1]))
        neg_scores_ta = list()
        for ta_name, ta in neg_ta_cluster_dict.items():
            if ta[1] <= -0.2:
                neg_scores_ta.append((ta[0], ta[1]))

        # best tend things and activities
        best_pos_scores = sorted(pos_scores_ta, key=lambda ta: ta[1], reverse=True)[:5]
        best_neg_scores = sorted(neg_scores_ta, key=lambda ta: ta[1])[:5]

        # arrange things and activities for their type
        # clsfied_pos_scores_dict = defaultdict(lambda: list())
        # for ta_name, ta in pos_ta_cluster_dict.items():
        #     if ta[1] >= 0.2:   # preference score is more than
        #         clsfied_pos_scores_dict[(ta[2], ta[3])].append((ta[0], ta[1]))
        # clsfied_neg_scores_dict = defaultdict(lambda: list())
        # for ta_name, ta in neg_ta_cluster_dict.items():
        #     if ta[1] <= -0.2:
        #         clsfied_neg_scores_dict[(ta[2], ta[3])].append((ta[0], ta[1]))
        #
        # if self.DEBUG:
        #     pprint(clsfied_pos_scores_dict)
        #     pprint(clsfied_neg_scores_dict)
        #     print()
        #
        # # trim up to 5 first element as best score
        # best_pos_scores_dict = dict()
        # for ta_type, ta_list in clsfied_pos_scores_dict.items():
        #     best_pos_scores_dict[ta_type] = sorted(ta_list, key=lambda ta: ta[1], reverse=True)[:5]
        # best_neg_scores_dict = dict()
        # for ta_type, ta_list in clsfied_neg_scores_dict.items():
        #     best_neg_scores_dict[ta_type] = sorted(ta_list, key=lambda ta: ta[1])[:5]
        #
        # if self.DEBUG:
        #     pprint(best_pos_scores_dict)
        #     pprint(best_neg_scores_dict)
        #     print()

        return best_pos_scores, best_neg_scores

    # return a list of (found_synset, lemma_word, words_set_type)
    def _find_synsets_in_wordsets(self, finding_word_list, plural=False):
        synset_list = list()
        for words_set_type, words_set in self.words_sets.items():
            found_synset, lemma_word = \
                self._find_synset_by_word_list(words_set, finding_word_list, plural)
            if found_synset:
                synset_list.append((found_synset, lemma_word, words_set_type))
        return synset_list

    def _find_comp_synsets_in_wordsets(self, finding_comp_word_list, plural=False):
        finding_word_list = list()
        for finding_comp_word in finding_comp_word_list:
            finding_word_list.append(finding_comp_word[0])
        return self._find_synsets_in_wordsets(finding_word_list, plural)

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
    def _find_synset_by_word_list(cls, words_set, finding_word_list, plural=False):
        length = len(finding_word_list)
        for i in range(0, length):
            lemma_word = ""
            for k in range(i, length):
                if i < k:
                    lemma_word += '_'
                lemma_word += finding_word_list[k]
            # print(lemma_word)
            synset = words_set.find_word(lemma_word)
            if synset is not None:
                return synset, lemma_word

        # if lemma is not found, but the noun in lemma is plural
        if plural:
            try:
                plural_noun = wn.synsets(finding_word_list[length - 1])[0].lemmas()[0].name()
                finding_word_list[length - 1] = plural_noun
                return cls._find_synset_by_word_list(words_set, finding_word_list, False)
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
    def _convert_scores_dict_to_list(cls, scores_pref):
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
    def _find_common_hypernyms(cls, ta_name_list, search_level=3):
        hypernym_count_dict = defaultdict(int)
        for ta_name in ta_name_list:
            synset = wn.synset(ta_name)
            # add self
            hypernym_count_dict[ta_name] += 1
            # add hypernyms
            if synset:
                for hypernym in _get_hypernyms(synset, search_level):
                    hypernym_count_dict[hypernym.name()] += 1
        max_hypernym_list = list()
        max_hypernym_count = 0
        for hypernym_name, count in hypernym_count_dict.items():
            if max_hypernym_count < count:
                max_hypernym_list.clear()
                max_hypernym_count = count
                max_hypernym_list.append(hypernym_name)
            elif max_hypernym_count == count:
                max_hypernym_list.append(hypernym_name)
        if max_hypernym_count >= len(ta_name_list):
            return max_hypernym_list
        return []

    @classmethod
    def _lable_for_cluster(cls, cluster):
        sum_score = 0
        cnt_score = 0

        count_item_dict = defaultdict(int)
        for ta in cluster:
            count_item_dict[ta[0]] += 1
            sum_score += ta[1]
            cnt_score += 1
        if cnt_score == 0:
            cnt_score = 1
        avg_score = sum_score / cnt_score

        max_ta_count = 0
        max_ta_list = list()
        for ta_name, count in count_item_dict.items():
            if max_ta_count < count:
                max_ta_list.clear()
                max_ta_count = count
                max_ta_list.append(ta_name)
            elif max_ta_count == count:
                max_ta_list.append(ta_name)

        rep_ta_name = ''
        if len(max_ta_list) == 1:
            rep_ta_name = _get_default_lemma(max_ta_list[0])
        else:
            common_hypernyms = cls._find_common_hypernyms(max_ta_list)
            if common_hypernyms:
                for idx in range(0, len(common_hypernyms)):
                    hypernym_name = common_hypernyms[idx]
                    if idx > 0:
                        rep_ta_name += ', '
                    rep_ta_name += _get_default_lemma(hypernym_name)
            else:
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
    hypernym_list = list()
    for hypernym in synset.hypernyms():
        hypernym_list.append(hypernym)
        if level > 1:
            hypernym_list.extend(_get_hypernyms(hypernym, level-1))
    return hypernym_list


def _get_hypernyms_name(synset, level=99999):
    hypernym_name_list = list()
    for hypernym in synset.hypernyms():
        hypernym_name_list.append(hypernym.name())
        if level > 1:
            hypernym_name_list.extend(_get_hypernyms_name(hypernym, level-1))
    return hypernym_name_list


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
            hypernym_list = _get_hypernyms(synset, level)
            for hypernym in hypernym_list:
                lemma_list.extend(hypernym.lemmas())
        return lemma_list
    elif word:
        for synset in wn.synsets(word):
            lemma_list = []
            if synset:
                hypernym_list = _get_hypernyms(synset, level)
                for hypernym in hypernym_list:
                    lemma_list.extend(hypernym.lemmas())
    return lemma_list


if __name__ == "__main__":
    # TEST_DIARY = "I like a banana. I really like an apple. I don't like a grape. I hate a sweet potato."
    # TEST_DIARY2 = """My main course was a half the dishes. Cumbul Ackard Cornish card little gym lettuce. Fresh Peas Mousser on mushrooms, Cocles and a cream sauce finished with a drizzle of olive oil wonderfully tender, and moist card. But I'm really intensify the flavor of the card there by providing a nice flavor contrast to the rich cream sauce. Lovely freshness, and texture from the little gym lettuce. A well executed dish with bags of flavour. Next, a very elegant vanilla, yogurt and strawberries and Candy Basil different strawberry preparations delivered a wonderful variety of flavor. Intensities is there was a sweet and tart lemon curd and yogurt sorbet buttery, Pepper Pastry Cramble Candied Lemons. Testing broken mrang the lemon curd had a wonderfully creamy texture and then ring was perfectly light and Chrissy and wonderful dessert with a great balance of flavors and textures. It's got sweetness. It's got scrunch. It's got acidity. It's got freshness."""
    # TEST_DIARY3 = "I like apples and bananas."
    # TEST_DIARY4 = "I don't like sweet potato. It makes me full!"
    # diary_tags = tagger.tag_pos_doc(TEST_DIARY)
    # diary_tags2 = tagger.tag_pos_doc(TEST_DIARY2)
    # diary_tags3 = tagger.tag_pos_doc(TEST_DIARY3)
    # diary_tags4 = tagger.tag_pos_doc(TEST_DIARY4)
    #
    # diary_tags = [[['I', 'PRP', '2', 'nsubj'], ['like', 'VBP', '0', 'root'], ['a', 'DT', '4', 'det'], ['banana', 'NN', '2', 'dobj'], ['.', None, None, None]], [['I', 'PRP', '3', 'nsubj'], ['really', 'RB', '3', 'advmod'], ['like', 'VBP', '0', 'root'], ['an', 'DT', '5', 'det'], ['apple', 'NN', '3', 'dobj'], ['.', None, None, None]], [['I', 'PRP', '4', 'nsubj'], ['do', 'VBP', '4', 'aux'], ["n't", 'RB', '4', 'neg'], ['like', 'VB', '0', 'root'], ['a', 'DT', '6', 'det'], ['grape', 'NN', '4', 'dobj'], ['.', None, None, None]], [['I', 'PRP', '2', 'nsubj'], ['hate', 'VBP', '0', 'root'], ['a', 'DT', '5', 'det'], ['sweet', 'JJ', '5', 'amod'], ['potato', 'NN', '2', 'dobj'], ['.', None, None, None]]]
    # diary_tags2 = [[['My', 'PRP$', '3', 'nmod:poss'], ['main', 'JJ', '3', 'amod'], ['course', 'NN', '6', 'nsubj'], ['was', 'VBD', '6', 'cop'], ['a', 'DT', '6', 'det'], ['half', 'NN', '0', 'root'], ['the', 'DT', '8', 'det'], ['dishes', 'NNS', '6', 'dep'], ['.', None, None, None]], [['Cumbul', 'NNP', '3', 'compound'], ['Ackard', 'NNP', '3', 'compound'], ['Cornish', 'NNP', '4', 'nsubj'], ['card', 'VBZ', '0', 'root'], ['little', 'JJ', '7', 'amod'], ['gym', 'NN', '7', 'compound'], ['lettuce', 'NN', '4', 'dobj'], ['.', None, None, None]], [['Fresh', 'NNP', '3', 'compound'], ['Peas', 'NNPS', '3', 'compound'], ['Mousser', 'NNP', '12', 'nsubj'], ['on', 'IN', '5', 'case'], ['mushrooms', 'NNS', '3', 'nmod'], [',', None, None, None], ['Cocles', 'NNP', '5', 'conj'], ['and', 'CC', '5', 'cc'], ['a', 'DT', '11', 'det'], ['cream', 'NN', '11', 'compound'], ['sauce', 'NN', '5', 'conj'], ['finished', 'VBD', '0', 'root'], ['with', 'IN', '15', 'case'], ['a', 'DT', '15', 'det'], ['drizzle', 'NN', '12', 'nmod'], ['of', 'IN', '20', 'case'], ['olive', 'JJ', '20', 'amod'], ['oil', 'NN', '20', 'compound'], ['wonderfully', 'NN', '20', 'compound'], ['tender', 'NN', '15', 'nmod'], [',', None, None, None], ['and', 'CC', '20', 'cc'], ['moist', 'NN', '24', 'compound'], ['card', 'NN', '20', 'conj'], ['.', None, None, None]], [['But', 'CC', '5', 'cc'], ['I', 'PRP', '5', 'nsubj'], ["'m", 'VBP', '5', 'aux'], ['really', 'RB', '5', 'advmod'], ['intensify', 'VBG', '0', 'root'], ['the', 'DT', '7', 'det'], ['flavor', 'NN', '5', 'dobj'], ['of', 'IN', '10', 'case'], ['the', 'DT', '10', 'det'], ['card', 'NN', '7', 'nmod'], ['there', 'RB', '5', 'advmod'], ['by', 'IN', '13', 'mark'], ['providing', 'VBG', '5', 'advcl'], ['a', 'DT', '17', 'det'], ['nice', 'JJ', '17', 'amod'], ['flavor', 'NN', '17', 'compound'], ['contrast', 'NN', '13', 'dobj'], ['to', 'TO', '22', 'case'], ['the', 'DT', '22', 'det'], ['rich', 'JJ', '22', 'amod'], ['cream', 'NN', '22', 'compound'], ['sauce', 'NN', '13', 'nmod'], ['.', None, None, None]], [['Lovely', 'NNP', '2', 'nsubj'], ['freshness', 'VBZ', '0', 'root'], [',', None, None, None], ['and', 'CC', '2', 'cc'], ['texture', 'NN', '2', 'conj'], ['from', 'IN', '10', 'case'], ['the', 'DT', '10', 'det'], ['little', 'JJ', '10', 'amod'], ['gym', 'NN', '10', 'compound'], ['lettuce', 'NN', '5', 'nmod'], ['.', None, None, None]], [['A', 'DT', '2', 'det'], ['well', 'NN', '3', 'nsubj'], ['executed', 'VBD', '0', 'root'], ['dish', 'NN', '3', 'dobj'], ['with', 'IN', '6', 'case'], ['bags', 'NNS', '3', 'nmod'], ['of', 'IN', '8', 'case'], ['flavour', 'NN', '6', 'nmod'], ['.', None, None, None]], [['Next', 'RB', '17', 'advmod'], [',', None, None, None], ['a', 'DT', '6', 'det'], ['very', 'RB', '6', 'advmod'], ['elegant', 'JJ', '6', 'dep'], ['vanilla', 'NN', '17', 'nsubj'], [',', None, None, None], ['yogurt', 'NN', '6', 'conj'], ['and', 'CC', '6', 'cc'], ['strawberries', 'NNS', '6', 'conj'], ['and', 'CC', '6', 'cc'], ['Candy', 'NNP', '13', 'compound'], ['Basil', 'NNP', '16', 'compound'], ['different', 'JJ', '16', 'amod'], ['strawberry', 'JJ', '16', 'amod'], ['preparations', 'NNS', '6', 'conj'], ['delivered', 'VBD', '0', 'root'], ['a', 'DT', '20', 'det'], ['wonderful', 'JJ', '20', 'amod'], ['variety', 'NN', '17', 'dobj'], ['of', 'IN', '22', 'case'], ['flavor', 'NN', '20', 'nmod'], ['.', None, None, None]], [['Intensities', 'NNS', '2', 'nsubj'], ['is', 'VBZ', '0', 'root'], ['there', 'EX', '4', 'expl'], ['was', 'VBD', '2', 'ccomp'], ['a', 'DT', '10', 'det'], ['sweet', 'JJ', '10', 'amod'], ['and', 'CC', '6', 'cc'], ['tart', 'JJ', '6', 'conj'], ['lemon', 'JJ', '10', 'amod'], ['curd', 'NN', '4', 'nsubj'], ['and', 'CC', '10', 'cc'], ['yogurt', 'NN', '14', 'compound'], ['sorbet', 'NN', '14', 'compound'], ['buttery', 'NN', '10', 'conj'], [',', None, None, None], ['Pepper', 'NNP', '20', 'compound'], ['Pastry', 'NNP', '20', 'compound'], ['Cramble', 'NNP', '20', 'compound'], ['Candied', 'NNP', '20', 'compound'], ['Lemons', 'NNP', '10', 'appos'], ['.', None, None, None]], [['Testing', 'NNP', '7', 'nsubj'], ['broken', 'VBN', '1', 'acl'], ['mrang', 'VBG', '2', 'xcomp'], ['the', 'DT', '6', 'det'], ['lemon', 'JJ', '6', 'amod'], ['curd', 'NN', '3', 'dobj'], ['had', 'VBD', '0', 'root'], ['a', 'DT', '11', 'det'], ['wonderfully', 'RB', '11', 'advmod'], ['creamy', 'JJ', '11', 'amod'], ['texture', 'NN', '7', 'dobj'], ['and', 'CC', '7', 'cc'], ['then', 'RB', '22', 'advmod'], ['ring', 'NN', '22', 'nsubj'], ['was', 'VBD', '22', 'cop'], ['perfectly', 'RB', '17', 'advmod'], ['light', 'JJ', '22', 'amod'], ['and', 'CC', '17', 'cc'], ['Chrissy', 'JJ', '17', 'conj'], ['and', 'CC', '19', 'cc'], ['wonderful', 'JJ', '19', 'conj'], ['dessert', 'NN', '7', 'conj'], ['with', 'IN', '26', 'case'], ['a', 'DT', '26', 'det'], ['great', 'JJ', '26', 'amod'], ['balance', 'NN', '22', 'nmod'], ['of', 'IN', '28', 'case'], ['flavors', 'NNS', '26', 'nmod'], ['and', 'CC', '28', 'cc'], ['textures', 'NNS', '28', 'conj'], ['.', None, None, None]], [['It', 'PRP', '3', 'nsubjpass'], ["'s", 'VBZ', '3', 'auxpass'], ['got', 'VBN', '0', 'root'], ['sweetness', 'NN', '3', 'dobj'], ['.', None, None, None]], [['It', 'PRP', '3', 'nsubjpass'], ["'s", 'VBZ', '3', 'auxpass'], ['got', 'VBN', '0', 'root'], ['scrunch', 'RB', '3', 'advmod'], ['.', None, None, None]], [['It', 'PRP', '3', 'nsubjpass'], ["'s", 'VBZ', '3', 'auxpass'], ['got', 'VBN', '0', 'root'], ['acidity', 'RB', '3', 'advmod'], ['.', None, None, None]], [['It', 'PRP', '3', 'nsubjpass'], ["'s", 'VBZ', '3', 'auxpass'], ['got', 'VBN', '0', 'root'], ['freshness', 'NN', '3', 'dobj'], ['.', None, None, None]]]
    # diary_tags3 = [[['I', 'PRP', '2', 'nsubj'], ['like', 'VBP', '0', 'root'], ['apples', 'NNS', '2', 'dobj'], ['and', 'CC', '3', 'cc'], ['bananas', 'NNS', '3', 'conj'], ['.', None, None, None]]]
    # diary_tags4 = [[['I', 'PRP', '4', 'nsubj'], ['do', 'VBP', '4', 'aux'], ["n't", 'RB', '4', 'neg'], ['like', 'VB', '0', 'root'], ['sweet', 'JJ', '6', 'amod'], ['potato', 'NN', '4', 'dobj'], ['.', None, None, None]], [['It', 'PRP', '2', 'nsubj'], ['makes', 'VBZ', '0', 'root'], ['me', 'PRP', '4', 'nsubj'], ['full', 'JJ', '2', 'xcomp'], ['!', None, None, None]]]

    # tend_analyzer.analyze_diary([diary_tags])
    # tend_analyzer.analyze_diary([diary_tags, diary_tags2, diary_tags3, diary_tags4])
    # tend_analyzer.analyze_diary([diary_tags, diary_tags3, diary_tags4])
    # tend_analyzer.analyze_diary([diary_tags, diary_tags3])
    # tend_analyzer.analyze_diary([diary_tags3])

    JENIIFER_DIARY = [
    ]

    # for i in range(0, len(JENIIFER_DIARY)):
    #     diary = JENIIFER_DIARY[i]
    #     print("start tagging diary #%s" % i)
    #     diary_tags = tagger.tag_pos_doc(diary, True)
    #     print("create piclke for tags of diary #%s" % i)
    #     tagger.tags_to_pickle(diary_tags, "pickles/jennifer" + str(i) + ".pkl")
    #     print()

    tend_analyzer = TendencyAnalyzer()

    # joanne_diaries = list()
    # for i in range(0, 30):
    #     diary_tags = tagger.pickle_to_tags("pickles/joanne" + str(i) + ".pkl")
    #     joanne_diaries.append(diary_tags[1])
    # print("load joanne diaries done.")
    # tend_analyzer.analyze_diary(joanne_diaries)

    # jeniffer_diaries = list()
    # for i in range(0, 36):
    #     diary_tags = tagger.pickle_to_tags("pickles/jennifer" + str(i) + ".pkl")
    #     jeniffer_diaries.append(diary_tags[1])
    # print("load jeniffer diaries done.")
    # tend_analyzer.analyze_diary(jeniffer_diaries,
    #         [('food', 'thing'), ('restaurant', 'thing'), ('weather', 'thing'),
    #          ('hobby', 'activity'), ('exercise', 'activity')])

    # smiley_diaries = list()
    # for i in range(0, 50):
    #     diary_tags = tagger.pickle_to_tags("pickles/smiley" + str(i) + ".pkl")
    #     smiley_diaries.append(diary_tags[1])
    # print("load smiley diaries done.")
    # tend_analyzer.analyze_diary(smiley_diaries,
    #         [('food', 'thing'), ('restaurant', 'thing'), ('weather', 'thing'),
    #          ('hobby', 'activity'), ('exercise', 'activity')])

    d_diaries = list()
    for i in range(0, 40):
        diary_tags = tagger.pickle_to_tags("pickles/diary_d" + str(i) + ".pkl")
        d_diaries.append(diary_tags[1])
    print("load D diaries done.")
    tend_analyzer.analyze_diary(d_diaries,
            [('food', 'thing'), ('restaurant', 'thing'), ('weather', 'thing'),
             ('hobby', 'activity'), ('exercise', 'activity')])

    # elize_diaries = list()
    # for i in range(0, 4):
    #     diary_tags = tagger.pickle_to_tags("pickles/eliz" + str(i) + ".pkl")
    #     elize_diaries.append(diary_tags[1])
    # print("load eliz diaries done.")
    # tend_analyzer.analyze_diary(elize_diaries, [('food', 'thing')])

    # TEST_DIARY4 = "I am having chili for supper. I have an apple."
    # diary_tags4 = tagger.tag_pos_doc(TEST_DIARY4)
    # tend_analyzer.analyze_diary([diary_tags4[1]])

