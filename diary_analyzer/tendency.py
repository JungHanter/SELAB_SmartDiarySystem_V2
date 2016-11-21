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
            if synset.name() is item[self.IDX_SYNSET].name():
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
    def __init__(self, *root_synsets, max_level=1):
        if type(root_synsets[0]) is list or type(root_synsets[0]) is tuple:
            root_synsets = root_synsets[0]
        self.synset_list = list()
        for synset in root_synsets:
            self.synset_list += self._collect_hyponyms(synset, max_level)

    def _collect_hyponyms(self, root_synset, max_level=1, now_level=0):
        if now_level == max_level: return []
        now_level += 1
        hyponym_list = list()
        for hyponym in root_synset.hyponyms():
            # names, counts = _lemmas_to_name_list(hyponym.lemmas(), True)
            # hyponym_list.append((hyponym, max_level, names, counts))

            # tuple (sysnet, level, lemma_words)
            hyponym_list.append((hyponym, now_level, _lemmas_to_name_list(hyponym.lemmas())))
            hyponym_list = hyponym_list + self._collect_hyponyms(hyponym, max_level, now_level)
        return hyponym_list


class HypernymRetriever(WordSetRetriever):
    def __init__(self, *root_synsets, max_level=1):
        if type(root_synsets[0]) is list or type(root_synsets[0]) is tuple:
            root_synsets = root_synsets[0]
        self.synset_list = list()
        for synset in root_synsets:
            self.synset_list += self._collect_hyponyms(synset, max_level)

    def _collect_hyponyms(self, root_synset, max_level=1, now_level=0):
        if now_level == max_level: return []
        now_level += 1
        hypernym_list = list()
        for hypernym in root_synset.hypernyms():
            # names, counts = _lemmas_to_name_list(hyponym.lemmas(), True)
            # hyponym_list.append((hyponym, max_level, names, counts))

            # tuple (sysnet, level, lemma_words)
            hypernym_list.append((hypernym, now_level, _lemmas_to_name_list(hypernym.lemmas())))
            hypernym_list = hypernym_list + self._collect_hyponyms(hypernym, max_level, now_level)
        return hypernym_list

    def get_list(self):
        return self.synset_list


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
            'objectivity': 1
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

                score_dict = {
                    'positivity': score_pos,
                    'negativity': score_neg,
                    'objectivity': score_obj,
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

    SIMILAR_PATH_MAX_HYPERNYM = 3
    SIMILAR_PATH_MAX_HYPONYM = 1

    CLUSTER_CUT_DIST = 1.5

    def __init__(self, senti_wordnet):
        self.senti_wordnet = senti_wordnet
        self.words_sets = dict()
        # and more word set ...

    def add_word_set(self, target, target_type, words_set):
        if words_set:
            self.words_sets[(target, target_type)] = words_set

    # input: a list of tagged diary
    def analyze_diary(self, diary_tags_list):
        pref_ta_list = list()

        for diary_tags in diary_tags_list:
            # step 2
            identified_sent_dict = self._identify_sentences(diary_tags)
            pprint(identified_sent_dict)
            print()

            # step 3
            scores_pref = self._compute_pref_scores(diary_tags, identified_sent_dict)
            pprint(scores_pref)
            print()

            # step 4
            scores_pref = self._compute_pref_scores_of_similars(scores_pref)
            # pprint(scores_pref)
            # print()

            # convert & add scores_pref dictionary to list
            scores_pref_list = self._convert_scores_dict_to_list(scores_pref)
            pref_ta_list += scores_pref_list

        # step 5
        clusters, pref_num = self._perform_clustering(pref_ta_list)
        print(len(clusters))
        pprint(clusters)
        print()

        # step 6
        pos_results, neg_result = self._figure_out_best_ta(clusters, len(diary_tags_list), pref_num)
        pprint(pos_results)
        pprint(neg_result)
        print()

        return pos_results, neg_result

    ###############################################################################
    # Step 2. Identifying Sentences including Things and Activities in Each Diary #
    ###############################################################################
    def _identify_sentences(self, diary_tags):
        identified_sent_dict = defaultdict(lambda: list())
        for sent_idx in range(0, len(diary_tags)):
            tagged_sent = diary_tags[sent_idx]

            prev_word_list = [] # for compound word
            for word_idx in range(0, len(tagged_sent)):
                word = tagged_sent[word_idx]

                # a word have pos and role
                if word[TAG_WORD_POS] is not None and word[TAG_WORD_ROLE] is not None:
                    if 'VB' in word[TAG_WORD_POS]:
                        # find activities as verb
                        found_synset_list \
                            = self._find_synsets_in_wordsets([word[TAG_WORD]])
                        for found_synset in found_synset_list:
                            identified_sent_dict[sent_idx].append(found_synset + (word_idx, 'v'))
                        prev_word_list.clear()

                    elif word[TAG_WORD_POS].startswith('NN') and \
                            ('subj' in word[TAG_WORD_ROLE] or 'obj' in word[TAG_WORD_ROLE] or
                                 word[TAG_WORD_ROLE is 'conj']):
                        prev_word_list.append(word[TAG_WORD])

                        # check the noun is plural
                        plural = False
                        if word[TAG_WORD_POS].endswith('S'):
                            plural = True

                        # find things and activities as noun
                        found_synset_list \
                            = self._find_synsets_in_wordsets(prev_word_list, plural)
                        for found_synset in found_synset_list:
                            identified_sent_dict[sent_idx].append(found_synset + (word_idx, 'n'))
                        prev_word_list.clear()

                    # For the compound words
                    elif (word[TAG_WORD_POS].startswith('JJ') and word[TAG_WORD_ROLE] == 'amod') or \
                            (word[TAG_WORD_POS].startswith('NN') and word[TAG_WORD_ROLE] == 'compound'):
                        prev_word_list.append(word[TAG_WORD])
        return identified_sent_dict

    ###############################################################################
    # Step 3. Computing Preference Scores of Things and Activities in Each Diary  #
    ###############################################################################
    def _compute_pref_scores(self, diary_sent_list, identified_sent_dict):
        scores_pref = dict()
        scores_pref_sent = defaultdict(lambda: {'score': 0.0, 'count': 0, 'type': None})

        # analyze sentence to compute preference score
        for sent_idx, identified_info_list in identified_sent_dict.items():
            tagged_sent = diary_sent_list[sent_idx]

            is_sent_mine = False  # 'I' is subject of sentence
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
                            senti_score = self.senti_wordnet.get_score_value(offset, 'v')
                            if senti_score == 0:
                                continue
                            else:
                                sent_score += senti_score
                                word_cnt += 1
                        if word_cnt == 0:
                            sent_score = 0
                        else:
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
                        for offset in offsets:
                            senti_score = self.senti_wordnet.get_score_value(offset, 'a')
                            if senti_score == 0:
                                continue
                            else:
                                sent_score += senti_score
                                word_cnt += 1
                        if word_cnt == 0:
                            sent_score = 0
                        else:
                            #average of sent score of sentiment word
                            sent_score = sent_score / word_cnt
                        if sent_score is not 0:
                            weight_sent += adjective_weight * sent_score
                            count_sent += 1

            # apply preference score of things and activities in the sentence
            for synset_name, word_weight in weights_word.items():
                if self.DEBUG:
                    print('================')
                    print('word_name: %s' % synset_name)
                    print('word_score: %s' % word_weight)
                # calculate weight of subjectivity
                weight_subj = 1 if is_sent_mine else (0.8 if is_sent_subj else 0.1)
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
                if self.DEBUG: print('adverb_score: %s' % adverb_score)
                if self.DEBUG: print('adverb_neg_score %s' % adverb_neg_score)

                score = 0
                if sent_score >= 0:
                    score = (word_weight['weight'] * 0.5) + (sent_score * word_weight['weight'] * 0.5)
                else:
                    score = -(word_weight['weight'] * 0.5) + (sent_score * word_weight['weight'] * 0.5)
                if self.DEBUG: print('word score with sent: %s' % score)
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
            scores_pref[synset_name]['score'] = sent_dict['score'] / sent_dict['count']
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

            # find hypernyms and hyponyms and score to them
            hypernyms = HypernymRetriever(wn.synset(synset_name), max_level=self.SIMILAR_PATH_MAX_HYPERNYM).get_list()
            for hypernym in hypernyms:
                scores_pref_rel[hypernym[0].name()]['score'] += score_dict['score'] / (hypernym[1]+1)
                scores_pref_rel[hypernym[0].name()]['count'] += 1
                if not scores_pref_rel[hypernym[0].name()]['type']:
                    scores_pref_rel[hypernym[0].name()]['type'] = score_dict['type']
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
            if u[3] == v[3]:    # is same type? (thing and activity)
                if u[2] == v[2]:    # is same thing or activity?
                    path_dist = wn.synset(u[0]).path_similarity(wn.synset(v[0]))
                    if path_dist is None:
                        path_dist = 0
                    dist = 1 - (path_dist * 1.5)
                    # pref_dist = abs(u[1] - v[1])
                    # dist = path_dist * pref_dist
                    return dist
                else:
                    return 3
            else:
                return 5

        # feature selection
        # filtering with things and activities which there is only one item
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

        # sort list for debugging
        if self.DEBUG:
            pref_ta_list = sorted(pref_ta_list, key=lambda pref_ta: pref_ta[0])

        # print list for debug
        if self.DEBUG:
            print("pref_ta_list: ")
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
            print("hac_result: ")
            print(hac_result)
            print()

        # figure out the number of clusters (determine where to cut tree)
        num_cluster = 1
        for matrix_y in hac_result:
            if matrix_y[2] > self.CLUSTER_CUT_DIST:
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
        #                 str(int(pref_ta_list[i][1]*100000)/100000.0)
        # ct = hac_result[-(num_cluster - 1), 2]
        #
        # p = hac.dendrogram(hac_result, labels=labels, color_threshold=ct)
        # plt.show()

        return clusters, pref_num

    ###############################################################################
    # Step 6. Figuring out Things and Activities having the Best Preference Score #
    ###############################################################################
    def _figure_out_best_ta(self, clusters, diary_num, pref_num):
        pos_ta_cluster_dict = dict()
        neg_ta_cluster_dict = dict()

        min_cluter_item = diary_num / 14.0
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

            # find things and activities ta having max prefs
            first_ta = cluster[0]
            max_pos_pref_ta_list = [first_ta]
            max_pos_pref_score = first_ta[1]
            max_neg_pref_ta_list = [first_ta]
            max_neg_pref_score = first_ta[1]
            for idx_ta in range(1, len(cluster)):
                ta = cluster[idx_ta]
                if ta[1] >= 0:
                    if max_pos_pref_score == ta[1]:
                        max_pos_pref_ta_list.append(ta)
                    elif max_pos_pref_score < ta[1]:
                        max_pos_pref_score = ta[1]
                        max_pos_pref_ta_list = [ta]
                else:
                    if max_neg_pref_score == ta[1]:
                        max_neg_pref_ta_list.append(ta)
                    elif max_neg_pref_score > ta[1]:
                        max_neg_pref_score = ta[1]
                        max_neg_pref_ta_list = [ta]

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
                        # add little score which is from other items in the same cluster
                        pos_ta[1] += 0.1 * wn.synset(pos_ta[0]).path_similarity(wn.synset(ta[0])) * ta[1]
            neg_pref_ta_list_cvt = list()
            for neg_ta in max_neg_pref_ta_list:
                neg_pref_ta_list_cvt.append(list(neg_ta))
            for neg_ta in neg_pref_ta_list_cvt:
                for ta in cluster:
                    if ta is not neg_ta:
                        # add little score which is from other items in the same cluster
                        neg_ta[1] += 0.1 * wn.synset(neg_ta[0]).path_similarity(wn.synset(ta[0])) * ta[1]

            # sometimes, final preference score is more than 1 because
            # there are many related ta or the ta is refered in diaries as much times..
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
            pprint(neg_ta_cluster_dict)
            print()

        # arrange things and activities for their type
        clsfied_pos_scores_dict = defaultdict(lambda: list())
        for ta_name, ta in pos_ta_cluster_dict.items():
            if ta[1] >= 0.2:   # preference score is more than
                clsfied_pos_scores_dict[(ta[2], ta[3])].append((ta[0], ta[1]))
        clsfied_neg_scores_dict = defaultdict(lambda: list())
        for ta_name, ta in neg_ta_cluster_dict.items():
            if ta[1] <= -0.2:
                clsfied_neg_scores_dict[(ta[2], ta[3])].append((ta[0], ta[1]))

        if self.DEBUG:
            pprint(clsfied_pos_scores_dict)
            pprint(clsfied_neg_scores_dict)
            print()

        # trim up to 5 first element as best score
        best_pos_scores_dict = dict()
        for ta_type, ta_list in clsfied_pos_scores_dict.items():
            best_pos_scores_dict[ta_type] = sorted(ta_list, key=lambda ta: ta[1], reverse=True)[:5]
        best_neg_scores_dict = dict()
        for ta_type, ta_list in clsfied_neg_scores_dict.items():
            best_neg_scores_dict[ta_type] = sorted(ta_list, key=lambda ta: ta[1])[:5]

        if self.DEBUG:
            pprint(best_pos_scores_dict)
            pprint(best_neg_scores_dict)
            print()

        return best_pos_scores_dict, best_neg_scores_dict

    # return a list of (found_synset, lemma_word, words_set_type)
    def _find_synsets_in_wordsets(self, finding_word_list, plural=False):
        synset_list = list()
        for words_set_type, words_set in self.words_sets.items():
            found_synset, lemma_word = \
                self._find_synset_by_word_list(words_set, finding_word_list, plural)
            if found_synset:
                synset_list.append((found_synset, lemma_word, words_set_type))
        return synset_list

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
            pref_ta = (synset_name, score_dict['score'],
                       score_dict['type'][0], score_dict['type'][1])
            pref_ta_list.append(pref_ta)
        return pref_ta_list


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


if __name__ == "__main__":
    #######################################################################################
    # Step 1. Retrieving Word Sets about Things, Activities, and Preferences from Corpora #
    #######################################################################################
    sw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordset',
                           'SentiWordNet_3.0.0_20130122.txt')
    senti_wordnet = SentiWordNet(sw_path)
    foods = HyponymRetriever(wn.synset('food.n.02'), wn.synset('food.n.01'), max_level=10)
    restaurants = HyponymRetriever(wn.synset('restaurant.n.01'), max_level=8)
    weathers = HyponymRetriever(wn.synset('weather.n.01'), max_level=8)
    exercises = HyponymRetriever(wn.synset('sport.n.01'), wn.synset('sport.n.02'),
                                 wn.synset('exercise.n.01'), wn.synset('exercise.v.03'),
                                 wn.synset('exercise.v.04'), ax_level=12)
    activities = HypernymRetriever(wn.synset('activity.n.01'), wn.synset('action.n.02'),
                                   wn.synset('natural_process.n.01'), wn.synset('act.n.01'),
                                   wn.synset('act.n.02'), wn.synset('act.n.05'),
                                   wn.synset('act.v.01'), wn.synset('act.v.02'), wn.synset('act.v.03'),
                                   wn.synset('act.v.05'), wn.synset('work.v.03'),
                                   wn.synset('act.v.08'), wn.synset('dissemble.v.03'), max_level=16)

    tend_analyzer = TendencyAnalyzer(senti_wordnet)
    tend_analyzer.add_word_set('food', 'thing', foods)
    tend_analyzer.add_word_set('restaurant', 'thing', restaurants)
    tend_analyzer.add_word_set('weather', 'thing', weathers)
    tend_analyzer.add_word_set('exercise', 'activity', exercises)
    tend_analyzer.add_word_set('all', 'activity', activities)

    # TEST_DIARY = "I like a banana. I really like an apple. I don't like a grape. I hate a sweet potato."
    # TEST_DIARY2 = """My main course was a half the dishes. Cumbul Ackard Cornish card little gym lettuce. Fresh Peas Mousser on mushrooms, Cocles and a cream sauce finished with a drizzle of olive oil wonderfully tender, and moist card. But I'm really intensify the flavor of the card there by providing a nice flavor contrast to the rich cream sauce. Lovely freshness, and texture from the little gym lettuce. A well executed dish with bags of flavour. Next, a very elegant vanilla, yogurt and strawberries and Candy Basil different strawberry preparations delivered a wonderful variety of flavor. Intensities is there was a sweet and tart lemon curd and yogurt sorbet buttery, Pepper Pastry Cramble Candied Lemons. Testing broken mrang the lemon curd had a wonderfully creamy texture and then ring was perfectly light and Chrissy and wonderful dessert with a great balance of flavors and textures. It's got sweetness. It's got scrunch. It's got acidity. It's got freshness."""
    # TEST_DIARY3 = "I like apples and bananas."
    # diary_tags = tagger.tag_pos_doc(TEST_DIARY)


    # diary_tags2 = tagger.tag_pos_doc(TEST_DIARY2)
    # diary_tags3 = tagger.tag_pos_doc(TEST_DIARY3)

    # diary_tags = [[['I', 'PRP', '2', 'nsubj'], ['like', 'VBP', '0', 'root'], ['a', 'DT', '4', 'det'], ['banana', 'NN', '2', 'dobj'], ['.', None, None, None]], [['I', 'PRP', '3', 'nsubj'], ['really', 'RB', '3', 'advmod'], ['like', 'VBP', '0', 'root'], ['an', 'DT', '5', 'det'], ['apple', 'NN', '3', 'dobj'], ['.', None, None, None]], [['I', 'PRP', '4', 'nsubj'], ['do', 'VBP', '4', 'aux'], ["n't", 'RB', '4', 'neg'], ['like', 'VB', '0', 'root'], ['a', 'DT', '6', 'det'], ['grape', 'NN', '4', 'dobj'], ['.', None, None, None]], [['I', 'PRP', '2', 'nsubj'], ['hate', 'VBP', '0', 'root'], ['a', 'DT', '5', 'det'], ['sweet', 'JJ', '5', 'amod'], ['potato', 'NN', '2', 'dobj'], ['.', None, None, None]]]
    # diary_tags2 = [[['My', 'PRP$', '3', 'nmod:poss'], ['main', 'JJ', '3', 'amod'], ['course', 'NN', '6', 'nsubj'], ['was', 'VBD', '6', 'cop'], ['a', 'DT', '6', 'det'], ['half', 'NN', '0', 'root'], ['the', 'DT', '8', 'det'], ['dishes', 'NNS', '6', 'dep'], ['.', None, None, None]], [['Cumbul', 'NNP', '3', 'compound'], ['Ackard', 'NNP', '3', 'compound'], ['Cornish', 'NNP', '4', 'nsubj'], ['card', 'VBZ', '0', 'root'], ['little', 'JJ', '7', 'amod'], ['gym', 'NN', '7', 'compound'], ['lettuce', 'NN', '4', 'dobj'], ['.', None, None, None]], [['Fresh', 'NNP', '3', 'compound'], ['Peas', 'NNPS', '3', 'compound'], ['Mousser', 'NNP', '12', 'nsubj'], ['on', 'IN', '5', 'case'], ['mushrooms', 'NNS', '3', 'nmod'], [',', None, None, None], ['Cocles', 'NNP', '5', 'conj'], ['and', 'CC', '5', 'cc'], ['a', 'DT', '11', 'det'], ['cream', 'NN', '11', 'compound'], ['sauce', 'NN', '5', 'conj'], ['finished', 'VBD', '0', 'root'], ['with', 'IN', '15', 'case'], ['a', 'DT', '15', 'det'], ['drizzle', 'NN', '12', 'nmod'], ['of', 'IN', '20', 'case'], ['olive', 'JJ', '20', 'amod'], ['oil', 'NN', '20', 'compound'], ['wonderfully', 'NN', '20', 'compound'], ['tender', 'NN', '15', 'nmod'], [',', None, None, None], ['and', 'CC', '20', 'cc'], ['moist', 'NN', '24', 'compound'], ['card', 'NN', '20', 'conj'], ['.', None, None, None]], [['But', 'CC', '5', 'cc'], ['I', 'PRP', '5', 'nsubj'], ["'m", 'VBP', '5', 'aux'], ['really', 'RB', '5', 'advmod'], ['intensify', 'VBG', '0', 'root'], ['the', 'DT', '7', 'det'], ['flavor', 'NN', '5', 'dobj'], ['of', 'IN', '10', 'case'], ['the', 'DT', '10', 'det'], ['card', 'NN', '7', 'nmod'], ['there', 'RB', '5', 'advmod'], ['by', 'IN', '13', 'mark'], ['providing', 'VBG', '5', 'advcl'], ['a', 'DT', '17', 'det'], ['nice', 'JJ', '17', 'amod'], ['flavor', 'NN', '17', 'compound'], ['contrast', 'NN', '13', 'dobj'], ['to', 'TO', '22', 'case'], ['the', 'DT', '22', 'det'], ['rich', 'JJ', '22', 'amod'], ['cream', 'NN', '22', 'compound'], ['sauce', 'NN', '13', 'nmod'], ['.', None, None, None]], [['Lovely', 'NNP', '2', 'nsubj'], ['freshness', 'VBZ', '0', 'root'], [',', None, None, None], ['and', 'CC', '2', 'cc'], ['texture', 'NN', '2', 'conj'], ['from', 'IN', '10', 'case'], ['the', 'DT', '10', 'det'], ['little', 'JJ', '10', 'amod'], ['gym', 'NN', '10', 'compound'], ['lettuce', 'NN', '5', 'nmod'], ['.', None, None, None]], [['A', 'DT', '2', 'det'], ['well', 'NN', '3', 'nsubj'], ['executed', 'VBD', '0', 'root'], ['dish', 'NN', '3', 'dobj'], ['with', 'IN', '6', 'case'], ['bags', 'NNS', '3', 'nmod'], ['of', 'IN', '8', 'case'], ['flavour', 'NN', '6', 'nmod'], ['.', None, None, None]], [['Next', 'RB', '17', 'advmod'], [',', None, None, None], ['a', 'DT', '6', 'det'], ['very', 'RB', '6', 'advmod'], ['elegant', 'JJ', '6', 'dep'], ['vanilla', 'NN', '17', 'nsubj'], [',', None, None, None], ['yogurt', 'NN', '6', 'conj'], ['and', 'CC', '6', 'cc'], ['strawberries', 'NNS', '6', 'conj'], ['and', 'CC', '6', 'cc'], ['Candy', 'NNP', '13', 'compound'], ['Basil', 'NNP', '16', 'compound'], ['different', 'JJ', '16', 'amod'], ['strawberry', 'JJ', '16', 'amod'], ['preparations', 'NNS', '6', 'conj'], ['delivered', 'VBD', '0', 'root'], ['a', 'DT', '20', 'det'], ['wonderful', 'JJ', '20', 'amod'], ['variety', 'NN', '17', 'dobj'], ['of', 'IN', '22', 'case'], ['flavor', 'NN', '20', 'nmod'], ['.', None, None, None]], [['Intensities', 'NNS', '2', 'nsubj'], ['is', 'VBZ', '0', 'root'], ['there', 'EX', '4', 'expl'], ['was', 'VBD', '2', 'ccomp'], ['a', 'DT', '10', 'det'], ['sweet', 'JJ', '10', 'amod'], ['and', 'CC', '6', 'cc'], ['tart', 'JJ', '6', 'conj'], ['lemon', 'JJ', '10', 'amod'], ['curd', 'NN', '4', 'nsubj'], ['and', 'CC', '10', 'cc'], ['yogurt', 'NN', '14', 'compound'], ['sorbet', 'NN', '14', 'compound'], ['buttery', 'NN', '10', 'conj'], [',', None, None, None], ['Pepper', 'NNP', '20', 'compound'], ['Pastry', 'NNP', '20', 'compound'], ['Cramble', 'NNP', '20', 'compound'], ['Candied', 'NNP', '20', 'compound'], ['Lemons', 'NNP', '10', 'appos'], ['.', None, None, None]], [['Testing', 'NNP', '7', 'nsubj'], ['broken', 'VBN', '1', 'acl'], ['mrang', 'VBG', '2', 'xcomp'], ['the', 'DT', '6', 'det'], ['lemon', 'JJ', '6', 'amod'], ['curd', 'NN', '3', 'dobj'], ['had', 'VBD', '0', 'root'], ['a', 'DT', '11', 'det'], ['wonderfully', 'RB', '11', 'advmod'], ['creamy', 'JJ', '11', 'amod'], ['texture', 'NN', '7', 'dobj'], ['and', 'CC', '7', 'cc'], ['then', 'RB', '22', 'advmod'], ['ring', 'NN', '22', 'nsubj'], ['was', 'VBD', '22', 'cop'], ['perfectly', 'RB', '17', 'advmod'], ['light', 'JJ', '22', 'amod'], ['and', 'CC', '17', 'cc'], ['Chrissy', 'JJ', '17', 'conj'], ['and', 'CC', '19', 'cc'], ['wonderful', 'JJ', '19', 'conj'], ['dessert', 'NN', '7', 'conj'], ['with', 'IN', '26', 'case'], ['a', 'DT', '26', 'det'], ['great', 'JJ', '26', 'amod'], ['balance', 'NN', '22', 'nmod'], ['of', 'IN', '28', 'case'], ['flavors', 'NNS', '26', 'nmod'], ['and', 'CC', '28', 'cc'], ['textures', 'NNS', '28', 'conj'], ['.', None, None, None]], [['It', 'PRP', '3', 'nsubjpass'], ["'s", 'VBZ', '3', 'auxpass'], ['got', 'VBN', '0', 'root'], ['sweetness', 'NN', '3', 'dobj'], ['.', None, None, None]], [['It', 'PRP', '3', 'nsubjpass'], ["'s", 'VBZ', '3', 'auxpass'], ['got', 'VBN', '0', 'root'], ['scrunch', 'RB', '3', 'advmod'], ['.', None, None, None]], [['It', 'PRP', '3', 'nsubjpass'], ["'s", 'VBZ', '3', 'auxpass'], ['got', 'VBN', '0', 'root'], ['acidity', 'RB', '3', 'advmod'], ['.', None, None, None]], [['It', 'PRP', '3', 'nsubjpass'], ["'s", 'VBZ', '3', 'auxpass'], ['got', 'VBN', '0', 'root'], ['freshness', 'NN', '3', 'dobj'], ['.', None, None, None]]]
    # diary_tags3 = [[['I', 'PRP', '2', 'nsubj'], ['like', 'VBP', '0', 'root'], ['apples', 'NNS', '2', 'dobj'], ['and', 'CC', '3', 'cc'], ['bananas', 'NNS', '3', 'conj'], ['.', None, None, None]]]

    # tend_analyzer.analyze_diary([diary_tags])
    # tend_analyzer.analyze_diary([diary_tags, diary_tags2, diary_tags3])
    # tend_analyzer.analyze_diary([diary_tags, diary_tags3])
    # tend_analyzer.analyze_diary([diary_tags3])

    # JENNIFER_DIARY = [
    #     """I've had many diaries in the past. This isn't the first, and it probably won't be the last, but this one will be different. I'm hoping that today will be the first day of the rest of my life. So much has happened I feel I've lost myself in the process and I've got to try and find myself, maybe for the first time in my life. I've heard that writing things down can be theraputic, so I'm banking on that. I need therapy. I've been struggling to deal with the abuse I suffered as a child. I was repeatedly molested and raped by my stepfather, until I became pregnant when I was sixteen. Now I'm struggling to raise the now six year old product of that abuse, on top of taking care of a new baby and a husband. On the outside, my life looks happy and normal, but I'm anything but. I feel lost. I don't know what I want to do with my life or who I want to be. I've been trying to hide for so long that now I can't seem to find me. I struggle with emotional eating and now at 212 pounds I NEED to make a change. I hate the way I look and feel. My husband is incredibly loving and understanding, but I can't burden him with these feelings and in the interest of my mental sanity, I figured a diary would be good. I chose to have an online diary that people could read, because I want someone else to read it and understand that people have problems and they can deal with them. I've been dealing with depression on top of everything else and I'm just sick of things the way they are. I'm going to change my life and maybe I can hold myself accountable instead of blaming everyone else for my unhappiness.""",
    #     """Here I am, awake, its 1am, everyone is asleep and I'm awake. I just finished ironing my slacks and Snookums shirt and pants for the hearing. I spent 20 minutes waxing my eyebrows and making sure my skin was exfoliated and I look clean, well-groomed, young and healthy. Why am I so obsessed with how I'm going to look tomorrow? I don't usually give a crap how I look, but it's so important to me that I look like I'm doing well, even if I'm not. Appearances are important in some instances. I even filed and painted my nails and cleaned my jewelry. I'm all set. Now if I could just get some sleep. Sleep deprivation doesn't look so healthy, now does it? I'm going to be running around like a chicken with my head cut off in the morning, because I'm going to be exhausted, I'm going to sleep too long, and I won't leave myself enough time to get ready. It is an hour and a half drive and a 30 minute ferry ride to get to Coupeville. I'm notorious for not leaving enough travel time. So, let me get my butt to bed before I kick my own ass in the morning.""",
    #     """This morning on Good Morning America I watched President Bush walk through Biloxi, Mississippi and I couldn't help but think it was far too little, too late. The leader of the NAACP said that he thinks that the reason aid isn't getting to the region quickly is because 99% of those affected and who couldn't evacuate are African American. I hate to think there might be some truth in that, but being of African American decent myself, I can't help but entertain the idea. I'v faced discrimination in my time, but this is no time for old feelings to resurface. I guess back in the 1940's Louisiana had a massive flood and they rounded up all the black people at gun point and used them as sandbags to stop the flood waters. I'm not even sure how that's possible and I don't want to think about it. In the defense of the government, though. I do admit that my "people" can be very ignorant and self-destructive. No matter how bad the situation in New Orleans, they need to bring out the best in themselves, not the worst. What the hell is the point of looting an electronics store, when there isn't any electricity, a house to put it in, and there won't be for months? Ignorant, I tell you. Well, that's my current events opinion for the week. I don't much like talking about current events, but this one is closer to home for me, since my family is originally from Louisiana. We date back all the way to the freed slaves and Native Americans of Louisiana. I suppose you could say, my history begins there, even though I haven't been since I was a small child. I still feel as though I've lost something important.""",
    #     """In less than 3 hours, I'll be saying goodbye to Snookums. I'm making a big deal out of it now, but I know that when he does actually leave, it will be business as usual around here. Minus one. I can tell that he's not happy about leaving. Everything he's done in the past 24 hours, he's done like it's for the last time. But on the other hand, I can sense some excitement on his part. He's looking forward to making such an important step in his career. I'm excited for him. Since I've known him, he's complained about always being at the bottom of the food chain (as far as the chain of command goes). Now, he's climbed a few rungs closer to the top. His Commanding Officer told him that this trip is an educational experience for him and that when he gets back, he'll be taking over part of the Navigation division while his First Class takes leave to get married and then ultimately for good when his First Class transfers next summer. Snookums is proud of himself, but scared at the same time. I'm proud of him, too. It's nice seeing him have pride in his career, again.I'm going to make him a good breakfast before he leaves. I'm starving myself, so I better get on it! Ciao for now.""",
    #     """OH, HAPPY DAY! I've had a magnificent day! I thought aside from reaching 197 pounds, today couldn't get any better than it already was, but it did. Genevieve came over around noon and we curled up on the couch and chatted, flipped through Avon brochures (she's an Avon rep) and caught each other up on what's been going on in our lives. We did that for about 4 hours, then we went to pick Annie up from daycare. I'd already had a fabulous day, but it only got better when we got home. When I walked into the living room I noticed that there were 2 new messages on the answering machine. I hit play and the marvelous sound of my Snookums' voice came on, saying that he'd just landed at SeaTac and he was on his way home! The second message was him saying he was even closer to home! He wasn't due home until late tomorrow evening. To make a long story short: Gen stayed here and watched the girls while I went to the airporter station to pick Snookums up. We went out to dinner (there was no way I was cooking tonight) and now he's sitting in the bedroom playing with our new cellphones (T-mobile FedExed them to us earlier this week, not important information though). If this entry seems abridged that's because I'm in a hurry to get off. I'm going to go cuddle up with my Snookums and take in his scent and warmth; and anything else that comes my way :o :) Ciao until I roll my happy ass out of bed tomorrow morning!!!""",
    #     """Kiki went to her 6 month wellbaby check-up this morning, and I'm happy to say, she's perfectly healthy and growing well. I'm doing everything right! After a lot of trial and error, with Annie (my guinea pig), I'm finally getting the hang of this motherhood thing. I must admit, it's a lot of fun (at times), but still quite demanding. Kallista is meeting all of the necessary milestones and then some. She's the size of a 12 month old, but her doctor says her weight will stabalize when she gets more active. I'm not overfeeding her, like some would like to think. We got all of her shots up to date and the first intallment of the flu shot. I'm high risk, so I got a flu shot too. When we go back for Kiki's second dose, we'll get Annie one, too. Then the whole family will be covered. After Kiki's appointment I met up with Gen at Starbucks. It was so nice connecting with her. She's probably the best friend I've had since I was a child. We've got a lot in common intellectually and we agree on a wide array of topics and that makes conversations easy. I hope we can develop a great friendship with time. I know she's got many other friends she gets a lot from, but I know she's got room for me :) I've got stuff to offer, too! Snookums is coming home from work early today, so he can comfort Kiki in her time of need (she's always a bit fussy after shots). This is the first appointment he's missed since I found out I was pregnant and I know it was hard for him to miss, even if he didn't say so. He's such a good father and husband. I'm so lucky. I'm going to go straighten up the house and throw in a load of laundry (all the clothes I've bled on in the last 24 hours, yuck!) and hopefully Snookums will think I'm a good wife (I know he already does :) Ciao for now.""",
    #     """When I got up this morning, I started planning my day out in my head: Feed baby, go for a walk, feed baby, clean the house, feed baby, make dinner, feed baby, go to bed. Pretty simple, pretty familiar. My day didn't end up working out that way, but it was a better day than I'd anticipated. And it wasn't as boring, which is good. Half an hour after Snookums left to take Annie to school, he returned. Thanks to the ice on the roads, a car hit a power line in the middle of nowhere, thus causing half of Kitsap county to lose power. Because of this, Annie's daycare wasn't taking children. Snookums brought her home so I could take her when they opened. This was what threw off my day. I was a little irritated, because I hadn't planned on going anywhere today and because Snookums returned my car to me, I was forced to run errands I'd previously managed to pawn off on him (I used the "since you're in the area anyway" argument and the "it'll save gas" rationale. Worked like a charm, too.) Let me just say, I LOVE the new automated package mailer at the post office. I needed to mail a teeny-tiny little package to an online friend, but it was hardly worth standing in line for an hour. With this new machine, I didn't have to stand in line at all! All I had to do was set my package on the scale, answer a few questions, swipe my credit card and I was on my way. It was great! The other errand I needed to attend to was a baby food run. I should buy stock in Gerber the way Kiki goes through baby food now. I spent about $40 on a week's worth of food. It may not sound like much, but when you consider one jar is only 32 cents, it's a lot. I did get her some juice, oatmeal and teething biscuits, too (which cost more than 32 cents) so I guess it was more like $35 on jars of food and another $5 for the other items, but why am I analzying this? Like anyone cares.....Ciao.""",
    #     """There is something so centering about getting your living space into perfect order. That's what we did today. As a whole family (minus the baby, of course) we purged, rearranged and organized our belongings into the unspoiled harmony that can only be achieved through hours of endless sorting. I'm a perfectionist. Not the kind of nut job that requires EVERYTHING to be the picture of zen utopia, but the kind of perfectionist that only wants to embark on a task that I know I can complete in the most sublime state of excellence I, as a mere human, can attain. That's how I felt about my task of picture hanging today. I started at around 10am. I measured, ruled out, charted imaginary graphs in my mind, laser leveled and eyeballed over 25 photo frames, most of the same basic dimensions. 7 hours later, I finished. Now I have the most beautiful, perfectly level, fully aligned, and evenly spaced gallery of family memorabilia this side of the Mississippi. I'm so proud :) I also got all of Snookums and my laundry done, every piece of it, but that isn't anything special. Tomorrow I'll do the girls' laundry. Snookums and Annie were quite busy themselves, today. They tackled the nuclear fallout site that WAS Annie's bedroom. I say WAS, because now it meets my definition of semi-PERFECT. She got rid of tons of toys and stuffed animals. I'm really proud of her. She came up to me several times throughout the day and said very sweetly "Mommy, it's sad seeing some of your stuff go, isn't it?" I said "Yes, baby, but you have to let go, so you can make room for new toys." This was enough of a reasoning for her, so she was willing to let go of a lot. I'm so pleased with how the house has cleaned up. The holidays really created a lot of clutter and I absolutely HATE clutter. Now I can leave my flawlessly groomed domain and pursue my professional life without thinking about the spring cleaning I probably won't have time to get to. I can feel the guilt melting away already. Well, I'm going to go brew myself a cup of tea and enjoy my recently redecorated living room. Ciao.""",
    #     """Starting tomorrow, my life as I know it is going to change. The party's over. My family is going back to work/school, respectively and my BABY is going to DAYCARE. I hate the very sound of it. But, I'm SO looking forward to going to the gym! I'll get to use my precious elliptical trainer and until I'm in my extern, I'll have time to myself during the day. Maybe just an hour or two, because my conscience won't let me leave Kiki in daycare any longer than I must. I'm sure she won't be traumatized, but It'll still take me time to adjust.We spent the day in much the same way we spent most of our holiday vacation...SHOPPING. We went to Wal-mart and got a big rug to put under Kiki's play yard, because it slides around on the wood floors and the play yard is designed to anchor down into carpet pile, so we got it some carpet to anchor to. Snookie went to Best Buy and got the new stereo system he's been wanting. The best reward for me was, I got the MP3 player I've been wanting!! It came with the system. I can download songs right off the radio, from CD's, whatever. No need for a computer, but I can do that, too if I want to. It isn't an iPod, but I'll take what I can get :) After that, we went to the mall and Snookie bought me new Nike crosstrainers and a pink gym bag. I'm all set! We went to AppleBee's for dinner and I was good. I ordered off the Weight Watcher's menu...I did order a Maple Butter Blondie for dessert, but I missed breakfast and lunch today and my dinner was only 370 calories, so I'm sure I'm not over my limit. I must go to bed now. I actually have to get up in the morning :( But, it's going to be great! Positive thinking is very important. Ciao.""",
    #     """Well, aren't I just the Queen of Amplification. I just got back from the stupid interview and it was absolutely no big deal. There was no pop quiz, no third-degree. Nothing. I talked to the Office Manager for about 15 minutes, got the grand tour of the practice and that was it. Nothing less, nothing more. She wants me to call and speak with her on Wednesday after she discusses it with all of the dentists (there's 3 in the practice). I don't think I did anything to NOT get the spot, but if I don't get the spot, oh well. I thought I was well-composed, put together, I looked nice, I spoke well. I did everything right. They've never had an extern before and they have to decide if it's something they think will be good for the practice. If I don't get the spot, it's okay. They work long hours and are always overly busy. I won't be upset if I find a more laid back practice. I KNOW I'll ultimately work at a laid back practice. I don't want to spend 12 or 14 hours a day at work. No way. It's only a little after 10am and I've already done more this morning than I've usually accomplished in a day. Definitely more than I did all day yesterday. Now I'll have to find something to keep myself busy with until Snookums gets home or it's time to pick up the girls. Whichever comes first. I know I'll eat something. Because I was so nervous about the interview, I kicked butt at the gym and burned over 700 calories. I had a slim-fast shake before I left the house, but it's long gone now and my tummy is rumbling. Snookie wants to go to the Olive Garden today, so I'm going to eat something light and then go on Calorieking.com and scope out the Olive Garden menu nutrition facts. Damn all this rain! I know I live in Washington State, but this rain is starting to affect my life, now and I don't like it. We've had 22 straight days of rain and this past weekend there was a mudslide that blocked (and destroyed) the road I take EVERY DAY. Now I've got to drive 5 miles out of my way and tack on an additional 15 minutes to my morning commute (traffic). The road will be closed indefinitely because of additional slide risks, so I'd better get use to my new route. I've really got to go eat now. My tummy is staging a mutiny and if I don't feed it soon, it'll take my small intestine hostage and eat it. Yeah, it's that bad. Ciao :)""",
    #     """I had yet another great workout today! Snookie came with me, again (he's had stand down the last couple of days, I'm not sure why, but I'm glad he got a little time at home) and this time he was tres motivated! He worked out really hard today, and I'm so proud of him, because for awhile there it seemed like I was the only thing standing between him and the "fat boy" program (the Navy's answer to overweight sailors, a mandatory work-out program that helps them lose the weight, but doesn't help them keep it off. He's done it before). I burned 1,000 calories and he burned 1,350. I've started a little race with myself. To burn more calories than I did the day before. At some point I may not, but it's good motivation trying.I started out not-so-motivated this morning. My period is on, I'm feeling a little weak and sluggish, my muscles are sore and on top of that I'm coming down with a cold for the first time in almost 2 years. I guess I'm long overdue, but I'd rather not be sick at all, ever. I took an 800mg Motrin (the ones they gave me after Kiki was born) before we left for the gym and it helped with the muscle pain. I plugged away on the treadmill at 3.5 mph until my muscles loosened up, then it was all gravy from there on out. I LOVE the gym! I talked to the lady that interviewed me on Monday, I got the spot. She and the extern coordinator have some things to iron out as far as scheduling goes, but everything is working out well. I got a spot on my first try! Maybe I'm not as much of a loser as I always make myself out to be. Maybe I am poised, mature and sophisticated like I'm always being told. Yeah, right. Whatever. After our workout, Snookie and I went to Central Market for lunch and to buy some new dishes. We did really well watching our intake. We had salad, a little bowl of chili and we split a 9-grain roll and a small plate of sushi. We also had coffee, but I had my usual non-fat, sugar-free hazelnut latte. Central Market's cafe uses only organic milk and coffee and I swear my latte tasted better there than it does at Starbucks. Maybe that's just my imagination. After lunch, we set about doing what the trip to Central Market was really all about. Right smack-dab in the middle of the store is a little Asian center. Where they make the sushi and sell a small selection of Japanese dishes. I got this brilliant idea from somewhere that I wanted to start using Japanese dishes and chopsticks to eat, with the idea that the smaller plate and the less efficient (for me, at least) utensil would slow and cut down on my eating. Snookie though it was a good idea and agreed to try it, too, so we bought really pretty sushi plates and chopsticks and we had dinner on them tonight. Both he and I can tell a difference already. We ate half the amount we usually do, yet no hunger pangs 3 hours later. Not to mention the whole meal looked more appealing on the pretty plates. Pork roast never looked so good... Well, I'm going to bed now. I'm still feeling a little ichy and I want to get some good sleep so I'll be ready for my workout tomorrow. Ciao.""",
    #     """I must say, I'm feeling pretty good today. I thought my day was going to go down the crapper when I woke up an hour late (7:14, instead of 6am). But since I'm hardly noticed at my extern, I didn't even bother to call in. Sure enough, I waltzed in at 9:04, posted myself behind the industrial shredder in the basement and not a single person noticed I wasn't there at 8am. So, I got an extra hour of sleep, one hour less as an indentured servant and the weather was MAGNIFICENT when I got off at 1am. Nothing could ruin my day! After work, I went to the mall to pick up Snookums' Valentine's Day present from the engraver. It looks so good! They even polished it up for me. The diamonds are twinkling and the gold and stainless steel are completely fingerprint free. I went to Hallmark, got some pretty bows and tissue, a gift bag (for a long jewelry box) and a great card. My shopping is officially done. I don't even care if all I get is a card from the ship store. I'm SO looking forward to giving this to Snookums. He does so much for me and puts up with my craziness, selfishness and over-indulgence. It's about time something nice happened to him. My workout was great. The gym was practically deserted, so there was no problem getting my favorite elliptical machine. When I came out, I felt so good, so relaxed and loose, limber and well-stretched. There really is nothing quite like a good workout to lift your spirits. The weather was even better at 3:30pm than it was at 1pm. I came out of the solid brick building into a wash of golden-orange that felt so GOOD. It was 60 degrees today. It reminded me of last summer and how I practically ran from the sun's touch at every opportunity. How I'd love to go back and fix myself so I wouldn't have lost a whole summer being depressed. But, since that isn't possible. I'm going to do my best to enjoy every moment I can. That's a promise to myself.""",
    #     """Well, tonight is my last night of sleeping alone for awhile. This time tomorrow my Snookums will be home! You don't even know how glad I'll be to finally have him home. I'll have someone to help me with the girls, understand my limitations (the pain) and just love me. I know the girls love me, but no one loves me like Snookums does. I've got an ambitious project on my agenda tonight. I'm making CHEESECAKE! I'm using lowfat cream cheese, and lowfat whipped cream, so it isn't going to be too bad, but I wanted to make something special for Snookums. I bought little ramikans and each little cheesecake will be portion controlled and finished off with a cocoa powder heart on top. I'm sure he'll love it.I just had a wonderful day today. Everything about it was good, even though I thought it wouldn't be because of the night I had last night (I took a Vicodin on an empty stomach and spent half the night praying to the porcelain god). But, I woke up feeling okay and things went great from there. I did 3 chairsides at work today, including a 2 tooth simple extraction. The day flew by! I got my nails done, they turned out gorgeous, I had a great workout, I went grocery shopping to get some Snookie-foods back in the house and now I'm ready to finish off my day. The kitchen is the only room of the house that needs my attention. I'll clean it up and make my cheesecake, then I'll retire early so I'll be well-rested tomorrow. Hopefully, I'll get to have another action-packed day! Sorry if this entry is rushed. It's almost 9am and I have things pressing. Ciao until tomorrow.""",
    #     """Thank goodness it's Friday! It's been awhile since I've felt this greatful for the weekend. Although this week seemed to have flown by, I'm still really tired and worn out. I'm going to sleep as long as possible tomorrow. Snookums has duty tonight, so I'll be sleeping alone, but he said he'll be home by 7:30am, so I'll be able to sleep in and he can take care of the girls' in the morning. I'm not trying to take advantage of my condition. I'm trying to keep up my end of the household, but I get so worn out from the constant tension and physical discomfort I'm in. Won't we all be glad when this is over? I feel bad for the people that read my diary. My pain appears to be the main topic of discussion for me, lately. Sorry :} Work sucked today. Partly because it was quiet and I didn't have a lot to do (other than pulling about 5 million charts and restocking) and because I was in a crabby mood. I wish Vicodin didn't make me loopy, because I'd take it during the day, but that isn't the case, so I have to tough it out and if things aren't busy to keep my mind off of the constant ache, it feels all the worse. Instead of bustling around getting x-rays developed, writing out patient treatment forms, updating the schedule and other such meaningful tasks, I spent 5 hours pulling Tuesday's and Wednesday's charts, restocking patient bibs, cotton rolls, chair covers, putting 150 bitewing tabs onto films, making full mouth x-ray bags and then another hour of walking around looking busy, when in reality I was bored stiff. Next week I'll only be working 3 days. You don't know how much I'm looking forward to that! Tonight, I'm going to be nice to myself and take a scalding, hot bath filled to the brim with Calgon bubble bath and lavender oil. I'm going to light a million candles, sink down into the water and just steep like a tea bag. Then, when I get out, maybe I'll have a cup of tea with my nightly narcotics and pass out peacefully while.""",
    #     """Compared to yesterday, today flew by. Snookums came home around 8am with work for me to do. He brought home all his uniforms to wash and I had the honor of pressing and creasing them according to military regulation. Fun. Just what I like to do on a lazy Sunday morning. He usually gives me at least a week's worth of notice before an inspection so I can drop his uniforms off at the cleaners so they can wash and press them, but no such luck this time. There is a surprise inspection Monday morning at 0700 hours. Even the fastest cleaners couldn't have them ready that quick. So, it was up to me to save his ass. But, once the work was done we got the chance to relax. Snookums bought me Forrest Gump on DVD as a peace offering and actually watched it with me! I was so surprised. We generally don't share the same taste in movies, so it was nice to watch a movie together without him falling dead asleep before the opening credits are done. He bought another movie, Bringing Down the House with Steve Martin and Queen Latifah, but we didn't finish it. It was 5pm and we need to get the girls' fed and ready for the week ahead. Tomorrow it's back to work, school and daycare.""",
    #     """Yes, I spelled pointe right. If you aren't familiar with the term, that's what it's called when you see ballerinas standing on their tippy toes. And this is my NEW ultimate goal. You heard me right...I SIGNED UP FOR BALLET CLASSES TODAY!!! I start this Thursday at Irene's School of Dance in Silverdale. I'm SO looking forward to it! When I was a girl (from the age of 6-13) I took ballet, tap and jazz. But, once I hit the teen years I gave it up, because my Mom was a bit of a stagemom and I couldn't handle the pressure to be the best. It took all the fun out of it. But, I think now I'm ready to try again. I don't expect professional caliber performance from myself. That would be like expecting a pack mule to become a Kentucky Derby winner, but I do think it's one more way I can learn to love my body. Dance is such a beautiful thing. I considered enrolling Annie, too. But, I tried putting her in dance when she was 4 and she stood in one spot for the entire hour and refused to move. After a month of that, I pulled her out. It obviously wasn't her thing. If she shows an interest now, I'll consider it since she's older. She definitely has a dancer's body even at 6. Tall, muscular and very linear. Me? Not so much. I'm more soft, round and curvy. Work was slow for the most part. I spent a good chunk of my day filing and stuffing. Filing patient charts the front office hadn't gotten to yet and stuffing statements into envelopes so they could be mailed. Not glamorous and certainly not fun. What does this have to do with dental again? I had to cut my workout short today, so I could make time to get my nails done. They look nice. I love being well groomed. I'm thinking about doing something different with my hair. I can't decide between braids or maybe a new cut and color. My hair has gotten really long and I've been embracing my curls lately, so it's super healthy, too. If I do get it cut, I want it to be something that doesn't lose much length and compliments my natural curls. I'm thinking maybe amber or blonde highlights over my natural light brown color. That would be nice, but I'll have a consultation done before I settle on anything. I'm going to go now. I'm sleepy and I know Snookums is too.""",
    #     """Snookums wants me to hurry, so I have to make this short! It's 10:38pm (very late for us) and we're going to take a shower and go to bed. Only, this morning I promised I'd write later, so here I am.Work:GOOD. Gym:GOOD. Rest of day:GREAT! That's the short version, since I'm in a hurry. Snookie got off from work around 1:30pm and I had another chit-chat with Sherilan about my impending work status, so I was running a little late. We met up at the gym and then hit Subway for lunch. After lunch, Snookie and I went to the ATM to withdraw money for gas (we always get good news when we're getting gas money!) Lo and behold, our balance was just over $10,000!!! Between our income tax return ($7,000), Snookie's paycheck ($2,000) and the money we already had in there (about $1,000), we're ROLLING in it. So, we decided to go shopping. We bought Annie the CUTEST pair of pink and white Air Jordan tennis shoes. Snookums got some new clothes and a pair of Timberland boots. I got some new warm-up suits for the gym, two new jackets (a Calvin Klein jean jacket and a shearling and tan suede coat), new jewelry (yellow and white gold matching heart earrings, necklace with pendant, and braclet), and a armband holder for my MP3 player. On Monday I'm going to get my hair cut and colored at the Regis salon in the mall. They have 3 stylists that specialize in curly hair. Now that we've indulged ourselves, the rest of the money is being squirreled away for a rainy day. Well, I'd better go now, or Snookie will fall asleep on me. Ciao!""",
    #     """Today was such a BEAUTIFUL day! The first day of Spring couldn't have been more perfect. The sun was shining, not a cloud in the sky. It was WARM. Just perfect. I'm pleased that my depression as cleared up enough for me to appreciate the sunshine again. Let's hope with the sunny weather coming I can enjoy it this year, instead of hiding inside like I did last year.I think the cheerful weather does something to my driving ability, because I caught myself going 80 a couple of times on the highway today. Which I SHOULD NOT be doing. For one it isn't safe and two, I don't need a ticket. Last year I got pulled over, doing 83 in a 60. I got a $183 dollar ticket! The judge deferred it (which means I don't have to pay it) on the condition that I go a full year without getting another ticket (April 22 of this year will be a full year from the court date). If I do get another ticket I have to pay the $183 ticket, the new ticket AND I paid a $200 dollar fee to keep it off my record, so I don't want to get another ticket. Alot of money is on the line. My day wasn't as hair-pullingly busy as I thought it would be. I went to the gym this morning and kicked butt. Seriously, I was sweating so hard and feeling so good. I'm proud of myself. I can feel myself getting firmer, more toned and I'm starting to see the results. Go ME! I went into work for the Dentrix (computer software) training and it was a scary experience for me. Not the training, but the food. Sherilan had set up a pretty impressive luncheon. Baby green's with bell pepper vinaigrette, cole slaw, fried chicken, potato chips, soda, grapes, apple pie. I was absolutely PETRIFIED to eat in front of all the other girls. It wasn't that I was afraid of the act of eating, it was that I hadn't planned on eating anything at all. I didn't want to eat foods I didn't know the calorie content of. So, I took a small bowl of the baby greens, a handful of grapes and a cup of diet soda. Total= 91 calories. I looked it up on calorieking.com when I got home. I'm under 700 calories for the day. The thing that frightened me a little was how out of control I felt. I didn't want to stand out by being the only person NOT eating, but I didn't want the calories. I didn't want the extra food. I just wanted to stay in control and I felt like I'd lost it. I was glad to get out of there and away from that food. I'm a foodaphobe, I guess. I was suppose to get my hair braided today, but Felicia (the lady that does my hair) was running behind and wouldn't have had enough time to get it done before I needed to pick up the girls from daycare. So, I'll make an appointment sometime next week and hope she can do it then. She's really in high demand, because she's the only one in there that does "ethnic" hair. It was just a whim anyway. I might decide I don't want it done, anyway. I bought new ballet gear today. I got a much more comfortable leotard than the one I got at Irene's (my school supplies a leotard and tights for people that don't want to buy the professional quality stuff) and I got a tutu to cover my bubble butt. I decided I wanted leather ballet slippers instead of the elastosplits, because I need the support for my high arches. This Thursday Snookums will be home, so there's no reason for me to miss class. Well, I'm going to see if I can retire early tonight. Maybe I'll paint my toenails or something. Ciao until tomorrow.""",
    #     """I just got back from getting my new tattoo. It was an experience that I won't soon forget. It wasn't my first tattoo, I've got two little butterflies on the left side of my chest (symbolizing my babies), but it was my first BIG tattoo. It's about 9 inches across the small of my back and about 3 inches high. I really had to focus my mind and think beyond the pain, because while he was tatting away on my spine, the pain was like white hot fire searing up my back. Amazingly, though. There was something really therapeutic about it. Like I was being cleansed or something. Butterflies symbolize rebirth and renewal and that's pretty much what I'm trying to do with my life. Now I've got this amazing art piece no one will know about unless I decide to share it with them :) This morning Snookums and I went to the gym together and did a pretty good workout. I think my kidney stone might be moving, because while we were there I went to the bathroom and pissed blood. Which is a sign that something is going on up there. Good. I want the damn thing out already. Anyway, my workout was good, even if I was preoccupied with my bloody urine and my impending tat appointment. After the gym we went to lunch at Subway and then did some grocery shopping. My tat appointment was at 2pm and took 3 hours to do, so now it's after 5pm and I'm drained. So spent. Snookums is picking up the girls and I think I'll throw us together some salads. I know I'm going to bed early tonight, so I'll just say goodbye for now! Ciao!""",
    #     """I don't have a lot of time to write. It's 10:37pm, I just got off the phone with Shannell, I had dance class tonight and Snookums has gone to bed without me, so I'm making this short. Snookums and I went to the gym together this morning, then we went to Annie's parent/teacher conference. She's doing amazingly well academically. She's reading at a 3rd grade level! She's only in the 1st grade, so that's really impressive, but she's got to work on talking in class. The same problem she had last semester. Dance class was great. I LOVE it so much! I'm getting more comfortable with the flow of the class and my teacher's style. Not to mention all the french positions and movements are starting to come back to me after all these years. Jazz is really getting fun, as well. We're working on getting the kinks out of our routine of All That Jazz for the summer recital. It should be really polished by then. Well, I must be going. I'm feeling a bit dizzy. I've only had 650 calories today and I know I burned that much in dance class alone, so I'm going to have a snack, maybe some fruit and yogurt with a little granola or some organic watermelon before bed. Nothing over 100 calories. Maybe I'll have the watermelon. I get more of it. Ciao.""",
    #     """I'm throughly exhausted. I'm happy, in good spirits, but tired to the bone. I know it's because I'm still not use to working a regular workday, but I'll get use to it. Each day is a little better than the one before. Not to mention I get a 3 day weekend every week. How many people can say that? This morning went really smoothly. I got ready, got the girls ready and got out the door right on time with no problem. Being on time first thing in the morning really sets the pace for the rest of the day. If I'm rushing right out the starting gate, I find myself rushing all day. I got off work a few hours early today (and I think I will tomorrow, too). I'm not sure why the office closed early, but I'm not complaining. I decided to hit the gym, even though I didn't have my gym stuff with me. So, imagine me power walking on the treadmill in scrubs...funny picture, huh? Well, I couldn't pass up an opportunity for calorie expenditure. Speaking of calories. I'm still having trouble keeping my calorie count low. Now that I have to be a functioning, productive member of society I can't run on such low reserves. I'm burning more calories from working all day (I'm on my feet and power walking all over the office, running for this, rushing to get that. It's a lot of work. Not to mention our stockroom is in the basement and I go up and down those stairs about 20 times a day), and I can't simply ingnore the hunger. I have other things to concentrate on and being weak, hungry and light-headed from low blood sugar isn't working out. I haven't even been keeping track of what I'm eating very well. Just rough estimates. I haven't been binging, just eating-dare I say it-NORMALLY. I know, strange. Me, doing something normally. Who'd have thought... I'm going to do more research on line to see what the best method would be to boost my metabolism, start losing weight again, but not starve myself. It's the only way I know how to lose, unfortunately , it isn't going to work so well for me in the long term. Why do things have to be so complicated? Dance was great. I'm loving it so much I can't even describe how glad I am that I had the courage to sign up. My instructor is great, I love the pace of both classes. Dance is definitely my thing. I'd love to chat more, but it's already after 11pm and I NEED to get sleep or I'm going to crash and burn early tomorrow. We don't want that to happen... Ciao.""",
    #     """What a day... It was good, though. Long, but good. For some reason, I ended up staying awake until 3am last night (well, this morning, I guess). I was having my own little party. I didn't do anything, really. I flipped through some magazines (Oprah's magazine, Parenting and the March issue of Hustler I found under the bed last week) and I watched late night tv. I suppose I was making up for going to bed at 8:30pm so many days last week. This morning, the girls' and I got ready and left the house around 11:30am. I took them to McDonald's for lunch, then dropped them off with Shannell, so I could go to CPR recertification at work. Annie had a great time at Shannell's grandmother's house. There was a big Easter egg hunt. Shannell's grandmother lives in an old house with beautiful gardens and landscaping, which was perfect for the occasion. I wish I could have stayed, but CPR is kind of required if you work in the healthcare field. Since we were having lunch after CPR, I stopped by Coldstone and got one of their ice cream cakes, which was a big hit. It was pretty good. I got the Peanut Butter Playground (devil's food cake, chocolate ice cream, and Reese's). There was pizza, salad, crudites (raw veggies), chips, soda, cookies and my ice cream cake. Not too bad a spread for short notice. We decided yesterday afternoon to have lunch, so it was kind of thrown together. After CPR, I went to Shannell's house to get the girls' and we hung out for awhile. Now I'm home. I'm sleepy, but I still have a few things I have to do, like straighten up my house. It's not dirty, just cluttered. So, I'm going to crank KUBE 93 online and get it done. Hopefully before Snookie calls. So, I'd better get going if I want to get done. Ciao!""",
    #     """It really hit me this morning, that my little baby is 1 year old. It's Kiki's birthday, today. Actually, this very moment last year was when I went into labor. Snookie had just left for work, I was sitting in bed watching the news when I started having MAJOR contractions. Since I drop babies very easily (and quickly), I had to call Snookums back (he didn't even make it to the ship) and by 10:30am(4 hours later), Kiki was here. I'm not going to get emotional about it, but it's just so hard to believe she's that big, now. It's been a good morning. I got on the scale and I'm only 1 pound away from normal, so all that water weight is gone. I wouldn't be surprised if that last pounds was punishment for all the binging I did last week, but I'm pretty sure I worked off enough to NOT gain any weight. I guess I'll see in the morning. If it's still there, then I know I messed up more than I realized. You live and you learn. I'm just glad to have my motivation back! I really do want to be smaller when Snookums gets home. That is important to me. Well, I need to be going. I plan on going to the gym, getting the Impala pimped out (okay, getting it washed, but it looks so good clean) and then I'll be picking the girls' up early and taking them to meet the ladies at work. So far, only Mindy has met them. I've run into Kristin in public before, but she hasn't actually gotten to meet them. So, hopefully everyone doesn't go out to lunch today :) Well, I'll write about the rest of the day, tonight. Ciao for now!""",
    #     """Why is it when you get a group of woman together, all we want to do is chat like clucking chickens? Gymtime is usually a solitary time for me. Aside from the occasional lewd comment from a sailor, I'm pretty much in my own world the majority of the time, but today (when I actually had somewhere to go, something to do) I ended up getting into a lengthy group therapy session in the sauna. I don't know. I guess it was nice, connecting and communicating with other human beings. I don't do that very often, so it's something I'm not familiar with, or totally comfortable with. But, it isn't really a big deal. I wanted to get the car washed before work (that's what I needed to do) but, it can just stay filthy. I've done well today. According to my 7-day cycle diet (which I've decided to stick with, because it's working and it's NORMAL. Well, I'm trying to stick with. Sometimes the calorie requirements are too high, in my crazy, warped mind) I'm suppose to have 1514 calories today. My logical mind knows that this is only 100 calories over my BMR, therefore, well below what my body would burn in a day. But the Disordered Eating side of my brain is saying "NO FUCKING WAY, you're eating 1514 calories! You'll never lose weight stuffing your face like that!". So far I've had 1185 and I'm feeling like that's enough. I haven't been hungry at all. I know how to space out as little as 500 calories in such a way as to avoid hunger pangs, so of course over 1000 calories wouldn't leave me hungry. But, I'm rambling, so I'll stop. I need to get my butt off this computer and get my house cleaned up. I have been seriously slacking off when it comes to housekeeping, so I'm going to make myself a TO DO list and start tackling it. After all, my Mother-In-Law is coming on Sunday. Can't have a dirty house :) Ciao.""",
    #     """It's late, I'm sleepy, so this is going to be short. Work was okay. If I had to put an adjective on the day, it would be "irritated" because it seemed like everything at work irritated me. I have neither the time nor energy to elaborate, just know that I was irritated... I also found out that I'll be working on Monday and Tuesday of next week (the days my mother-in-law is suppose to come). My schedule is still easy. Only 3 or 4 days a week, but I had to call and tell her that she might have to reschedule her visit. Snookums is suppose to be coming home on Wednesday, so I have that day off and that worked out good. Then he told me he might be coming in on Tuesday, which doesn't work so well because I work that day. We'll see, I guess. Dance was great! Yes, I went today. It really helped get rid of my "irritated" mood. I'm signing off for the night and heading to bed, because I stayed up late last night and I really need more than 4 hours of sleep. Ciao :) """,
    #     """Work was beyond busy, but oh so productive. I walked in at 7:15am (on time for once! So, I had plenty of time to review my patient charts) It felt like I blinked and it was 5:30pm and I was shutting down my operatory and heading home. I'm getting so comfortable with the flow and order of the office, now. There isn't the slightest bit of apprehension (like there was a few months ago) or nervousness when it comes to assisting. I'm finding my niche :)Snookums has duty tonight, which is fine. Even though it feels like he just had duty. It's actually been a week. I hate to admit it, but I get better sleep when he's gone than when he's home (he's a violent sleeper). So, tonight I'm hitting the sack early. I'm taking a hot shower, then I'm passing out. The girls' are doing better today than yesterday. Annie is over her injured face (everyone has gotten used to seeing it and isn't making fun or questioning it anymore) and Kiki's rash is about 90% gone. She's got such a fair complexion that she can't help but always be a little pinky-red. Both girls are in a good mood (or they were before they went to bed), so I'm happy. I'm so sleepy, I can barely keep my eyes open. Now that I'm coming down off my work-induced high, I'm ready to crash. Literally. Ciao until tomorrow :)""",
    #     """I'm exhausted. Thoroughly tired. Pooped. And any other way you can think of saying tired. Last night I only got 5 hours of sleep. Not because I couldn't sleep, but because I stayed up too late. Since Snookie stayed home with Kiki today (whom did not appear to be ill in the least bit. I hate the CDC's sick policy!) he didn't have to get up early. So, he felt no need to go to bed early. I however, neglected to remember that I DID have to get up early. So, I'm paying for it now with a tired headache (the kind you get from lack of sleep, that only sleep itself can cure). Work was excellent today. The symbiotic relationship we have at work is almost scary. All three of the doctor's believe in the philosophy that any number of people working symbiotically will equal 1. Meaning, that everyone (all 15 assistants and 3 doctors) working towards the same common goal (treating patients as best we can medically and emotionally. Getting them in and out on time and following their treamtment plan) should create 1 efficient practice. All the necessary systems are implemented to make that happen and I'm just starting to understand why things are done the way they're done and I'm fitting into the puzzle! It's a great feeling. I'm loving my job! Which is nice, considering so many people don't love their jobs. After work, I picked Annie up from daycare, stopped at home to get Snookie and Kiki, then we went to the Chinese Buffet. It was good, as usual. I'm not really thinking too much about my weight right now. Other than the occasional guilt of being so far off track I can't even see the tracks anymore. I need to get my ass in gear and stop indulging myself whenever I want. I haven't weighed myself in days for fear of the number I'll see. Maybe that's what I need to do to get my ass back in gear. Maybe tomorrow. I know for a fact that I'll lose a ton of weight once Snookums is gone again. It's just so hard when he's home! Ciao :)""",
    #     """Shame on me for not writing last night, I know. How could I? Well, the fact of the matter is, I fell asleep. On the couch. All I remember is slightly waking up while Snookie put my pajamas on and then falling back to sleep. Only to wake up to the sound of the alarm clock this morning at 5am. I turned it off and slept for another two hours. It was so lovely sleeping in. I've got five days off in a row and I'm going to enjoy them! Snookie and I went to the gym together this morning. Then we came home and cleaned the house up for a couple of hours (I at least wanted the house to be presentable for his mother). After mutual clean-up time, we showered (together), went to lunch (at the Pancake House), then strolled around Wal-mart for a bit (Snookie's step-mother sent us a $100 gift card. Not my first choice, but it was a gift). Half-way through the store, Snookie's cellphone rang and it was Sue, saying she was at the Tacoma Narrows (a bridge about an hour away from us). So, we payed for our crap, and went to pick up the girls. Our first evening with Sue hasn't been bad. She ate a Lean Cuisine for dinner, we had chocolate torte for dessert and she's drank 3 beers (It keeps her mellow. When she comes is the only time beer enters our house, but she brings it herself, so I don't care). Now Snookie and Sue are watching Comedy Central. It's Carlos Mencia on right now and I'm distracted, so I'm going to go watch, too.""",
    #     """I wish the last weekend I'm going to have with my husband for the next two months WASN'T being shared with his mother, but what can I do? I don't get much in life my way, why should this be any different? At least there hasn't been any fighting. I should be grateful for small favors. Today was such a boring day. We spent most of the morning doing nothing other than watching tv and keeping Annie and Kiki occupied. The dragging hours were intermitted with loads of laundry either being put into the washing machine or being taken out of the dryer. It's sad when you're only hope of entertainment is putting laundry away. Sad, just sad. Around 4pm, we dragged everyone out of the house, crammed into the microscopicly compact car and went to Applebee's for dinner, which was a disaster. The meal itself was great, but when it came time to pay, Sue (who offered to take US out to dinner) couldn't pay because her card was declined twice. So, we footed the bill for the meal we were suppose to be treated to, but oh well. At least we get 3% cash back for eating out. By the time we got home, all the fat and chocolate I ate was starting to get to me. Now, my IBS is raging like a lion in my intestines and I'm miserable. But, I guess it'll pass and I'll live. I'm going to go for now. I've got nothing else interesting to write about and I'm sure you don't want to hear about my irritating mother-in-law, my irritable bowels or any of the other irritants in my life. Ciao"""
    #     """I got a most wonderful surprise today! I was on my way to Jackson Park (I wanted to jog on the beach before I picked the girls up) when my cellphone rang. It was the auto body shop. My Impala was ready! So, I made a U-turn and went to get my car. It looks SO GREAT. They blended the new paint into the old, so it doesn't look like it had anything done to it. They even buffed the entire car, so it's really shiny and clean. I'm so pleased with it! Once again, my car is perfect. It was so lovely getting back into my big, spacious, luxury sedan. It felt so good resting my back against the cushy lumbar support, having tons of leg room and not feeling like I was crammed into a clown car. Not to mention I missed my digital stereo and actual working speakers. Did I mention I love my big car? I'm so glad to have Pala home (that's what I call it. I haven't decided if Pala is a girl or a boy car yet, but at least I've got a name). I put a couple pictures of the car on my "website" (my t-mobile album) so Snookie will be able to see the improvement. I almost wish I'd put a couple of damaged ones on there, too, but that would have just been depressing. Oh, and I did still get to have my jog on the beach, by the way. It was just cut in half. I took the girls to the playground, too. Today turned out to be a not-so-bad day, after all :) The girls will be going to bed here shortly and as soon as they do, I'm going to sneak in a few more ab sets on the stability ball, then I'm going to soak in a nice, hot bath (with copious amounts of white tea and jasmine body soak crystals) while I read my book. I'm still trying to finish Paradise (by Toni Morrison, my favorite author), which is overdue. It should have gone back to the library on the 19th, but I haven't had the time to finish it. I will this  weekend, for sure. Anyway, that's my plan for the evening. I work tomorrow, so I'm going to try to sleep more  restfully than I did last night. I hate sleeping alone. Ciao.""",
    #     """I'm in the middle of working on my therapy assignment, so I don't have much time to write. Yes, I procrastinated. My appointment is tomorrow morning and I'm just now doing it. But, at least I'm doing it. That should get me some kudos. There's nothing interesting to write about my day. I went to the gym, went to the park (no problems with the creepy guy) and then home. Nothing spectacular. So, if you don't mind, I'll be getting back in touch with my feelings. Ciao!""",
    #     """I'm not going to go into great detail, but for the past two weeks I've been doing something I know I shouldn't. Something that, if discovered, would ruin my marriage (thus my life). So, it can never be revealed. But, in the process of ALMOST transgressing (or actually transgressing, depending on how you want to look at it) I learned a valuable lesson: If I'm too ashamed to write about it in my diary, than I have no business doing it. I won't go any further into the subject and I know my enigmatic attempt at beating around the bush only piques your curiousity, but I truly can't talk about what I did. It isn't something than I want floating around in cyberspace. That's all I have to say about that. As for my day. It was okay. I was caught in a shame spiral and needed both a binge and retail therapy, so I headed to the only place you can get both under one roof: The mall. I had a quarter pounder with cheese and fries for lunch followed by 3 Mrs. Fields cookies. After that, I stopped in Victoria's Secret, because I can't miss the semi-annual sale (I managed to find 3 perfect-fitting bras and 5 pairs of V-string panties, $113 damage, not too bad). After that I stopped in Forever 21 and found the perfect "Welcome Home" dress for Snookie. It fits me to perfection, it's a little long, but that's a simple altering job. Nothing I can't handle. It's a halter style dress, which is very flattering. I love it! I also bought some dangly gold earrings to wear with it. I bought the new microwave today. They still had the red one and the matching toaster, so I got them. I'm thinking about going back for the matching coffee pot, just to keep the red enamel theme going. I don't drink much coffee, so it seemed like a waste to get rid of our perfectly good white coffee pot for a prettier red one, but I probably will end up doing it, eventually. I'm sleepy (then again, what's new?) so I'm ready to sign off for the night. Ciao.""",
    #     """can't beat this sadness. It's sticking with me no matter how delightful a day I may have. I can't get out of my own head long enough to see the good there is in the world. And that's sad, because I'm sure there is good in the world. Otherwise the suicide rate would be higher than it already is. I guess I'm just a sullen girl, what can I say? I started out my day feeling really high. I had a great workout at the gym, I was feeling really great, but it didn't last long and just as illusively as it came, the happiness went. Like a balloon losing it's helium. I struggled to enjoy the rest of the day, even though it was a good day. Gen and I spent it together. I find her company very pleasant. She takes me for what I am, doesn't expect much from me and doesn't ask much of me in return. This is good, seeing as how I don't have much to give. We had breakfast at The Diner in Poulsbo, walked around the waterfront, then went to see The Lake House (it was good, but a little enigmatic). Afterwards, we went shopping. I bought a bunch of new tank tops and a couple of skirts. I'm seriously lacking in the summer wardrobe department, since I'm only 80 pounds lighter than I was this time last summer. Gen and I came back to the house and had a long conversation. It was nice having someone to talk to (other  than my therapist). I realize no one can fully understand what I'm going through unless they themselves have or are going through it, but Gen is about as close as I think I'm going to get. Empathy is a specialty of her's. Even if she isn't aware of it. I'm tired. I'm going to attempt sleep and hope I'm successful. Sleep is so illusive. Ciao.""",
    #     """Sorry I didn't write the past couple of nights. Our internet connection is cable and our modem hasn't wanted to remain online. So, it drops whenever it wants to. This is both irritating and convenient, because I needed an excuse to explain to Snookie why I won't be answering his e-mails next week. A downed internet connection is perfect! Wave came by today and looked at the line. When the building was built (in the 70's) it was fitted with antenna line, but cable wasn't around yet. So, it was retrofitted later and the old line is having a hard time supporting the new technology. I absolutely refuse to pay for a new line to run through this place. I don't own it and I can't take the line with me, therefore, I won't do it. Whatever they did to fix it today, they'll just have to keep doing until I move out.I'm procrastinating. I'm really looking forward to this trip, but I have no interest in doing the things I need to so I can go on the trip. I don't want to wash clothes, pack bags, clean up the house. Nothing. I just want to get in the car and go. If I didn't have two little children, I could do that. Make my was as I go, fly by the seat of my pants! But, children don't work that way and I'll need a little more preparation than "get up and go". I figure if I get all the laundry done today, pack tonight and clean tomorrow, I'll be both fully occupied the entire weekend and well rested because I didn't cram all the work into one day. Probably Sunday if I hadn't been able to get myself out of bed this morning. I went to Wal-mart this morning to get some provisions. Drinks and snacks for the girls, water for me and sunblock. Lots of sunblock. I also got Kiki a baby raft, since I'm sure she doesn't remember swimming in utero. I'm so ready to run! Ciao.""",
    #     """Snookums looked like he was going to pass out when he saw me sitting in that chaise by the pool. He was completely fooled and had no clue whatsoever what was going on right beneath his nose. IT WAS BRILLIANT! We spent another hour or so, just playing in the pool having a great time as a FAMILY. That is still such a foreign concept to me. But I'm learning to love it more and more each day. After the pool, we all came home to shower and get ready for dinner. We ended up having dinner at a Chinese Restaurant up the street (which was good), then we came home and just chilled. Annie went to sleep around 9pm (she hasn't been following her usual schedule and isn't getting enough sleep, so she literally passed out tonight). Kiki is completely off schedule too, but it isn't so bad. She's taking 3 naps a day and sleeping through the night, so who can complain? I've been getting amazingly good sleep and I know tonight will be the best yet, because I'll have Snookums by my side, I'm happy and sex-sleepy. All of which add up to a great night's sleep. Never have I felt so important or cared about in ALL of my life. Just 10 minutes ago, we were all sitting in front of the tv (Me, Chris, Dad and Snookums) watching Who's Line Is It Anyway? When, I decided that I wanted ice cream. Which I just so happened to say out loud. Not even a minute later, all three of them hoppped up, put on shoes and shirts and headed out to buy ME ice cream. Just because I said I wanted it. Chocolate, nothing crunchy in it. I'm the Queen and I'm treated as such, which is great fun! But, they're going to give me a big head :) Tomorrow is going to be a lovely day. We're going to Seaworld! I think. Ciao until tomorrow.""",
    #     """Well, there's been a slight change in plans. Instead of leaving today, like I'd intended to, I'll be leaving tomorrow. Dad didn't seem to think I was emotionally ready to leave and I'd have to agree with him. The thought of returning to my eerily quiet home, with all the ghosts and demons I ran away from, waiting for me to return, is a frighteningly unsettling proposition. I'm just not ready to face that reality just yet. Dad is Buddhist. Which isn't so much a religion (even though it is catagorized as one) as it is a way of being, thinking and living. It's something (in my quest for comfort and enlightenment) that I've considered pursuing (Remember this conversation, Gen?) but haven't felt convicted enough to go through with. So, Dad has given me several books, shared his experiences over the past 30 years and I think I'll look into it. What have I got to lose? My sanity? I think not. I'll have to reschedule my therapy appointment on Thursday seeing as how I won't be home by then. I feel ready to go home, but there's something so comforting about being here. Chris, Annie and Kiki keep me busy all day. Dad and I philisophically connect (talk) in the evening, I SLEEP all night, wake up and start the whole process over again. It's been heaven. The only thing I miss is the familiarily of my personal belongings, not living out of an overnight bag and being able to get around without directions from a 12 year old. But other than that, I'd be perfectly happy never going back to Washington. No offense, it's a great state, but too much shit has happened to me there. I need a fresh start, unfortunately, now is not the time for that fresh start (Snookums is kinda stationed in Washington for the next few years). The children are getting restless, so I'd better get going. I'm off to the pool. Ciao."""
    # ]
    #
    # for i in range(0, len(JENNIFER_DIARY)):
    #     diary = JENNIFER_DIARY[i]
    #     print("start tagging diary #%s" % i)
    #     diary_tags = tagger.tag_pos_doc(diary, True)
    #     print("create piclke for tags of diary #%s" % i)
    #     tagger.tags_to_pickle(diary_tags, "pickles/jennifer" + str(i) + ".pkl")
    #     print()

    jeniffer_diaries = list()
    for i in range(0, 35):
        diary_tags = tagger.pickle_to_tags("pickles/jennifer" + str(i) + ".pkl")
        jeniffer_diaries.append(diary_tags[1])
    print("load jeniffer diaries done.")
    # print(jeniffer_diaries)
    tend_analyzer.analyze_diary(jeniffer_diaries)

