from collections import defaultdict
from nltk.corpus import wordnet as wn
from pprint import pprint
import csv
import os
import operator

from diary_analyzer import tagger
from diary_analyzer.tagger import TAG_WORD, TAG_WORD_POS, \
    TAG_DEPENDENCY, TAG_WORD_ROLE, TAG_NAMED_ENTITY


class HyponymRetriever(object):
    IDX_SYNSET = 0
    IDX_LEVEL = 1
    IDX_LEMMA_WORDS = 2
    IDX_LEMMA_COUNT = 3

    def __init__(self, *root_synsets, max_level=1):
        if type(root_synsets[0]) is list or type(root_synsets[0]) is tuple:
            root_synsets = root_synsets[0]
        self.hyponym_list = list()
        for synset in root_synsets:
            self.hyponym_list += self._collect_hyponyms(synset, max_level)

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

    def get_list(self):
        return self.hyponym_list

    def find_synset(self, synset):
        for item in self.hyponym_list:
            if synset.name() is item[self.IDX_SYNSET].name():
                return item[self.IDX_SYNSET]
        return None

    def check_synset_in(self, synset):
        if self.find_synset(synset) is not None:
            return True
        else:
            return False

    def find_word(self, word):
        for item in self.hyponym_list:
            if word in item[self.IDX_LEMMA_WORDS]:  # lemma list
                return item[self.IDX_SYNSET]  # synset
        return None

    def check_word_in(self, word):
        if self.find_word_synset(word) is not None:
            return True
        else:
            return False


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
    TARGET_TYPE_THING = 'thing'
    TARGET_TYPE_ACTIVITY = 'activity'

    def __init__(self, senti_wordnet,
                 food_set=None, hobby_set=None, sport_set=None):
        self.senti_wordnet = senti_wordnet
        self.food_set = food_set
        self.hobby_set = hobby_set
        self.sport_set = sport_set
        self.other_sets = dict()

    def addWordSet(self, target, targetType, words_set):
        pass

    def scorePreferenceToDiary(self, diary_tags_list, target, targetType):
        words_set = None
        if words_set == 'food' and self.food_set:
            words_set = self.food_set
        else:
            if (target, targetType) in self.other_sets.keys():
                words_set = self.other_sets[(target, targetType)]
            else:
                if targetType == 'thing':
                    synsets = _get_synsets(target, ['n'])
                    words_set = HyponymRetriever(synsets, max_level=8)
                    # add loaded list
                    self.other_sets[(target, targetType)] = words_set
                elif targetType == 'activity':
                    synsets = _get_synsets(target, ['n', 'v'])
                    words_set = HyponymRetriever(synsets, max_level=8)
                    self.other_sets[(target, targetType)] = words_set
                else:
                    return None

        score_pref = defaultdict(float)
        score_pref_sum = defaultdict(lambda: {'score': 0.0, 'count': 0})
        for sentence_tags in diary_tags_list:
            is_sent_mine = False    # 'I' is subject of sentence
            is_sent_neg = False     # the sentence is negative
            is_sent_suj = False     # the word is subject of sentence

            prev_word_list = []
            scores_word = defaultdict(float)
            weight_subj = 1     # weight of subjectivity (subject is i?, negative sent?)
            weight_sent = 0     # weight of sentiment words
            count_sent = 0      # the number of sentiment words
            adverb_neg = {'score': 0.0, 'count': 0}
            adverb_pos = {'score': 0.0, 'count': 0}

            for word in sentence_tags:
                # check roles and extract weight
                if word[TAG_WORD_POS] is not None and word[TAG_WORD_ROLE] is not None:
                    # Check the subject of sentence is I
                    if 'I' == word[TAG_WORD] and 'subj' in word[TAG_WORD_ROLE]:
                        is_sent_mine = True
                        pass

                    # Check the sentence is negative
                    elif 'neg' in word[TAG_WORD_ROLE]:
                        is_sent_neg = True
                        prev_word_list.clear()

                    # Check a sentiment score of an adverb in the sentence
                    elif word[TAG_WORD_POS].startswith('RB'):
                        offsets = _find_offsets_from_word(word[TAG_WORD], 'r')
                        for offset in offsets:
                            now_scores = self.senti_wordnet.get_score(offset, 'r')
                            if now_scores['positivity'] == 1 and now_scores['negativity'] == 0:
                                continue
                            adverb_neg['score'] += now_scores['negativity']
                            adverb_neg['count'] += 1
                            adverb_pos['score'] += now_scores['positivity']
                            adverb_pos['count'] += 1
                        prev_word_list.clear()

                    # Check a sentiment score of a verb in the sentence
                    elif 'VB' in word[TAG_WORD_POS]:
                        # find an activity
                        found_synset, lemma_word \
                            = TendencyAnalyzer._find_synset_by_word_list(words_set, [word[TAG_WORD]], False)
                        if found_synset:
                            # Frequency Weight
                            # print(found_synset, ' ', lemma_word)
                            word_count = TendencyAnalyzer._count_word_in_corpus(lemma_word, pos='n')
                            # print(word_count)
                            count_sum = 0
                            count_for_synset = 0
                            for synset_word, count in word_count.items():
                                count_sum += count + 1
                                if found_synset.name() == synset_word:
                                    count_for_synset = count + 1
                            word_freq_weight = count_for_synset / count_sum
                            # print(count_sum, count_for_synset, word_freq_weight, '\n')

                            word_score = word_freq_weight
                            synset_name = found_synset.name()
                            scores_word[synset_name] = word_score \
                                if word_score > scores_word[synset_name] \
                                else scores_word[synset_name]
                        else:
                            verb_weight = 0.5
                            sent_score = 0
                            if word[TAG_WORD_ROLE] == 'root':  # main verb...
                                verb_weight = 1
                            offsets = _find_offsets_from_word(word[TAG_WORD], 'v')
                            word_cnt = 0
                            for offset in offsets:
                                now_score = self.senti_wordnet.get_score_value(offset, 'v')
                                if now_score == 0:
                                    continue
                                else:
                                    sent_score += now_score
                                    word_cnt += 1
                            if word_cnt == 0:
                                sent_score = 0
                            else:
                                sent_score = sent_score / word_cnt
                            if sent_score is not 0:
                                weight_sent += verb_weight * sent_score
                                count_sent += 1
                        prev_word_list.clear()

                    # Check a thing or activity and their score in the sentence
                    elif word[TAG_WORD_POS].startswith('NN') and \
                            ('subj' in word[TAG_WORD_ROLE] or 'obj' in word[TAG_WORD_ROLE] or
                                 word[TAG_WORD_ROLE is 'conj']):
                        prev_word_list.append(word[TAG_WORD])

                        plural = False
                        if word[TAG_WORD].endswith('S'):
                            plural = True

                        # find a thing or an activity
                        found_synset, lemma_word \
                            = TendencyAnalyzer._find_synset_by_word_list(words_set, prev_word_list, plural)
                        if found_synset:
                            # Frequency Weight
                            # print(found_synset, ' ', lemma_word)
                            word_count = TendencyAnalyzer._count_word_in_corpus(lemma_word, pos='n')
                            # print(word_count)
                            count_sum = 0
                            count_for_synset = 0
                            for synset_word, count in word_count.items():
                                count_sum += count + 1
                                if found_synset.name() == synset_word:
                                    count_for_synset = count + 1
                            word_freq_weight = count_for_synset / count_sum
                            # print(count_sum, count_for_synset, word_freq_weight, '\n')

                            word_score = word_freq_weight
                            synset_name = found_synset.name()
                            scores_word[synset_name] = word_score \
                                if word_score > scores_word[synset_name] \
                                else scores_word[synset_name]

                            if 'subj' in word[TAG_WORD_ROLE]:
                                is_sent_suj = True
                        prev_word_list.clear()

                    # Check a sentiment score of an adjective in the sentence
                    elif 'JJ' in word[TAG_WORD_POS] and word[TAG_WORD_ROLE] == 'root':
                        adjective_weight = 2
                        sent_score = 0
                        offsets = _find_offsets_from_word(word[TAG_WORD], 'a')
                        word_cnt = 0
                        for offset in offsets:
                            now_score = self.senti_wordnet.get_score_value(offset, 'a')
                            if now_score == 0:
                                continue
                            else:
                                sent_score += now_score
                                word_cnt += 1
                        if word_cnt == 0:
                            sent_score = 0
                        else:
                            #average of sent score of sentiment word
                            sent_score = sent_score / word_cnt
                        if sent_score is not 0:
                            weight_sent += adjective_weight * sent_score
                            count_sent += 1
                        # prev_word_list.clear()

                    else:
                        prev_word_list.clear()

            # apply to score sentiemnt
            print('====== scoring ======')
            pprint(scores_word)
            # print(flag_subj_intention_tfp, weight_sent, flag_neg, sep=' | ')
            # print('============================================')
            for synset_name, word_score in scores_word.items():
                print('================')
                print('word_name: %s' % synset_name)
                print('word_score: %s' % word_score)
                # calculate weight of subjectivity
                weight_subj = 1 if is_sent_mine else (0.8 if is_sent_suj else 0.1)
                weight_subj *= -0.8 if is_sent_neg else 1
                print('weight_subj: %s' % weight_subj)

                # calculate average of weight of sentiment words
                weight_sent = weight_sent / count_sent
                if weight_sent < 0.25 and weight_sent >= 0:
                    weight_sent = 0.25  # the minimum neutral weight of sent
                print('weight_sent: %s' % weight_sent)

                sent_score = weight_subj * weight_sent
                print('sent_score: %s (subj*sent)' % sent_score)

                adverb_score = 0
                adverb_neg_score = 0
                adverb_neg_flag = False
                if sent_score >= 0:
                    if adverb_pos['count'] > 0:
                        adverb_score = sent_score * (adverb_pos['score'] / adverb_pos['count'])
                    if adverb_neg['count'] > 0 and adverb_neg['score'] > adverb_pos['score']:
                        adverb_neg_score = (adverb_score + sent_score) * (adverb_neg['score'] / adverb_neg['count']) * -1
                        adverb_neg_flag = True
                else:
                    if adverb_pos['count'] > 0:
                        adverb_score = sent_score * (adverb_pos['score'] / adverb_pos['count']) * -1
                    if adverb_neg['count'] > 0 and adverb_neg['score'] > adverb_pos['score']:
                        adverb_neg_score = (adverb_score + sent_score) * (adverb_neg['score'] / adverb_neg['count']) * -1
                        adverb_neg_flag = True
                print('adverb_score: %s' % adverb_score)
                print('adverb_neg_score %s' % adverb_neg_score)

                score = 0
                if sent_score >= 0:
                    score = (word_score * 0.5) + (sent_score * word_score * 0.5)
                else:
                    score = -(word_score * 0.5) + (sent_score * word_score * 0.5)
                print('word score with sent: %s' % score)
                score = (score + adverb_score) * (adverb_neg_score if adverb_neg_flag else 1)
                print('result score: %s' % score)

                # scores_word[synset_name] = score
                score_pref_sum[synset_name]['score'] += score
                score_pref_sum[synset_name]['count'] += 1

        for synset_name, sum_dict in score_pref_sum.items():
            if sum_dict['count'] > 0:
                score_pref[synset_name] = sum_dict['score'] / sum_dict['count']

        return score_pref

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
    def _find_synset_by_word_list(cls, collect, word_list, plural=False):
        length = len(word_list)
        for i in range(0, length):
            lemma_word = ""
            for k in range(i, length):
                if i < k:
                    lemma_word += '_'
                lemma_word += word_list[k]
            # print(lemma_word)
            synset = collect.find_word(lemma_word)
            if synset is not None:
                return synset, lemma_word

        # if lemma is not found, but the noun in lemma is plural
        if plural:
            try:
                plural_noun = wn.synsets(word_list[length-1])[0].lemmas()[0].name()
                word_list[length-1] = plural_noun
                return cls._find_synset_by_word_list(collect, word_list, False)
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
    sw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordset',
                           'SentiWordNet_3.0.0_20130122.txt')
    senti_wordnet = SentiWordNet(sw_path)

    foods = HyponymRetriever(wn.synset('food.n.01'), max_level=2)
    # pprint(foods.get_list())

    tend_analyzer = TendencyAnalyzer(senti_wordnet, food_set=foods)

    DIARIY = "I like an apple. I really like a banana. I don't like a grape. I hate a sweet potato."
    diary_tags = tagger.tag_pos_doc(DIARIY)
    pprint(diary_tags)
    print()
    print()
    pprint(tend_analyzer.scorePreferenceToDiary(diary_tags[1], 'food', 'thing'))
