from collections import defaultdict
from nltk.corpus import wordnet as wn
from pprint import pprint
import csv
import os
import operator

from diary_analyzer import tagger
from diary_analyzer.tagger import TAG_POS_WORD, TAG_POS_DEPENDENCY, \
    TAG_POS_MORPHEME, TAG_POS_NAMED_ENTITY, TAG_POS_WORD_ROLE


class HyponymThingsRetriever(object):
    """For Synsets Retriever from Hyponyms of a Word"""
    IDX_SYNSET = 0
    IDX_LEVEL = 1
    IDX_LEMMA_WORDS = 2

    def __init__(self, *root_synsets, max_level=1):   # if max level < 0, only find leaf
        self.hyponym_list = list()
        if max_level < 0:
            for synset in root_synsets:
                self.hyponym_list += self._collect_hyponyms_leaf(synset)
        else:
            for synset in root_synsets:
                self.hyponym_list += self._collect_hyponyms(synset, max_level)

    def _collect_hyponyms(self, root_synset, max_level=1):
        if max_level == 0: return []
        hyponym_list = list()
        for hyponym in root_synset.hyponyms():
            # names, counts = _lemmas_to_name_list(hyponym.lemmas(), True)
            # hyponym_list.append((hyponym, max_level, names, counts))

            # tuple (sysnet, level, lemma_words)
            hyponym_list.append((hyponym, max_level, _lemmas_to_name_list(hyponym.lemmas())))
            hyponym_list = hyponym_list + self._collect_hyponyms(hyponym, max_level-1)
        return hyponym_list

    def _collect_hyponyms_leaf(self, root_synset):
        leaf_list = list()
        for hyponym in root_synset.hyponyms():
            if len(hyponym.hyponyms()) == 0:     # is leaf hyponym
                leaf_list.append(hyponym)
            else:
                leaf_list = leaf_list + self._collect_hyponyms_leaf(hyponym)
        return leaf_list

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


class HobbyList(object):
    IDX_WORD = 0
    IDX_LEMMA_WORDS = 1
    IDX_SYNSET = 2

    def __init__(self, filepath):
        self.hobby_synset_list = list()
        self.hobby_nosynset_list = list()
        self._collect_hobbies(filepath)

    def _collect_hobbies(self, filepath):
        with open(filepath, 'r') as hobby_file:
            while True:
                hobby = hobby_file.readline().rstrip('\r\n')
                if not hobby: break

                synsets = wn.synsets(_text_to_lemma_format(hobby))

                if len(synsets) > 0:
                    lemma_list = list()
                    for synset in synsets:
                        lemma_list += _lemmas_to_name_list(synset.lemmas())
                    self.hobby_synset_list.append([hobby, lemma_list, synsets[0].name()])
                else:
                    self.hobby_nosynset_list.append([hobby])    # for compatibility and expandability
                    pass
                # hobby
            hobby_file.close()

    def get_synset_list(self):
        return self.hobby_synset_list

    def get_nosynset_list(self):
        return self.hobby_nosynset_list


class SentiWordNet(object):
    """ Load SentiWordNet """

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
            'score_pos': 0.5,
            'score_neg': 0.5
        }

        with open(filepath, 'r') as sent_file:
            csvReader = csv.reader(sent_file, delimiter='\t')
            for row in csvReader:
                if len(row) == 0 or row[0].startswith('#'):
                    continue

                score_pos = float(row[self.IDX_SCORE_POS])
                score_neg = float(row[self.IDX_SCORE_NEG])
                if score_pos == 0 and score_neg == 0:
                    continue

                score_dict = {
                    'score_pos': score_pos + 0.5,
                    'score_neg': score_neg + 0.5
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
        return score_dict['score_pos'] - score_dict['score_neg']


class Finder(object):
    def find_most_lemma(self, word):
        # for synset in wn.synset('health')
        pass


class LifeStylesAnalyzer(object):
    """Perform life style analysis"""

    def __init__(self, senti_wordnet=None,
                 food_collect=None, hobby_collect=None, sport_collect=None):
        self.senti_wordnet = senti_wordnet
        self.food_collect = food_collect
        self.hobby_collect = hobby_collect
        self.sport_collect = sport_collect

    def _analyze_noun_thing(self, collect, diary_tags):
        score_sentiments = defaultdict(float)
        for sentence in diary_tags:
            flag_subj_intention_tfp = False
            flag_neg = 1

            prev_word_list = []
            score_sentence = defaultdict(float)
            weight_sent = 0

            for word in sentence:
                if word[TAG_POS_MORPHEME] is not None and word[TAG_POS_WORD_ROLE] is not None:
                    # check morphemes
                    if 'I' == word[TAG_POS_WORD] and 'subj' in word[TAG_POS_MORPHEME]:  # subj is I
                        flag_subj_intention_tfp = True
                        prev_word_list.clear()

                    elif 'neg' in word[TAG_POS_MORPHEME]:
                        flag_neg = -0.8
                        prev_word_list.clear()

                    elif word[TAG_POS_WORD_ROLE].startswith('RB'):
                        word_score = 0
                        offsets = _find_offsets_from_word(word[TAG_POS_WORD], 'r')
                        word_cnt = 0
                        for offset in offsets:
                            now_score = senti_wordnet.get_score_value(offset, 'r')
                            if now_score == 0:
                                continue
                            else:
                                word_score += now_score
                                word_cnt += 1
                        if word_cnt == 0: word_score = 0.125
                        else: word_score = word_score / word_cnt
                        weight_sent += 0.75 * word_score
                        prev_word_list.clear()

                    elif 'VB' in word[TAG_POS_WORD_ROLE]:
                        word_weight = 1
                        word_score = 0
                        if word[TAG_POS_MORPHEME] == 'root':  # main complement
                            word_weight = 1.5
                        offsets = _find_offsets_from_word(word[TAG_POS_WORD], 'v')
                        word_cnt = 0
                        for offset in offsets:
                            now_score = senti_wordnet.get_score_value(offset, 'v')
                            if now_score == 0:
                                continue
                            else:
                                word_score += now_score
                                word_cnt += 1
                        if word_cnt == 0: word_score = 0.125
                        else: word_score = word_score / word_cnt
                        weight_sent += word_weight * word_score
                        prev_word_list.clear()

                    # check things and their score
                    elif word[TAG_POS_WORD_ROLE].startswith('NN') and \
                            ('subj' in word[TAG_POS_MORPHEME] or 'obj' in word[TAG_POS_MORPHEME] or
                             word[TAG_POS_MORPHEME is 'conj']):
                        prev_word_list.append(word[TAG_POS_WORD])

                        plural = False
                        if word[TAG_POS_WORD_ROLE].endswith('S'):
                            plural = True

                        # find the word
                        found_synset, lemma_word \
                            = LifeStylesAnalyzer._find_synset_by_word_list(collect, prev_word_list, plural)
                        if found_synset:
                            # Frequency Weight
                            # print(found_synset, ' ', lemma_word)
                            word_count = LifeStylesAnalyzer._count_word_in_corpus(lemma_word, pos='n')
                            # print(word_count)
                            count_sum = 0
                            count_for_synset = 0
                            for synset_word, count in word_count.items():
                                count_sum += count+1
                                if found_synset.name() == synset_word:
                                    count_for_synset = count+1
                            word_freq_weight = count_for_synset / count_sum
                            # print(count_sum, count_for_synset, word_freq_weight, '\n')

                            # sentiment = 1 * word_freq_weight
                            sentiment = word_freq_weight
                            synset_name = found_synset.name()
                            score_sentence[synset_name] += sentiment

                            # score to hypornyms
                            # self._score_to_hyponyms(found_synset, sentiment, score_sentiments, False)
                        prev_word_list.clear()

                    elif (word[TAG_POS_WORD_ROLE].startswith('JJ') and word[TAG_POS_MORPHEME] == 'amod') or \
                            (word[TAG_POS_WORD_ROLE].startswith('NN') and word[TAG_POS_MORPHEME] == 'compound'):
                        prev_word_list.append(word[TAG_POS_WORD])

                        if 'JJ' in word[TAG_POS_WORD_ROLE]:
                            word_weight = 1
                            word_score = 0
                            offsets = _find_offsets_from_word(word[TAG_POS_WORD], 'a')
                            word_cnt = 0
                            for offset in offsets:
                                now_score = senti_wordnet.get_score_value(offset, 'a')
                                if now_score == 0:
                                    continue
                                else:
                                    word_score += now_score
                                    word_cnt += 1
                            if word_cnt == 0: word_score = 0.125
                            else: word_score = word_score / word_cnt
                            weight_sent += word_weight * word_score

                    elif 'JJ' in word[TAG_POS_WORD_ROLE] and word[TAG_POS_MORPHEME] == 'root':
                        word_weight = 2
                        word_score = 0
                        offsets = _find_offsets_from_word(word[TAG_POS_WORD], 'a')
                        word_cnt = 0
                        for offset in offsets:
                            now_score = senti_wordnet.get_score_value(offset, 'a')
                            if now_score == 0:
                                continue
                            else:
                                word_score += now_score
                                word_cnt += 1
                        if word_cnt == 0: word_score = 0.125
                        else: word_score = word_score / word_cnt
                        weight_sent += word_weight * word_score
                        # prev_word_list.clear()

                    else:
                        prev_word_list.clear()
                # print(word[TAG_POS_WORD], ' -> ', weight_sent)

            # apply to score sentiemnt
            # print('=== scoring: ', '===')
            # pprint(score_sentence)
            # print(flag_subj_intention_tfp, weight_sent, flag_neg, sep=' | ')
            # print('============================================')
            for synset_name, sentiment in score_sentence.items():
                value = weight_sent * sentiment * flag_neg
                if flag_subj_intention_tfp:
                    value *= 2
                score_sentiments[synset_name] += value
                # score_sentiments[synset_name] += sentiment

        return score_sentiments

    def _score_to_hyponyms(self, hypernym_synset, hypernym_score, score_sentiments, continue_hyponyms=False):
        hyponyms = hypernym_synset.hyponyms()
        length = len(hyponyms)
        if hypernym_score == 0 or length == 0:
            return
        subscore = hypernym_score / length
        for hyponym in hyponyms:
            synset_name = hyponym.name()
            score_sentiments[synset_name] += subscore
            if continue_hyponyms:
                self._score_to_hyponyms(hyponym, subscore, score_sentiments)

    def analyze_food(self, diary_tags):
        return self._analyze_noun_thing(self.food_collect, diary_tags)

    def analyze_hobby(self, diary_tags):
        return self._analyze_noun_thing(self.hobby_collect, diary_tags)

    def analyze_sport(self, diary_tags):
        return self._analyze_noun_thing(self.sport_collect, diary_tags)

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


class CorrelationAnalyzer(object):
    def __init__(self, category_1_synset_names, category_2_synset_names, dist_func):
        self.category1 = category_1_synset_names
        self.category2 = category_2_synset_names
        self.dist_func = dist_func

    def analyze(self, diary_scores):

        pass

    def _find_category(self, synset_name, category):
        synset = wn.synset(synset_name)
        hypernyms = synset.hypernyms()
        for hypernym in hypernyms:
            name = hypernym.name()
            if name in category:
                return name
        for hypernym in hypernyms:
            name = hypernym.name()
            cat = self._find_category(name, category)
            if cat:
                return cat
        return None


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


def ranking_lifestyle(lifestyle_list, like):
    calculated_list = {}
    for lifestyle_item in lifestyle_list:
        if not (lifestyle_item['thing'] in calculated_list):
            calculated_list[lifestyle_item['thing']] = lifestyle_item['score']
        else:
            calculated_list[lifestyle_item['thing']] = calculated_list[lifestyle_item['thing']] + lifestyle_item['score']

    # sort by value
    ranked_list = sorted(calculated_list.items(), key=operator.itemgetter(1), reverse=like)

    # Cutting MOST 3 things
    final_result = []
    cnt = 0
    for item in ranked_list:
        if item[1] > 3:
            msg = 'very liked'
        elif 0.5 < item[1] <= 3:
            msg = 'liked'
        elif 0 < item[1] <= 0.5:
            msg = 'netural'
        elif -0.5 < item[1] <= 0:
            msg = 'disliked'
        else:
            msg = 'very disliked'
        tmp_dict = {'thing': _get_default_lemma(item[0]), 'value': item[1], 'message': msg}
        final_result.append(tmp_dict)
        # result_lemma = _get_default_lemma(item[0])
        # final_result.append(result_lemma)
        cnt += 1
        if cnt == 3:
            break

    return final_result


def _get_default_lemma(synset_name):
    return wn.synset(synset_name).lemmas()[0].name()


sw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordset',
                     'SentiWordNet_3.0.0_20130122.txt')
senti_wordnet = SentiWordNet(sw_path)

hbw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordset',
                        'hobbies_from_wikipedia.txt')
hobbies = HobbyList(hbw_path)
foods = HyponymThingsRetriever(wn.synset('food.n.02'), max_level=8)
sports = HyponymThingsRetriever(wn.synset('sport.n.01'), max_level=7)
exercise = HyponymThingsRetriever(wn.synset('exercise.n.01'), wn.synset('sport.n.01'), max_level=7)
# people
# location
# day of week
# season
analyzer = LifeStylesAnalyzer(senti_wordnet=senti_wordnet,
                              food_collect=foods, sport_collect=sports)


if __name__ == "__main__":
    # htc = HyponymThingsCollector()
    # food_collect = htc.collect_hyponyms(wn.synset('food.n.02'), 4)
    # pprint(food_collect)
    # print()

    # print(wn.synset('barmbrack.n.01').lemmas())
    # print(wn.synset('cupcake.n.01').lemmas())
    # print(wn.synset('fish.n.02').lemmas())
    # print(wn.synset('lobster.n.01').lemmas())
    # print(wn.synset('crab.n.05').lemmas())
    # print()
    #
    # synset = wn.synset('crab.n.05')
    # print(synset, synset.name(), synset.pos(), synset.offset(), synset.frame_ids(),
    #       synset.definition(), synset.examples(), synset.lexname(), sep=' | ')
    # lemma = wn.lemma('crab.n.05.crab')
    # print(lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
    #       lemma.frame_strings(), lemma.count(), sep=' | ')
    # print()


    # a = defaultdict(float)
    # key = 'past.n.02'
    # print(type(key))
    # a[key] = a[key] + 3
    # print(a)
    # print()


    # print(wn.synset("dog.n.01").lemmas())
    # print(wn.synsets("gorgonzola"))
    # print(wn.synsets("Potato"))
    # print(wn.synsets("Sweet_Potato"))
    # print(wn.synsets("Sweet_Potatoes"))
    # print()

    # for synset in wn.synsets('game'):
    #     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
    #           synset.examples(), synset.lexname(), sep=' | ')
    # print()
    # for synset in wn.synsets('date'):
    #     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
    #           synset.examples(), synset.lexname(), sep=' | ')
    # print()

    # foods = HyponymThingsCollector(wn.synset('food.n.02'), 8)

    # pprint(foods.get_list())
    # print()
    # print()
    # print()
    # pprint(sports.get_list())
    # print()
    # print()
    # print()

    TEST_DIARY = """I like eating potato as a snack. I usually have eaten potatoes with sugar since childhood. However, today I ate them without sugar because of my girlfriend's suggestion. It was very delicious more than I thought!"""
    # TEST_DIARY = """I usually have eaten apples with sugar since childhood."""
    TEST_DIARY2 = """My main course was a half the dishes. Cumbul Ackard Cornish card little gym lettuce. Fresh Peas Mousser on mushrooms, Cocles and a cream sauce finished with a drizzle of olive oil wonderfully tender, and moist card. But I'm really intensify the flavor of the card there by providing a nice flavor contrast to the rich cream sauce. Lovely freshness, and texture from the little gym lettuce. A well executed dish with bags of flavour. Next, a very elegant vanilla, yogurt and strawberries and Candy Basil different strawberry preparations delivered a wonderful variety of flavor. Intensities is there was a sweet and tart lemon curd and yogurt sorbet buttery, Pepper Pastry Cramble Candied Lemons. Testing broken mrang the lemon curd had a wonderfully creamy texture and then ring was perfectly light and Chrissy and wonderful dessert with a great balance of flavors and textures. It's got sweetness. It's got scrunch. It's got acidity. It's got freshness."""
    # for synset in wn.synsets('potatoes'):
    #     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
    #           synset.examples(), synset.lexname(), sep=' | ')
    # print()
    #
    # diary_tags = tagger.tag_pos_doc(TEST_DIARY)[1]
    # pprint(diary_tags)
    # print()
    #
    # lsa = LifeStylesAnalyzer()
    # result = lsa.analyze_food(food_collect, diary_tags)
    # print(result)
    # print()


    # from diary_analyzer import sample_diaries
    diaries = []
    # diaries.append(tagger.tag_pos_doc("The apple is delicious."))
    # diaries.append(tagger.tag_pos_doc("I like apple."))
    # diaries.append(tagger.tag_pos_doc("I've liked apple."))
    # diaries.append(tagger.tag_pos_doc("I liked apple."))
    # diaries.append(tagger.tag_pos_doc("I don't like apple."))
    # diaries.append(tagger.tag_pos_doc("I like tomato pasta."))
    # diaries.append(tagger.tag_pos_doc("I hate tomato pasta."))
    # diaries.append(tagger.tag_pos_doc("I ate bread."))
    # diaries.append(tagger.tag_pos_doc("I like bread."))
    # diaries.append(tagger.tag_pos_doc("I don't like bread."))
    # diaries.append(tagger.tag_pos_doc("I love bread."))
    # diaries.append(tagger.tag_pos_doc("I really love bread."))
    # diaries.append(tagger.tag_pos_doc("I hate bread."))
    # diaries.append(tagger.tag_pos_doc("I did not like apple and pineapple."))
    # for diary_text in sample_diaries.NICOLEXLOVE13:
    #     diary_tags = tagger.tag_pos_doc(diary_text)
    #     diaries.append(diary_tags)
        # pprint(diary_tags)
    # test_tags = tagger.tag_pos_doc(TEST_DIARY)
    # test_tags = tagger.tag_pos_doc(TEST_DIARY2)
    #
    # pprint (test_tags)
    # diaries.append(test_tags)

    # test_tags = tagger.tag_pos_doc("Today I went Soongsil Univerity with James.", True)
    # pprint(test_tags)

    # pprint(tags)
    # print()
    # for diary_tags in diaries:
    #     print(diary_tags[0])
    #     print(diary_tags[1])
    #     result = analyzer.analyze_food(diary_tags[1])
    #     print(result)
    #     print()
    # print()
    # for diary_tags in diaries:
    #     result = analyzer.analyze_sport(diary_tags[1])
    #     print(result)
    # print()

    # pprint(LifeStylesAnalyzer._count_word_in_corpus('bread'))
    # pprint(LifeStylesAnalyzer._count_word_in_corpus('bread', 'n'))

    # pprint(tagger.tag_pos_doc("There is an apple."))

    # pprint(wn.synset("cheese.n.01").lemmas()[0])
    # pprint(wn.synset("cheese.n.01").lemmas()[0].name())

    # pprint(wn.synset("pasta.n.02").hyponyms())
    # pprint(wn.synset("noodle.n.01").hypernyms())

    # pprint (wn.synsets("semantics"))
    # print()
    # pprint(tagger.tag_pos_doc("I'm reading."))
    # print()

    # pprint(hobbies.get_synset_list())
    # pprint(hobbies.get_nosynset_list())

    wish = HyponymThingsRetriever(wn.synset('wish.v.02'), max_level=8)
    like = HyponymThingsRetriever(wn.synset('like.v.02'), max_level=8)
    good1 = HyponymThingsRetriever(wn.synset('good.a.01'), max_level=8)
    good2 = HyponymThingsRetriever(wn.synset('good.s.06'), max_level=8)
    good3 = HyponymThingsRetriever(wn.synset('good.s.07'), max_level=8)

    pprint(wish.get_list())
    print()
    print()
    print()
    print()
    print()
    pprint(like.get_list())
    print()
    print()
    print()
    print()
    print()
    pprint(good1.get_list())
    print()
    print()
    print()
    print()
    print()
    pprint(good2.get_list())
    print()
    print()
    print()
    print()
    print()
    pprint(good3.get_list())
