from collections import defaultdict
from nltk.corpus import wordnet as wn
from pprint import pprint

from diary_analyzer import tagger
from diary_analyzer.tagger import TAG_POS_WORD, TAG_POS_DEPENDENCY, \
    TAG_POS_MORPHEME, TAG_POS_NAMED_ENTITY, TAG_POS_WORD_ROLE

class HyponymThingsCollector(object):
    """For Synsets Retriever from Hyponyms of a Word"""
    IDX_SYNSET = 0
    IDX_LEVEL = 1
    IDX_LEMMA_WORDS = 2

    def collect_hyponyms(self, synset, max_level=1):
        if max_level == 0: return []
        hyponym_list = list()
        for hyponym in synset.hyponyms():
            # names, counts = _lemmas_to_name_list(hyponym.lemmas(), True)
            # hyponym_list.append((hyponym, max_level, names, counts))
            hyponym_list.append((hyponym, max_level, _lemmas_to_name_list(hyponym.lemmas())))
            hyponym_list = hyponym_list + self.collect_hyponyms(hyponym, max_level-1)
        return hyponym_list

    def collect_hyponyms_leaf(self, synset):
        leaf_list = list()
        for hyponym in synset.hyponyms():
            if len(hyponym.hyponyms()) == 0:     # is leaf hyponym
                leaf_list.append(hyponym)
            else:
                leaf_list = leaf_list + self.collect_hyponyms_leaf(hyponym)
        return leaf_list

    @classmethod
    def _find_synset(cls, collect, synset):
        for item in collect:
            if synset.name() is item[0].name():
                return item[0]
        return None

    @classmethod
    def _check_synset_in(cls, collect, synset):
        if cls.find_synset(collect, synset) is not None:
            return True
        else:
            return False

    @classmethod
    def _find_word_synset(cls, collect, word):
        for item in collect:
            if word in item[2]:  # lemma list
                return item[0]  # synset
        return None

    @classmethod
    def _check_word_in(cls, collect, word):
        if cls.find_word_synset(collect, word) is not None:
            return True
        else:
            return False


class Finder(object):
    def find_most_lemma(self, word):
        # for synset in wn.synset('health')
        pass


class LifeStylesAnalyzer(object):
    """Perform life style analysis"""

    def analyze_food(self, food_collect, diary_tags):
        food_sentiments = defaultdict(float)
        for sentence in diary_tags:
            prev_word_list = []
            for word in sentence:
                if word[TAG_POS_WORD_ROLE] is None:
                    prev_word_list.clear()
                elif word[TAG_POS_WORD_ROLE].startswith('NN') and \
                        ('subj' in word[TAG_POS_MORPHEME] or 'obj' in word[TAG_POS_MORPHEME] or
                         word[TAG_POS_MORPHEME is 'conj']):
                    prev_word_list.append(word[TAG_POS_WORD])

                    plural = False
                    if word[TAG_POS_WORD_ROLE].endswith('S'):
                        plural = True

                    #find the word
                    found_synset, lemma_word = LifeStylesAnalyzer._find_synset_by_word_list(food_collect,
                                                                                            prev_word_list, plural)
                    if found_synset:
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

                        sentiment = 1 * word_freq_weight
                        synset_name = str(found_synset.name())
                        food_sentiments[synset_name] += sentiment
                    prev_word_list.clear()

                elif (word[TAG_POS_WORD_ROLE].startswith('JJ') and word[TAG_POS_MORPHEME] == 'amod') or \
                        (word[TAG_POS_WORD_ROLE].startswith('NN') and word[TAG_POS_MORPHEME] == 'compound'):
                    prev_word_list.append(word[TAG_POS_WORD])
                else:
                    prev_word_list.clear()
        return food_sentiments

    def analyze_hobby(self, diary_tags):
        pass

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
            synset = cls._find_lemma(collect, lemma_word)
            if synset is not None:
                return synset, lemma_word

        # if lemma is not found, but the noun in lemma is plural
        if plural:
            plural_noun = wn.synsets(word_list[length-1])[0].lemmas()[0].name()
            word_list[length-1] = plural_noun
            return cls._find_synset_by_word_list(collect, word_list, False)
        return None, None

    @classmethod
    def _find_lemma(cls, collect, lemma):
        for item in collect:
            if lemma in item[2]:  # lemma list
                return item[0]  # synset
        return None

    @classmethod
    def _check_lemma_in(cls, collect, lemma):
        if cls.find_word_synset(collect, lemma) is not None:
            return True
        else:
            return False

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


if __name__ == "__main__":
    htc = HyponymThingsCollector()
    food_collect = htc.collect_hyponyms(wn.synset('food.n.02'), 4)
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

    TEST_DIARY = """I like tomato pasta and bread. I usually have eaten sweet potatoes with sugar since childhood.
                    However, today I dated with my girlfriend and
                    ate them without sugar. It was very delicious more thant I thought!"""
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


    from diary_analyzer import sample_diaries
    diaries = []
    for diary_text in sample_diaries.NICOLEXLOVE13:
        diary_tags = tagger.tag_pos_doc(diary_text)
        diaries.append(diary_tags)
        pprint(diary_tags)
    diaries.append(tagger.tag_pos_doc(TEST_DIARY))
    #
    lsa = LifeStylesAnalyzer()
    for diary_tags in diaries:
        result = lsa.analyze_food(food_collect, diary_tags[1])
        print(result)

    # pprint(LifeStylesAnalyzer._count_word_in_corpus('bread'))
    # pprint(LifeStylesAnalyzer._count_word_in_corpus('bread', 'n'))
