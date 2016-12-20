import os;
from pprint import pprint
from diary_analyzer import tagger
from diary_analyzer.tools import ner_tagger

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import csv

# SAMPLE_DIARY = """I had to rush my mom to the hospital today, and it was the scariest thing ever. It really made me realize how much I love my mom. I cried so much before going into the hospital. Turns out she has a kidney stone on each kidney, and that's nothing good, but it's better than some things that it could have been. We went in not knowing what was wrong, and i'm so glad it wasn't something that could result in death. I have never been that scared before and I will never again take my moms name in vain."""
# pprint(pos_tagger.tag_pos_doc(SAMPLE_DIARY))


# text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'
# pos_texts = tagger.tag_pos_doc(text, True)
# pprint(pos_text)

# tagger.tags_to_pickle(pos_texts, "/Users/hanter/SEL/Smart Diary System/smartdiarysystem/pickles/test.pkl")
# pickle = tagger.pickle_to_tags("/Users/hanter/SEL/Smart Diary System/smartdiarysystem/pickles/test.pkl")
# pprint(pickle)


# for synset in wn.synsets('dog'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
# print()
#
#
# for synset in wn.synsets('chase'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
# print()
#
#
# for lemma in wn.synset('chase.v.01').lemmas():
#     print(lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#           lemma.frame_strings(), lemma.count(), sep=' | ')
# print()
#
#
# for synset in wn.synsets('unrecognized'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
# print()
#
# for lemma in wn.synset('unrecognized.s.01').lemmas():
#     print(lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#           lemma.frame_strings(), lemma.count(), sep=' | ')
# print()
#
#
# for lemma in wn.synset('relative.a.01').lemmas():
#     print(lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#           lemma.frame_strings(), lemma.count(), sep=' | ')
# print()

# print("### Food Synsets ###")
# for synset in wn.synsets('food'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
# print()
# print("### Food Lemmas ###")
# for lemma in wn.synset('food.n.02').lemmas():
#     print(lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#           lemma.frame_strings(), lemma.count(), sep=' | ')
# print()
# print("### Food Hyponyms ###")
# print(len(wn.synset('food.n.02').hyponyms()))
# for synset in wn.synset('food.n.02').hyponyms():
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
# print()
# print("### Pasta Hyponyms ###")
# print(len(wn.synset('health_food.n.01').hyponyms()))
# for synset in wn.synset('health_food.n.01').hyponyms():
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
# print()


# for synset in wn.synsets('date'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()
#
# for synset in wn.synsets('game'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()

# for synset in wn.synsets('food'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()
#
# for synset in wn.synsets('sports'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()

# for synset in wn.synsets('hobby'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()
#
# for synset in wn.synsets('act'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()
#
# for synset in wn.synsets('activity'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()

# for synset in wn.synsets('delicious'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame
# _strings(), lemma.count(), sep=' | ')
#     print()
# print()


# asd = wn.synsets('adsf')
# pprint(asd)

# for synset in wn.synsets('dsfasdfadfsa'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()
#
# for synset in wn.synsets('work'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()

# cafe.n.01
# restaurant.n.01
# building.n.01
# structure.n.01
# artifact.n.01


# for synset in wn.synset('sushi.n.01').hypernyms():
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()

# for synset in wn.synset('bad_weather.n.01').hyponyms():
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()
#
# for synset in wn.synset('shooting.n.01').hyponyms():
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()

def nounify(verb_word):
    set_of_related_nouns = list()
    for lemma in wn.lemmas(wn.morphy(verb_word, wn.VERB), pos="v"):
        for related_form in lemma.derivationally_related_forms():
            for synset in wn.synsets(related_form.name(), pos=wn.NOUN):
                # if wn.synset('person.n.01') in synset.closure(lambda s:s.hypernyms()):
                set_of_related_nouns.append(synset)
    return set_of_related_nouns

def nounify2(verb_synset):
    set_of_related_nouns = list()
    for lemma in verb_synset.lemmas():
        for related_form in lemma.derivationally_related_forms():
            for synset in wn.synsets(related_form.name(), pos=wn.NOUN):
                set_of_related_nouns.append(synset)
    return set_of_related_nouns

# for synset in wn.synsets('side_dish'):
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     # for lemma in synset.lemmas():
#     #     print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#     #           lemma.frame_strings(), lemma.count(), sep=' | ')
#     # print()
# print()

# print(wn.synset('coffee.n.01').lemmas()[0].name())

# print(nounify('sketch'))
# print(nounify2(wn.synset('macrame.v.01')))
# print(nounify2(wn.synset('sculpt.v.01')))


# synset = wn.synset('gardening.n.01')
# print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#       synset.examples(), synset.lexname(), sep=' | ')
# for lemma in synset.lemmas():
#     print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#           lemma.frame_strings(), lemma.count(), sep=' | ')
# print()
#

# for synset in wn.synset('dislike.v.01').hypernyms():
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()
# for synset in wn.synset('dislike.v.01').hyponyms():
#     print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
#           synset.examples(), synset.lexname(), sep=' | ')
#     for lemma in synset.lemmas():
#         print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
#               lemma.frame_strings(), lemma.count(), sep=' | ')
#     print()
# print()

# synset_u = wn.synset('egg.n.02')
# synset_v = wn.synset('juice.n.01')
# distance = synset_u.shortest_path_distance(synset_v,
#         simulate_root=True and synset_u._needs_root())
# print(distance)
# pprint(tagger.tag_pos_doc("That puzzles is so fun for us."))
# pprint(tagger.tag_pos_doc("The puzzle makes us feel excited."))
# pprint(tagger.tag_pos_doc("I like apple and banana."))
# pprint(tagger.tag_pos_doc("I like apple and hate grape."))
# pprint(tagger.tag_pos_doc("I like apple which is well aged."))
# pprint(tagger.tag_pos_doc("I like apple but I don't like pineapple."))
# pprint(tagger.tag_pos_doc("I like soccer to enhance friendship."))
# pprint(tagger.tag_pos_doc("The apple and banana was very delicious, but grape wasn't."))
# pprint(tagger.tag_pos_doc("The banana was really good to me."))
# pprint(tagger.tag_pos_doc("Definitely this bread looks delicious."))
# pprint(tagger.tag_pos_doc("I know what you want."))
# pprint(tagger.tag_pos_doc("What is this very delicious food?"))
# pprint(tagger.tag_pos_doc("I was repeatedly molested and raped by my stepfather, until I became pregnant when I was sixteen."))
# pprint(tagger.tag_pos_doc("But recently, I've discovered that what many of us think of as great leardership doesn not work when it comes to leading innovation."))

# for i in range(0, 10)[::-1]:
#     print(i)


import pickle
### update pickles
for i in range(1, 56):
    file_path = "diary_pickles/as_travel_" + str(i) + ".pkl"
    try:
        pickle_file = open(file_path, mode='rb+')
        tags = pickle.load(pickle_file, encoding="utf-8")
        pickle_file.close()

        for sent in tags[1]:
            for entity in sent:
                if entity[1] is None:
                    entity[1] = ''
                if entity[3] is None:
                    entity[3] = ''

        pickle_file = open(file_path, mode='wb+')
        pickle.dump(obj=tags, file=pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_file.close()

    except Exception as e:
        print(e)
