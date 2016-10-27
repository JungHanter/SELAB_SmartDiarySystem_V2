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

for synset in wn.synsets('delicious'):
    print(synset, synset.pos(), synset.offset(), synset.frame_ids(), synset.definition(),
          synset.examples(), synset.lexname(), sep=' | ')
    for lemma in synset.lemmas():
        print('\t', lemma, lemma.name(), lemma.syntactic_marker(), lemma.frame_ids(),
              lemma.frame_strings(), lemma.count(), sep=' | ')
    print()
print()

