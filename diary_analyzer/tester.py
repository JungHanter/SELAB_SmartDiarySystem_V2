import os;
from pprint import pprint
from diary_analyzer import tagger
from diary_analyzer.tools import ner_tagger

from nltk.tokenize import word_tokenize

# SAMPLE_DIARY = """I had to rush my mom to the hospital today, and it was the scariest thing ever. It really made me realize how much I love my mom. I cried so much before going into the hospital. Turns out she has a kidney stone on each kidney, and that's nothing good, but it's better than some things that it could have been. We went in not knowing what was wrong, and i'm so glad it wasn't something that could result in death. I have never been that scared before and I will never again take my moms name in vain."""
# pprint(pos_tagger.tag_pos_doc(SAMPLE_DIARY))


# text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'
# pos_texts = tagger.tag_pos_doc(text, True)
# pprint(pos_text)

# tagger.tags_to_pickle(pos_texts, "/Users/hanter/SEL/Smart Diary System/smartdiarysystem/pickles/test.pkl")
pickle = tagger.pickle_to_tags("/Users/hanter/SEL/Smart Diary System/smartdiarysystem/pickles/test.pkl")

pprint(pickle)
