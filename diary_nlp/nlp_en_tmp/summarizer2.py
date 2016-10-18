from collections import defaultdict, Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer

from diary_nlp.nlp_en_tmp import settings


def frequent_words(sentences):
    tokenizer = RegexpTokenizer(r'\w+')
    frequency = defaultdict(int)
    for sentence in sentences:
        for token in tokenizer.tokenize(sentence):
            frequency[token] += 1
    return frequency


def sent_frequent_words(sentences, frequency, n):
    rank = defaultdict(int)
    maximum = float(max(frequency.values()))
    for word in frequency.keys():
        frequency[word] /= maximum

    for idx, sentence in enumerate(sentences):
        for token in word_tokenize(sentence):
            rank[idx] += frequency[token]

    common = Counter(rank).most_common(int(n))
    return [sentences[freq_idx] for idx, (freq_idx, freq) in enumerate(common)]


def demo(_text):
    sentences = sent_tokenize(_text)
    frequency = frequent_words(sentences)
    sent_list = sent_frequent_words(sentences, frequency, int(len(sentences)/3))
    print("summary :\n", sent_list)
    print("frequency :\n", Counter(frequency).most_common(int(len(sentences)/3)))


if __name__ == '__main__':
    print("nltk version should be 3.1. current version is ", nltk.__version__, )
    # file = open('/'.join([settings.PATH_BASE, '1660-06-01']), mode='r', encoding='utf-8')
    # text = file.read()
    text = """
    The WORST thing ever in the history of the world has happened!!\n\nMy dad took away my phone!!!! Can you BELIEVE that???\n\nI am pretty sure this is what my history teacher means when she talks about "cruel and unusual punishment." Because taking someone\'s phone away JUST because they were texting their friends is definitely up there with bread and water for life.\n\nI get why there\'s no texting at school. It\'s distracting or whatever. But EXCUSE me!!! I was in my own home!! I NEED to be distracted from the KA-RAY-ZEE all around me!!!\n\nUGH!!!\n\nI was minding my own business, texting Chloe and Zoey about how we should handle Twin Day, since there\'s three of us. And I GUESS Brianna had been asking me to play her Princess Sugar Plum video game for a while, but I was choosing not to hear her. (BTW, my parents do this selective hearing thing ALL THE TIME. But when I do it, I\'m "rude" and "unkind" and "obsessed with my phone.")\n\nSo, my dad LOST HIS MIND!!! He waved his arms around in the air and jumped up and down like an angry gorilla I saw in a documentary once. And he shouted, "No phones! No video games! No screens!"\n\nBut, I was definitely going to need my phone to call for the loony bin to take him away!! I was pretty sure he was going to start picking bugs out of my hair next. If I had bugs in my hair. Which I DON\'T!!!\n\nHe told us spring had arrived and kids need fresh air and we better play together peacefully outside, or we\'d never see a screen again. (My mom is away on a getaway trip with her college sorority sisters. I don\'t think my dad is a huge fan of being a single parent.)\n\nSo THEN I was shoved out the door with Brianna, who was wailing about her precious Princess Sugar Plum game. And let me tell you, spring is NOT the same as summer. "Duh, Nikki," you say. But it was FREEZING out there! I\'m just saying!!\n\nAlso, I\'m pretty sure it ISN\'T spring yet. Like, technically. I think the spring equinox is later in the month, but I couldn\'t look it up because I DIDN\'T HAVE MY PHONE!!!\n\nI stomped off the porch, straight into a puddle. Of COURSE I wasn\'t wearing boots or anything, since I had no plans to hang out in nature today, so my sneakers filled right up with water. My dad might have let me change my shoes, but Brianna was blocking the door and I had to get away from her hissy fit. I was kind of having a hissy fit too, but I also had the DECENCY to be quiet about it!\n\nSoggy shoes and all, I stomped further into the yard. Then Brianna screamed, "Nikki, FREEZE!!!" and there was no way I could ignore it.\n\nI froze. It sounded like Brianna knew something I didn\'t, like I was about to step on a wasp nest. Or a sleeping MacKenzie.\n\nI looked around and didn\'t see anything. "What is it, Brianna?"\n\nShe\'d completely forgotten about her video game. She ran over to me and got on her knees in the muddy grass.\n\n"What are you doing??"\n\n"Nikki, look."\n\nSo I squatted down and followed her crazy eyes to a flower. A crocus, I think.\n\n"The first flower of spring," she whispered, reaching out her hand.\n\n"Careful," I said. I mean, it was a really sweet nature moment, but I also know that Brianna is the kind of kid who gives a goldfish a bubble bath, so I was bracing for her to squash it without meaning to.\n\nBut she just barely brushed it with the tip of her finger and stared at it in wonder.\n\nI kind of wished I had my phone to take a picture of it and send it to my friends. I also wanted to look up the kind of flower, to be sure. Also to check when the spring equinox is, so I could officially tell my dad he\'s wrong about spring.\n\nBut I didn\'t have my phone. So I sat there with Brianna and looked at the first flower of spring. It was nice.\n\nFor about a minute.\n\nThen we concocted a plan to sneak back inside the house without Dad noticing. But hey! We were working together! And we didn\'t even use any screens!!
    """
    demo(text)


# class Summarizer:
#     def __init__(self):
#         pass
#
#     def summarize(self, document):
#         """ Summarize document
#
#         :param document: daily diary
#         :return: summarized diary
#         """
#         sentences = sent_tokenize(document)
#         frequency = self.frequent_words(sentences)
#         sent_list = self.sent_frequent_words(sentences, frequency, int(len(sentences) / 3))
#         return ' '.join(sent_list)
#
#     def frequent_words(self, sentences):
#         """ calculate frequency
#
#         :param sentences: sentences of dictionary
#         :return: frequent words in dictionary format
#         """
#         tokenizer = RegexpTokenizer(r'\w+')
#         frequency = defaultdict(int)
#         for sentence in sentences:
#             for token in tokenizer.tokenize(sentence):
#                 frequency[token] += 1
#         return frequency
#
#     def sent_frequent_words(self, sentences, frequency, limit):
#         """ filter out less popular sentences
#
#         :param sentences: sentences of document
#         :param frequency: {(word: frequency)...} dictionary
#         :param limit: limit
#         :return:
#         """
#         rank = defaultdict(int)
#         maximum = float(max(frequency.values()))
#         for word in frequency.keys():
#             frequency[word] /= maximum
#
#         for idx, sentence in enumerate(sentences):
#             for token in word_tokenize(sentence):
#                 rank[idx] += frequency[token]
#
#         common = Counter(rank).most_common(min(len(sentences), limit))
#         return [sentences[freq_idx] for idx, (freq_idx, freq) in enumerate(common)]



            #     # summarizer
# print("\n summarize example")
# raw_document = documents[1]
# sum = Summarizer()
# summarized = sum.summarize(raw_document)
#
# print("\nraw : ")
# for raw in sent_tokenize(raw_document):
#     print(raw)
# print("\nsummarized :")
# for summed in sent_tokenize(summarized):
#     print(summed)