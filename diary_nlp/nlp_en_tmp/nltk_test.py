import os
import re

import nltk
from nltk.parse import stanford  # Parser, DependencyParser, NeuralDependencyParser
from nltk.tag import StanfordNERTagger

from diary_nlp.nlp_en_tmp import settings


# http://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
# http://textminingonline.com/getting-started-with-sentiment-analysis-and-opinion-mining


def separate_text(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    return sentences


def separate_segment(sentence):
    # 중국어의 경우 StanfordSegmenter 사용해야 합니다.
    # segmenter = StanfordSegmenter(path_to_jar, path_to_sihan_corpora_dict,path_to_dict)
    # segmenter.segment(sentence)
    tokens = nltk.tokenize.word_tokenize(sentence)
    # TreebankWordTokenizer, PunktWordTokenizer, WordPunctTokenizer 참조
    return tokens

# http://cs.williams.edu/~andrea/cs108/Lectures/NLP/NLP.html
# diary_nlp = generation + understanding
# understanding


def find_gra(dep_rel_list, gra):
    dep = []
    for dep_relation in dep_rel_list:
        if dep_relation[3] == gra:
            dep.append(dep_rel_list)
    return dep


def separate_small_sent(sent_tree, result_set=None):
    if result_set is None:
        result_set = []
    if isinstance(sent_tree, nltk.Tree):
        front_np = False
        sentence = []
        for sub_sent_tree in sent_tree:
            if isinstance(sub_sent_tree, nltk.Tree):
                if sub_sent_tree.label() == 'NP':
                    front_np = True
                    sentence.append(sub_sent_tree)
                elif sub_sent_tree.label() == 'VP':
                    if front_np:
                        sentence.append(sub_sent_tree)
                        new_sent = nltk.Tree('S', sentence)
                        result_set.append(new_sent)
                separate_small_sent(sub_sent_tree, result_set)
    return result_set


def traverse(tree):
    print([tree])
    for sub_tree in tree:
        if type(sub_tree) is not str:
            traverse(sub_tree)
    return


def traverse_dep(dep_rel_list, dep_to_list, token_list, idx):
    for dep in dep_rel_list:
        if int(dep[2]) == idx:
            dep_to_list.append(dep)
            for _idx, token in enumerate(token_list):
                if token == dep[0]:
                    traverse_dep(dep_rel_list,  dep_to_list, token_list, _idx+1)
    return


def nlp_process(text):
    dep_parser = nltk.parse.stanford.StanfordDependencyParser(
        path_to_jar=settings.PATH_ST_PARSER,
        path_to_models_jar=settings.PATH_ST_PARSER_MODEL)
    parser = nltk.parse.stanford.StanfordParser(
        path_to_jar=settings.PATH_ST_PARSER,
        path_to_models_jar=settings.PATH_ST_PARSER_MODEL, encoding='utf-8')
    ner_tagger = StanfordNERTagger(
        model_filename=os.path.join(settings.PATH_ST_NER, 'classifiers', 'english.all.3class.distsim.crf.ser.gz'),
        path_to_jar=os.path.join(settings.PATH_ST_NER, 'stanford-ner.jar'), encoding='utf-8')

    sentences = separate_text(text)
    for sentence in sentences:
        # print(ner_tagger.tag(sentence))

        tokens = nltk.word_tokenize(sentence)
        # 분석 대상 고르기
        pos_tagged = nltk.pos_tag(tokens)
        noun_idx = [idx for idx, (noun, pos_tag) in enumerate(pos_tagged) if pos_tag[:2] == 'NN']
        pro_noun_idx = [idx for idx, (pronoun, pos_tag) in enumerate(pos_tagged) if pos_tag[:2] == 'PRN']

        # dependency 분석
        dep_parsed = next(dep_parser.raw_parse(sentence))
        dep_relations = dep_parsed.to_conll(4).split('\n')
        dep_of_list = [[' ', 'S', '-1', 'root']]
        for dep_relation in dep_relations:
            dep_rel = re.sub(r'(\t)', ',', dep_relation).split(',')
            if len(dep_rel) == 4:
                dep_of_list.append(dep_rel)
        for idx, dep in enumerate(dep_of_list):
            print(idx, " ", dep)
        # 문장의 주어
        main_subj = []
        dep_parsed.tree().draw()
        for subject in filter(lambda s: s[3] == 'nsubj', dep_of_list):
            print(tokens[int(subject[2])])

            # for idx, noun in enumerate(tokens):
            #     if noun in [_noun for idx, _noun in enumerate(tokens) if idx in noun_idx]:
            #         for dep in filter(lambda d: d[0] == noun, dep_of_list):



def demo():
    print("nltk version should be 3.1. current version is ", nltk.__version__, )
    file = open('/'.join([settings.PATH_BASE, '1660-06-01']), mode='r', encoding='utf-8')
    text = file.read()
    # 받아 적은 내용
    # text = 'The President said he will ask Congress to increase grants to states for vocational rehabilitation'
    # text = """While hunting in Africa, I shot an elephant in my pajamas. How he got into my pajamas, I don't know."""
    # text = """It is inefficient in the way it blindly expands categories without checking whether they are compatible with the input string, and in repeatedly expanding the same non-terminals and discarding the results.."""

    # text = "Having done such hard jobs, I noticed upcoming danger."
    nlp_process(text)


if __name__ == '__main__':
    demo()
