import re
from collections import defaultdict

import nltk
from nltk.parse import stanford
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize, sent_tokenize

from diary_nlp.nlp_en_tmp import settings


class RoleFinder:
    def __init__(self):
        import os
        self.dep_parser = nltk.parse.stanford.StanfordDependencyParser(
            path_to_jar=settings.PATH_ST_PARSER,
            path_to_models_jar=settings.PATH_ST_PARSER_MODEL)
        self.parser = nltk.parse.stanford.StanfordParser(
            path_to_jar=settings.PATH_ST_PARSER,
            path_to_models_jar=settings.PATH_ST_PARSER_MODEL,
            encoding='utf-8')
        self.ner_tagger = StanfordNERTagger(
            model_filename=os.path.join(settings.PATH_ST_NER, 'classifiers', 'english.all.3class.distsim.crf.ser.gz'),
            path_to_jar=os.path.join(settings.PATH_ST_NER, 'stanford-ner.jar'), encoding='utf-8')

    def dep_tagger(self, sentence):
        tokens = word_tokenize(sentence)
        t_idx = 0
        dep_tagged = [['0', str(t_idx), ' ', 'ROOT', '-1', 'ROOT']]
        dep_parsed = next(self.dep_parser.raw_parse(sentence))
        dep_relations = dep_parsed.to_conll(4).split('\n')
        print(dep_relations)
        for idx, dep_rel in enumerate(dep_relations):
            dep_tag = re.sub(r'(\t)', ',', dep_rel).split(',')
            while t_idx < len(dep_relations):
                if t_idx == len(tokens):
                    break
                token = tokens[t_idx]
                if token != dep_tag[0]:
                    dep_tagged.append([str(t_idx+1), token, 'PUNCT', str(t_idx), 'PUNCT'])
                    t_idx += 1
                else:
                    dep_tagged.append([str(t_idx+1)] + dep_tag)
                    t_idx += 1
                    break
        # dep_parsed.tree().draw()
        return dep_tagged


    def find_role(self, text):
        sentences = sent_tokenize(text)
        for sentence in sentences:
            pos_tagged = self.parser.raw_parse(sentence)
            # next(pos_tagged).draw()
            roles = defaultdict(list)
            print(sentence)
            # dependency_tag
            dep_tagged = self.dep_tagger(sentence)
            # 동사 찾기
            print(dep_tagged)

            for candidate in filter(lambda v: v[4] == 'root', dep_tagged):
                roles['ROOT'] = candidate
                print(candidate)
                # noun
                if candidate[2][:2] == 'NN':
                    print("root is NN - Object or Complement")
                    for candidate2 in filter(lambda v: v[3] == candidate[0], dep_tagged):
                        print(candidate2)
                        if candidate2[4] == 'cop':
                            roles['VERB'] = candidate2
                        if candidate2[4] == 'ccop':
                            roles['OBJECT'] = candidate2
                #adj
                if candidate[2] == 'SG':
                    pri+\
                    nt("root is ADV")
                    for candidate2 in filter(lambda v: v[3] == candidate[0], dep_tagged):
                        print(candidate2)
                #verb
                if candidate[2][:2] == 'VB':
                    print("root is VERB")
                    roles['VERB'].append(candidate)
                    for candidate2 in filter(lambda v: v[3] == candidate[0], dep_tagged):
                        print(candidate2)

            for candidate in filter(lambda v: v[3] == roles['VERB'][0][0], dep_tagged):
                if candidate[4] == 'nsubj':
                    roles['SUBJECT'] = candidate
            print(roles['SUBJECT'], roles['VERB'])
            # print(roles['ROOT'])
            # # 동사 확장
            # for verb in filter(lambda v: v[3] == roles['ROOT'][0][0], dep_tagged):
            #     print(verb)


def demo():
    print("nltk version should be 3.1. current version is ", nltk.__version__, )
    file = open('/'.join([settings.PATH_BASE, 'example1']), mode='r', encoding='utf-8')
    text = file.read()
    text = """There must be a better mexican place in Rockland"""
    text = """The problem is that this has never been tried."""

    rf = RoleFinder()
    rf.find_role(text)


if __name__ == '__main__':
    demo()

    # to get simple pos tag
    # use
    # simplepos2 = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in pos_tag(tokens2)]
    # print(simplepos2)