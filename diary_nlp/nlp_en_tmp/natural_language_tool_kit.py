import re

import nltk
from nltk.parse import stanford  # Parser, DependencyParser, NeuralDependencyParser

from diary_nlp.nlp_en_tmp import settings


# http://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
# sentiment 분석
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
    front_np = False

    if result_set is None:
        result_set = []

    for sub_tree in sent_tree:
        if isinstance(sub_tree, nltk.Tree):
            front_np = True
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
        settings.PATH_ST_PARSER, settings.PATH_ST_PARSER_MODEL)
    parser = nltk.parse.stanford.StanfordParser(
        settings.PATH_ST_PARSER, settings.PATH_ST_PARSER_MODEL, encoding='utf-8')
    neural_dep_parser = nltk.parse.stanford.StanfordNeuralDependencyParser(
        settings.PATH_ST_CORE, settings.PATH_ST_CORE_MODEL)

    ner_tagger = nltk.tag.StanfordNERTagger(
        model_filename='/'.join([settings.PATH_ST_NER, 'classifiers/english.all.3class.distsim.crf.ser.gz']),
        path_to_jar='/'.join([settings.PATH_ST_NER, 'stanford-ner.jar']), encoding='utf-8')
    # pos_tagger = nltk.tag.stanford.StanfordPOSTagger(
    #     '/'.join([settings.PATH_ST_TAGGER, 'models/english-bidirectional-distsim.tagger']),
    #     '/'.join([settings.PATH_ST_TAGGER, 'stanford-postagger.jar']))
    # 문장 나누기
    sentences = separate_text(text)

    for sentence in sentences:
        # 분석 대상 설정
        tokens = nltk.tokenize.word_tokenize(sentence)

        poses = nltk.pos_tag(tokens)
        nouns = [word for (word, pos) in poses if pos[:2] == 'NN']
        # ner = ner_tagger.tag(tokens)  # 왜 오류가 날까
        print(nouns)

        # 의존 관계 분석
        dep_parsed = next(dep_parser.raw_parse(sentence))
        dep_relations = dep_parsed.to_conll(4).split('\n')
        dep_of_list = []
        dep_to_list = []
        for dep_relation in dep_relations:
            dep_rel = re.sub(r'(\t)', ',', dep_relation).split(',')
            if len(dep_rel) == 4:
                dep_of_list.append(dep_rel)
        traverse_dep(dep_of_list,  dep_to_list, tokens, 0)

        print(dep_of_list)
        print(dep_to_list)


        # 대상의 행동
        for idx, noun in enumerate(tokens):
            if noun in nouns:
                print(noun, "은 ", end="")
                for dep in filter(lambda d : d[0]==noun, dep_of_list):
                    # if tokens[int(dep[2])-1] == 'to':  # to 부정사
                    print(tokens[int(dep[2])-1], end="")
                    if dep[3] == 'nsubj':
                        print("의 주어입니다")
                    elif dep[3] == 'dobj':
                        print("의 직접목적어 입니다")
                    elif dep[3] == 'nmod':
                        print("에 의해 부사처럼 쓰입니다")
                    elif dep[3] =='compound':
                        print("와 한 단어처럼 읽을 수 있습니다")
                    elif dep[3] =='acl':
                        print("한 상태입니다.")
                    elif dep[3] == 'root':
                        print(" 문장의 중심 동사입니다.")
                    elif dep[3] == 'conj':
                        print("에 의해 다른 것과 묶입니다.")
                    else:
                        print(dep[3])


        # 주어 찾기
        # subject = find_gra(dep_rel_list, 'nsubj')
        # print("주어는 " + subject[0][0] + " 입니다")
        # main_verb = find_gra(dep_rel_list, 'root')
        # print("동사는 " + main_verb[0][0] + " 입니다")
        # role_list = find_roles(dep_rel_list, token_list, 'root')
        # print(role_list)
        # find_roles(dep_rel_list)


def main():
    file = open('/'.join([settings.PATH_BASE, 'example1']), mode='r', encoding='utf-8')
    text = file.read()
    # 받아 적은 내용
    text = 'The President said he will ask Congress to increase grants to states for vocational rehabilitation'
    # text = """While hunting in Africa, I shot an elephant in my pajamas. How he got into my pajamas, I don't know."""
    # text = """It is inefficient in the way it blindly expands categories without checking whether they are compatible with the input string, and in repeatedly expanding the same non-terminals and discarding the results.."""

    # text = "Having done such hard jobs, I noticed upcoming danger."
    nlp_process(text)


if __name__ == '__main__':
    main()
