import pickle, os
import nltk
from copy import copy
from diary_analyzer.tools import dep_parser, ner_tagger

TAG_POS_WORD = 0
TAG_POS_WORD_ROLE = 1
TAG_POS_DEPENDENCY = 2
TAG_POS_MORPHEME = 3
TAG_POS_NAMED_ENTITY = 4


def tag_pos_doc(document, ne_tag=False):
    document_lines = document.splitlines()
    sentences = []
    for line in document_lines:
        sentences.extend(nltk.sent_tokenize(line))

    words_list = []
    for sent in sentences:
        words_list.append(nltk.word_tokenize(sent))

    if ne_tag:
        ners_list = []
        for words in words_list:
            ners_list.append(ner_tagger.tag(words))

    tags_list = []
    dep_trees_list = dep_parser.raw_parse_sents(sentences)
    for dep_trees in dep_trees_list:
        for dep_tree in dep_trees:
            tags_list.append([morpheme.split('\t') for morpheme in dep_tree.to_conll(4).split('\n')])

    fixed_tags_list = []
    if not ne_tag:
        for tags, words in zip(tags_list, words_list):
            fixed_tags = []
            tag_idx = 0
            for w in words:
                result_word = None
                if tag_idx < len(tags):
                    result_word = tags[tag_idx]
                    if len(result_word) < 2:
                        tag_idx += 1

                if result_word is not None and w == result_word[0]:
                    fixed_tags.append(result_word)
                    tag_idx += 1
                else:
                    fixed_tags.append([w, None, None, None])
            fixed_tags_list.append(fixed_tags)

    else:
        for tags, words, ners in zip(tags_list, words_list, ners_list):
            fixed_tags = []
            tag_idx = 0
            for w in words:
                result_word = None
                ner_tag = None
                if tag_idx < len(tags):
                    result_word = copy(tags[tag_idx])
                    ner_tag = ners[tag_idx]
                    if len(result_word) < 2:
                        tag_idx += 1

                if result_word is not None and w == result_word[0]:
                    result_word.append(None if ner_tag[1] is 'O' else ner_tag[1])
                    fixed_tags.append(result_word)
                    tag_idx += 1
                else:
                    fixed_tags.append([w, None, None, None, None])
            fixed_tags_list.append(fixed_tags)

    return sentences, fixed_tags_list


def tags_to_pickle(tags, file_path):
    try:
        pickle_dir = os.path.join(os.path.abspath(os.path.dirname(file_path)))
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, 0o777)
        pickle_file = open(file_path, mode='wb+')
        pickle.dump(obj=tags, file=pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_file.close()
    except Exception as e:
        print(e)
        return False
    return True


def pickle_to_tags(file_path):
    tags = None
    try:
        pickle_file = open(file_path, mode='rb+')
        tags = pickle.load(pickle_file, encoding="utf-8")
        pickle_file.close()
    except Exception as e:
        print(e)
    return tags

