import string, nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.tag import pos_tag, map_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords, verbnet, sentiwordnet
from diary_nlp.nlp_en_tmp.settings import PATH_ST_NER, PATH_ST_NER_MODEL, PATH_ST_PARSER, PATH_ST_PARSER_MODEL
from nltk.tag.stanford import StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser, StanfordParser
from nltk.tree import ParentedTree, Tree
from nltk.chunk import regexp
from nltk import chunk
from collections import OrderedDict, defaultdict, Counter
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer

class TextAnalyzer:
    def __init__(self):
        self.meaningless = stopwords.words('english')
        self.meaningless.extend(string.punctuation)
        self.emotionless = self.meaningless

        self.wordList = defaultdict(list)
        self.emotionList = defaultdict(list)
        self._load_nrc_emotion_lexicon()

        self.text_emotion = Counter()

        self.tokenizer = RegexpTokenizer(r'\w+')
        self.ner_tagger = StanfordNERTagger(PATH_ST_NER_MODEL, PATH_ST_NER)
        self.dep_parser = StanfordDependencyParser(PATH_ST_PARSER, PATH_ST_PARSER_MODEL)
        self.parser = StanfordParser(PATH_ST_PARSER, PATH_ST_PARSER_MODEL)
        self.lemmatizer = WordNetLemmatizer()

        self.text = None
        from nltk.corpus import propbank

    def _load_nrc_emotion_lexicon(self):
        from diary_nlp.nlp_en_tmp.settings import PATH_NRC_DATA
        nrc_file = open(PATH_NRC_DATA, 'r')
        reader = csv.reader(nrc_file, delimiter='\t')
        headerRows = [i for i in range(0, 46)]
        for i in headerRows:
            next(reader)
        for word, emotion, persent in reader:
            if int(persent) == 1:
                self.wordList[word].append(emotion)
                self.emotionList[emotion].append(word)
        nrc_file.close()

    def analyze_sentence(self, sent):
        """
        Return a result of an analysis
        :param sent: sentence to analyze
        """
        # sent = sent.lower()

        result = {}
        # get full sentence's dependency
        dep = self._get_conll4(self.dep_parser.raw_parse(sent))
        ne = self._analyze_ne(dep)
        result['sentence_dep'] = dep

        pos = self._get_dep_pos(dep)
        result['sentence'] = [next(self.parser.tagged_parse(pos))]

        # split sentence into clauses
        clauses = self._analyze_clauses(pos)
        result['clause'] = clauses

        # check emotion table
        for clause in clauses:
            self._calc_emotion(clause.leaves())

        # # get clauses' dependency using original dependency
        # c_dep1 = []
        # c_start, cur = [], 1
        # for i in reversed(range(0, len(clauses))):
        #     _from = cur
        #     c_start.append(_from)
        #     _to = cur + len(clauses[i].leaves())
        #     _c_dep = {}
        #     for j in range(_from, _to):
        #         _c_dep[j] = dep[j]
        #     c_dep1.append(_c_dep)
        #     cur = _to - 1

        # get clauses' dependency using parsed dependency
        c_dep2 = []
        c_pos = 0
        for c in reversed(clauses):
            c_dep2.append(self._get_conll4(self.dep_parser.tagged_parse(c.pos()), c_pos))
            c_pos += len(c.pos()) - 1
        c_dep_mark = {}
        for c in reversed(c_dep2):
            for key in c.keys():
                c_dep_mark[key] = c[key][3]
        result['clause_dep'] = c_dep2

        # extract syntax relation from each sentence
        sr = []
        for c in reversed(c_dep2):
            sr.append(self._analyze_syntax(c))
        sr_mark = {}
        for c in reversed(sr):
            for k in sorted(c.keys()):
                sr_mark[k] = c[k][1]
        result['clause_role'] = sr

        # identify pronoun

        # extract key phrases
        key1 = self._find_keyphrase(pos_tag(word_tokenize(sent)))
        key2 = []
        for c in reversed(clauses):
            key2.extend(self._find_keyphrase(c.pos()))
        key = (set(key1) & set(key2))
        result['key'] = key

        # for print info
        clause_mark, word_i = {}, 0
        for i, clause in enumerate(reversed(clauses)):
            clause_mark[word_i] = '(S*'
            for w in clause.leaves():
                word_i += 1
                if "#^&$*#$#%@" in w:
                    break
                clause_mark[word_i] = '*'
            if i > 0:
                word_i -= 1

        self._print(dep, ne, clause_mark, c_dep_mark, sr_mark)
        print("key phrase\n", key)
        print("key phrase rel\n")

        for k in sorted(result.keys()):
            print(k, result[k])
        return result

    def analyze_text(self, text):
        """
        Return a result set of an analysis
        :param text: text to analyze
        """
        sentences = sent_tokenize(text)
        print(sentences)
        result_set = []
        self.text_emotion = None
        self.text_emotion = Counter()
        for sent in sentences:
            result_set.append(self.analyze_sentence(sent))
        print(self.text_emotion)
        return result_set

    def _calc_emotion(self, tokens):
        for token in tokens:
            self.text_emotion += Counter(self.wordList[token.lower()])

    def _get_conll4(self, dep_tree, key_offset=0):
        """

        :param dep_tree: stanford dependency
        :return:
        """
        # Analyze Dependencies
        tags = {}
        dependency = next(dep_tree)
        for i, d in enumerate(dependency.to_conll(4).split('\n')):
            if d is not '':
                dep_tmp = (d.split('\t'))
                dep = []
                if int(dep_tmp[2]) == 0:
                    dep = dep_tmp
                else:
                    dep = [dep_tmp[0], dep_tmp[1], (str(int(dep_tmp[2]) + key_offset)), dep_tmp[3]]
                tags[i+1 + key_offset] = dep

        return tags

    def _get_dep_pos(self, conll4):
        pos = []
        for key, value in conll4.items():
            _set = tuple(value[0:2])
            pos.append(_set)
        return pos

    def _analyze_ne(self, dep):
        """

        :param dep: stanford dependency in followed format - 4 column CoNLL
        :return:
        """
        tokens = [dep[i][0] for i in range(1, len(dep)+1)]
        ne = self.ner_tagger.tag(tokens)
        ne_tags = {}
        for i in range(0, len(dep)):
            ne_tags[i+1] = (ne[i][1])
        return ne_tags

    def _analyze_clauses(self, pos):
        """

        :param tags: stanford dependency in followed format - 4 column CoNLL
        :return:
        """
        parsed = next(self.parser.tagged_parse(pos))
        sentence, clauses, phrase = [], [], []
        clause_i = 0
        sentence = ParentedTree.fromstring(str(parsed))
        for tree_pos in [tp for tp in reversed(sentence.treepositions()) if isinstance(parsed[tp], nltk.tree.Tree)]:
            if sentence[tree_pos].label() == 'SBAR':
                sentence[tree_pos].set_label('ROOT')
                clauses.append(Tree.fromstring(str(sentence[tree_pos])))
                sentence.__delitem__(tree_pos)
                sentence[tree_pos[:-1]].insert(tree_pos[-1], make_ptree('(NN '+ 'CLAUSE' + str(clause_i) + ')'))
                clause_i += 1
        clauses.append(Tree.fromstring(str(sentence)))
        return clauses

    def _analyze_syntax(self, dep):
        # rel = {}
        # obj = {}
        role = {}
        for key in reversed(sorted(dep.keys())):
            d = dep[key]
            # dd = [d[0], d[1], d[2], d[3], key]
            # subject
            if d[3] == 'nsubj':
                # rel['subject'] = dd
                role[key] = (d[0], 'Agent')
            # object
            elif d[3] == 'nmod':
                # obj['nmod'] = dd
                role[key] = (d[0], 'Instrument')
            elif d[3] == 'avmod':
                # obj['advmod'] = dd
                role[key] = (d[0], 'Instrument')
            elif d[3] == 'dobj':
                # obj['dobj'] = dd

                role[key] = (d[0], 'Patient')
            # negation
            elif d[3] == 'neg':
                # rel['NEGATIONS'] = dd
                role[key] = (d[0], 'neg')
            # verbs
            elif d[1][0] == 'V':
                # rel['verb'] = dd
                lemma = self.lemmatizer.lemmatize(d[0], pos='v')
                v = verbnet.classids(lemma=lemma)
                if len(v) > 0:
                    role[key] = (d[0], 'V'+str(v))
                else:
                    role[key] = (d[0], 'V('+str(d[0]) +')')
            else:
                role[key] = (d[0], '*')
        # rel['obj'] = obj
        return role

    def _recognize_sr(self, tokens):
        """

        :param tokens:
        :return:
        """
        pass

    def _find_keyphrase(self, pos):
        nn_rule = r"""KEY:{(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}"""
        nn_parser = chunk.RegexpParser(nn_rule)
        chunked = nn_parser.parse(pos)
        result = []
        for subtree in chunked.subtrees(filter=lambda t: t.label() == 'KEY'):
            tmp = [token for token, tag in subtree.leaves()]
            result.append(' '.join(tmp))
        return result

    def _analyze_cor(self):
        pass

    def _print(self, dep, ne=None, ps=None, cd=None, sr=None):
        temp = []
        if sr is not None:
            temp.append(['key', 'word', 'pos', 'dep', 'rel', 'ne', 'ps', 'clause_dep', 'sr'])
        elif cd is not None:
            temp.append(['key', 'word', 'pos', 'dep', 'rel', 'ne', 'ps', 'clause_dep'])
        elif ps is not None:
            temp.append(['key', 'word', 'pos', 'dep', 'rel', 'ne', 'ps'])
        elif ps is not None:
            temp.append(['key', 'word', 'pos', 'dep', 'rel', 'ne'])
        else:
            temp.append(['key', 'word', 'pos', 'dep', 'rel'])
        for key, value in dep.items():
            tmp = [key]
            tmp.extend(value)
            if ps is not None:
                tmp.extend([ne[key]])
            if ps is not None:
                tmp.extend([ps[key]])
            if cd is not None:
                tmp.extend([cd[key]])
            if sr is not None:
                tmp.extend([sr[key]])
            temp.append(tmp)
        from tabulate import tabulate
        print()
        print(tabulate(temp, headers="firstrow"))


def pos_to_wn_tag(tag_start):
    # tag_start = tag[0]
    if tag_start == 'V':
        return 'v'  # wordnet.VERB
    elif tag_start == 'R':
        return 'r'  # wordnet.ADV
    elif tag_start == 'J':
        return 'a'  # wordnet.ADJ
    elif tag_start == 'N':
        return 'n'  # wordnet.NOUN
    else:
        return 'n'  # wordnet.NOUN


def make_ptree(s):
    all_ptrees = []
    ptree = ParentedTree.convert(Tree.fromstring(s))
    all_ptrees.extend(t for t in ptree.subtrees()
                      if isinstance(t, Tree))
    return ptree

if __name__ == '__main__':
    r1 = []
    r2 = []
    for i in range(0,10):
        r2.append(r1)

    textanalyzer = TextAnalyzer()
    # textanalyzer.analyze_text("""I am pretty sure this is what my history teacher means when she talked about cruel and unusual punishment. """)
    textanalyzer.analyze_text("""

        I met matthew cornell at the Independence Day.

     """)
    # textanalyzer.analyze_sentence("justification is often not necessary for knowledge outside science")