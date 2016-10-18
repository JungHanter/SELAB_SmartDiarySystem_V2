import string, nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.tag import pos_tag, map_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from diary_nlp.nlp_en_tmp.settings import PATH_ST_NER, PATH_ST_NER_MODEL, PATH_ST_PARSER, PATH_ST_PARSER_MODEL
from nltk.tag.stanford import StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser, StanfordParser
from nltk.tree import ParentedTree, Tree

class TextAnalyzer:
    def __init__(self):

        self.meaningless = stopwords.words('english')
        self.meaningless.extend(string.punctuation)
        self.tokenizer = RegexpTokenizer(r'\w+')

        self.ner_tagger = StanfordNERTagger(PATH_ST_NER_MODEL, PATH_ST_NER)
        self.dep_parser = StanfordDependencyParser(PATH_ST_PARSER, PATH_ST_PARSER_MODEL)
        self.parser = StanfordParser(PATH_ST_PARSER, PATH_ST_PARSER_MODEL)

        self.text = None
        from nltk.corpus import propbank

    def analyze_sentence(self, sent):
        """
        Return a result of an analysis
        :param sent: sentence to analyze
        """
        result = {}
        # get full sentence's dependency
        dep = self._analyze_tokens(sent)
        ne = self._analyze_ne(dep)

        # split sentence into clauses
        pos = []
        for key, value in dep.items():
            pos.append((value[0:2]))
        clauses = self._analyze_clauses(pos)

        # get clauses' dependency
        c_dep = []
        c_start, cur = [], 1
        for i in reversed(range(0, len(clauses))):
            _from = cur
            c_start.append(_from)
            _to = cur + len(clauses[i].leaves())
            _c_dep = {}
            for j in range(_from, _to):
                _c_dep[j] = dep[j]
            c_dep.append(_c_dep)
            cur = _to - 1


        # extract syntax relation from each sentence
        sr = []
        for c in reversed(c_dep):
            rel = self._analyze_syntax(c)
            print("rel :", rel)
            self._print(c)

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
        self._print(dep, ne, clause_mark)
        return result

    def analyze_text(self, text):
        """
        Return a result set of an analysis
        :param text: text to analyze
        """
        sentences = sent_tokenize(text)
        result_set = []
        for sent in sentences:
            result_set.append(self.analyze_sentence(sent))
        return result_set

    def _analyze_tokens(self, sent):
        """

        :param sent:
        :return:
        """
        # Analyze Dependencies
        tags = {}
        dependency = next(self.dep_parser.raw_parse(sent))
        for i, d in enumerate(dependency.to_conll(4).split('\n')):
            if d is not '':
                tags[i+1] = (d.split('\t'))

        return tags

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
                sentence[tree_pos[:-1]].insert(tree_pos[-1], make_ptree('(SBAR '+ 'CLAUSE' + str(clause_i) + ')'))
                clause_i += 1
        clauses.append(Tree.fromstring(str(sentence)))
        return clauses

    def _analyze_syntax(self, dep):
        rel = {}
        obj = {}
        for key in reversed(sorted(dep.keys())):
            d = dep[key]
            print(d)
            # subject
            if d[3] == 'nsubj':
                rel['subject'] = d
            # object
            elif d[3] == 'nmod':
                obj['nmod'] = d
            elif d[3] == 'avmod':
                obj['advmod'] = d
            elif d[3] == 'dobj':
                obj['dobj'] = d
            # negation
            elif d[3] == 'neg':
                rel['NEGATIONS'] = d
            # verbs
            elif d[1][0] == 'V':
                rel['verb'] = d
        rel['obj'] = obj
        return rel

    def _analyze_clause(self, tokens):
        """

        :param tokens:
        :return:
        """
        pass

    def _recognize_sr(self, tokens):
        """

        :param tokens:
        :return:
        """
        pass

    def _analyze_cor(self):
        pass

    def _print(self, dep, ne=None, ps=None):
        temp = []
        if ps is not None:
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
            temp.append(tmp)
        from tabulate import tabulate
        print()
        print(tabulate(temp, headers="firstrow"))

def make_ptree(s):
    all_ptrees = []
    ptree = ParentedTree.convert(Tree.fromstring(s))
    all_ptrees.extend(t for t in ptree.subtrees()
                      if isinstance(t, Tree))
    return ptree

if __name__ == '__main__':

    textanalyzer = TextAnalyzer()
    textanalyzer.analyze_sentence("""I am pretty sure this is what my history teacher means when she talked about cruel and unusual punishment. """)
    textanalyzer.analyze_sentence("justification is often not necessary for knowledge outside science")