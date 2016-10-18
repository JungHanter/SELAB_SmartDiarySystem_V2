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
from nltk.corpus import wordnet_ic


class TextAnalyzer:
    def __init__(self):
        self.meaningless = stopwords.words('english')
        self.meaningless.extend(string.punctuation)
        self.emotionless = self.meaningless

        self.emotions = {}
        self.load_emotions()

        self.tokenizer = RegexpTokenizer(r'\w+')
        self.ner_tagger = StanfordNERTagger(PATH_ST_NER_MODEL, PATH_ST_NER)
        self.dep_parser = StanfordDependencyParser(PATH_ST_PARSER, PATH_ST_PARSER_MODEL)
        self.parser = StanfordParser(PATH_ST_PARSER, PATH_ST_PARSER_MODEL)
        self.lemmatizer = WordNetLemmatizer()

        self.text = None
        from nltk.corpus import propbank

    def load_emotions(self, emotion_set=None):
        if emotion_set is None:
            emotion_set = {
                'joy':
                    {'n': 'joy.n.02', 'v': 'rejoice.v.01', 'a': 'happy.a.01', 's': 'glad.s.02'},
                'anger':
                    {'n': 'anger.n.02', 'v': 'anger.v.02', 'a': 'angry.a.02', 's': 'angry.s.02'},
                # 'sad':
                #     {'n': "sadness.n.01", 'v': "suffer.v.06", 'a': "sad.a.01", 's': "deplorable.s.01"},
                'fear':
                    {'n': 'concern.n.02', 'v': 'worry.v.01', 'a': 'restless.a.03', 's': 'apprehensive.s.02'},
                'love':
                    {'n': 'love.n.04', 'v': 'love.v.01', 'a': 'warm.a.02', 's': 'adorable.s.01'},
                'hate':
                    {'n': 'hate.n.01', 'v': 'hate.v.01', 'a': 'disliked.a.01', 's': 'offensive.s.01'}
            }
        for _category in emotion_set.keys():
            emotion = {}
            for _type in emotion_set[_category].keys():
                emotion[_type] = wordnet.synset(emotion_set[_category][_type])
            self.emotions[_category] = emotion

    def analyze_sentence(self, sent):
        """
        Return a result of an analysis
        :param sent: sentence to analyze
        """
        sent = sent.lower()

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

        # pick delegate
        my_act = c_dep2
        my_emotions = {}
        for category in self.emotions:
            my_emotions[category] = 0
        for act in my_act:
            emotions = self._analyze_emotion2(self._get_dep_pos(act))
            for category in self.emotions:
                my_emotions[category] += emotions[category]
        for k in my_emotions.keys():
            print(k, "\t: ", my_emotions[k])
        print()
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
        for sent in sentences:
            result_set.append(self.analyze_sentence(sent))
        return result_set

    def analyze_emotion(self, sent):
        dep = next(self.dep_parser.raw_parse(sent))
        pos = self._get_dep_pos(self._get_conll4(dep))
        return self._analyze_emotion2(pos)

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

    def _analyze_emotion2(self, pos,):
        emotions = {}
        for category in self.emotions:
            emotions[category] = float(0)

        for category in emotions.keys():
            count, total = 0, 0
            for (token, tag) in pos:
                if tag[0] in ['P', 'R', 'D', 'W', 'I', 'C']:
                    continue
                if token[:-1] == 'CLAUSE':
                    continue
                sent_words = [sent_w for sent_w in list(sentiwordnet.senti_synsets(token, pos=pos_to_wn_tag(tag[0]))) if sent_w.obj_score() < 0.75]
                print(sent_words)
                for sent_word in sent_words:
                    word = sent_word.synset
                    if word.name().split('.', 1)[0] in self.emotionless:
                        continue
                    if word.pos() == 'r':
                        return float(0)
                    try:
                        similarity = word.lin_similarity(self.emotions[category][word.pos()], wordnet_ic.ic('ic-bnc.dat'))
                    except:
                        print(self.emotions[category][word.pos()], word)
                        similarity = float(0)
                    total += similarity
                    count += 1

            if count != 0:
                emotions[category] += total/count
            print(category, emotions[category], total, count)
        return emotions

    def extract_sentiment(self, sent):
        emotions = {}
        return emotions

    def _calc_emotion(self, res):
        max_value = max(res.values())
        min_value = 1
        for k in res.keys():
            if res[k] == 0:
                break
            if res[k] < min_value:
                min_value = res[k]
        if min_value < max_value:
            for key in res.keys():
                res[key] -= min_value
        return res

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
    # w1 = wordnet.synset(wordnet.synsets('happy')[2].name())
    # w2 = wordnet.synset(wordnet.synsets('pleasure')[2].name())
    # print(w1)
    # print(w2)
    # print(w1.path_similarity(w2))
    # info_contents = ['ic-bnc-add1.dat', 'ic-bnc-resnik-add1.dat',
    #                  'ic-bnc-resnik.dat', 'ic-bnc.dat',
    #
    #                  'ic-brown-add1.dat', 'ic-brown-resnik-add1.dat',
    #                  'ic-brown-resnik.dat', 'ic-brown.dat',
    #
    #                  'ic-semcor-add1.dat', 'ic-semcor.dat',
    #
    #                  'ic-semcorraw-add1.dat', 'ic-semcorraw-resnik-add1.dat',
    #                  'ic-semcorraw-resnik.dat', 'ic-semcorraw.dat',
    #
    #                  'ic-shaks-add1.dat', 'ic-shaks-resnik.dat',
    #                  'ic-shaks-resnink-add1.dat', 'ic-shaks.dat',
    #
    #                  'ic-treebank-add1.dat', 'ic-treebank-resnik-add1.dat',
    #                  'ic-treebank-resnik.dat', 'ic-treebank.dat']
    # from nltk.corpus import wordnet_ic
    # # flag = True
    # # idx = 0
    # # print(w1.res_similarity(w2, ic=wordnet_ic.ic(info_contents[idx])))
    # # while(flag):
    # #     try:
    # #         print(w1.res_similarity(w2, ic=wordnet_ic.ic(info_contents[idx])))
    # #         break
    # #     except :
    # #         print()
    # #         idx+=1
    # w1 = wordnet.synsets('happy')
    # w2 = wordnet.synsets('pleasure')
    # _sim = float(0)
    # for _w1 in w1:
    #     for _w2 in w2:
    #         try:
    #             print(_w1.lemma_names(), '\t', _w2.lemma_names())
    #             _sim = wordnet.synset(_w1.name()).jcn_similarity(wordnet.synset(_w2.name()), wordnet_ic.ic('ic-bnc.dat'))
    #             print(_w1, _w2)
    #             break
    #         except:
    #             continue
    # print(_sim)

    _emotions = {}
    from diary_nlp.nlp_en import get_synonyms

    sy = get_synonyms('loving')
    print(sy)
    for s in sy:
        print("\n", s)
        for syn in wordnet.synsets(s):
            print(syn, syn.definition())
            print(syn.lemmas())

 #    textanalyzer = TextAnalyzer()
 #    # textanalyzer.analyze_text("""I am pretty sure this is what my history teacher means when she talked about cruel and unusual punishment. """)
 #    textanalyzer.analyze_text("""
 # The WORST thing ever in the history of the world has happened!!\n\nMy dad took away my phone!!!! Can you BELIEVE that???\n\nI am pretty sure this is what my history teacher means when she talks about "cruel and unusual punishment." Because taking someone\'s phone away JUST because they were texting their friends is definitely up there with bread and water for life.\n\nI get why there\'s no texting at school. It\'s distracting or whatever. But EXCUSE me!!! I was in my own home!! I NEED to be distracted from the KA-RAY-ZEE all around me!!!\n\nUGH!!!\n\nI was minding my own business, texting Chloe and Zoey about how we should handle Twin Day, since there\'s three of us. And I GUESS Brianna had been asking me to play her Princess Sugar Plum video game for a while, but I was choosing not to hear her. (BTW, my parents do this selective hearing thing ALL THE TIME. But when I do it, I\'m "rude" and "unkind" and "obsessed with my phone.")\n\nSo, my dad LOST HIS MIND!!! He waved his arms around in the air and jumped up and down like an angry gorilla I saw in a documentary once. And he shouted, "No phones! No video games! No screens!"\n\nBut, I was definitely going to need my phone to call for the loony bin to take him away!! I was pretty sure he was going to start picking bugs out of my hair next. If I had bugs in my hair. Which I DON\'T!!!\n\nHe told us spring had arrived and kids need fresh air and we better play together peacefully outside, or we\'d never see a screen again. (My mom is away on a getaway trip with her college sorority sisters. I don\'t think my dad is a huge fan of being a single parent.)\n\nSo THEN I was shoved out the door with Brianna, who was wailing about her precious Princess Sugar Plum game. And let me tell you, spring is NOT the same as summer. "Duh, Nikki," you say. But it was FREEZING out there! I\'m just saying!!\n\nAlso, I\'m pretty sure it ISN\'T spring yet. Like, technically. I think the spring equinox is later in the month, but I couldn\'t look it up because I DIDN\'T HAVE MY PHONE!!!\n\nI stomped off the porch, straight into a puddle. Of COURSE I wasn\'t wearing boots or anything, since I had no plans to hang out in nature today, so my sneakers filled right up with water. My dad might have let me change my shoes, but Brianna was blocking the door and I had to get away from her hissy fit. I was kind of having a hissy fit too, but I also had the DECENCY to be quiet about it!\n\nSoggy shoes and all, I stomped further into the yard. Then Brianna screamed, "Nikki, FREEZE!!!" and there was no way I could ignore it.\n\nI froze. It sounded like Brianna knew something I didn\'t, like I was about to step on a wasp nest. Or a sleeping MacKenzie.\n\nI looked around and didn\'t see anything. "What is it, Brianna?"\n\nShe\'d completely forgotten about her video game. She ran over to me and got on her knees in the muddy grass.\n\n"What are you doing??"\n\n"Nikki, look."\n\nSo I squatted down and followed her crazy eyes to a flower. A crocus, I think.\n\n"The first flower of spring," she whispered, reaching out her hand.\n\n"Careful," I said. I mean, it was a really sweet nature moment, but I also know that Brianna is the kind of kid who gives a goldfish a bubble bath, so I was bracing for her to squash it without meaning to.\n\nBut she just barely brushed it with the tip of her finger and stared at it in wonder.\n\nI kind of wished I had my phone to take a picture of it and send it to my friends. I also wanted to look up the kind of flower, to be sure. Also to check when the spring equinox is, so I could officially tell my dad he\'s wrong about spring.\n\nBut I didn\'t have my phone. So I sat there with Brianna and looked at the first flower of spring. It was nice.\n\nFor about a minute.\n\nThen we concocted a plan to sneak back inside the house without Dad noticing. But hey! We were working together! And we didn\'t even use any screens!!
 #
 # """)
 #    textanalyzer.analyze_sentence("justification is often not necessary for knowledge outside science")