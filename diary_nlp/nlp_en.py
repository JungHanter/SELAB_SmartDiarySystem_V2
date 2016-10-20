from collections import defaultdict, Counter
from pandas import DataFrame
from smart_diary_system import database
from langdetect import detect


import string, nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.tag import pos_tag, map_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords, verbnet, sentiwordnet, wordnet_ic
from diary_nlp.nlp_en_tmp.settings import PATH_ST_NER, PATH_ST_NER_MODEL, PATH_ST_PARSER, PATH_ST_PARSER_MODEL
from nltk.tag.stanford import StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser, StanfordParser
from nltk.tree import ParentedTree, Tree
from nltk.chunk import regexp
from nltk import chunk


class SimilarityAnalyzer:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lemmatizer = WordNetLemmatizer()
        self.meaningless = stopwords.words('english')

        self.current_sent = ""
        self.current_lemma = []

        self.diary_db = database.AudioDiaryManager()
        self.c_text_db = database.TextDiaryManager()

    def find_most_similar_docs(self, query_sentence, user_id='lhs', limit=10):
        """Find N most similar documents
        N is min(<# of sentences>, limit).

        :param query_sentence:
        :param user_id:
        :param limit: an integer
        :return: a list of indexes in documents
        """

        query_sentence = replace_negations(query_sentence)
        if query_sentence != self.current_sent:
            tagged = [(word, tag) for word, tag in pos_tag(self.tokenizer.tokenize(query_sentence))]


            lemmas = []
            for token, tag in [(token, tag) for token, tag in tagged if token not in self.meaningless]:
                lemmas.extend(get_synonyms(token.lower(), pos=pos_to_wn_tag(tag[0])))
            print(lemmas)
            self.current_sent = query_sentence
            self.current_lemma = lemmas

        diary_list = self.c_text_db.get_converted_text_list({'user_id': 'lhs'})
        print("diaries loaded")
        similarity_score = []
        c_text_ids = []
        ca = ContextAnalyzer()
        process = []  # http://miscel.tistory.com/38
        for diary in diary_list:
            document = diary['text']
            if len(document) == 0:
                continue
            if detect(query_sentence) != detect(document):
                continue
            document = ca.summarize(document)
            score = self._calc_similarity(document)
            print(score)
            similarity_score.append(score)
            c_text_ids.append(diary['c_text_id'])

        list_t = DataFrame({'similarity': similarity_score, 'c_text_id': c_text_ids})
        return list_t.sort_values(by='similarity', ascending=False)

    def _calc_similarity(self, d):
        """ Calculate similarity score among the two input sentences

        :param d: document == list of compared sentence
        :return: a similarity score
        """

        syn_verbs, syn_nouns, basic_lemmas = [], [], []
        basic_lemmas = set(self.current_lemma)
        basic_len = len(basic_lemmas)
        sentences = sent_tokenizer(d)
        full_ratio = 0
        for sent in sentences:
            # compared_lemmas = set([self.lemmatizer.lemmatize(word=token.lower(), pos=pos_to_wn_tag(tag[0]))
            #                    for token, tag in tagged1])
            # common_words = (list(compared_lemmas & basic_lemmas))
            # ratio = float(len(common_words) / len(basic_lemmas))
            # # print(ratio, " \t", common_words)
            # full_ratio += ratio

            tagged1 = [(word, tag) for word, tag in pos_tag(self.tokenizer.tokenize(sent)) if word not in self.meaningless]
            verbs, advs, adjs, nouns = set(), set(), set(), set()
            compared = {'v': verbs, 'r': advs, 'a': adjs, 'n': nouns}
            for word, tag in tagged1:
                compared[pos_to_wn_tag(tag[0])].add(self.lemmatizer.lemmatize(word.lower(), pos_to_wn_tag(tag[0])))
            score_v = len(verbs & basic_lemmas) / basic_len
            # score_adv = len(advs & basic_lemmas) / basic_len
            # score_adj = len(adjs & basic_lemmas) / basic_len
            score_nouns = len(nouns & basic_lemmas) / basic_len
            full_ratio += score_v + score_nouns * 2
        if len(sentences) > 0:
            return full_ratio / len(sentences)
        else:
            return 0.0


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


class ContextAnalyzer:
    def __init__(self):
        import nltk, string
        self.meaningless = nltk.corpus.stopwords.words('english')
        self.meaningless.extend(string.punctuation)
        self.tokenizer = RegexpTokenizer(r'\w+')

        from diary_nlp.nlp_en_tmp.settings import PATH_ST_NER, PATH_ST_NER_MODEL, PATH_ST_PARSER, PATH_ST_PARSER_MODEL
        from nltk.tag.stanford import StanfordNERTagger
        from nltk.parse.stanford import StanfordDependencyParser, StanfordParser
        self.ner_tagger = StanfordNERTagger(PATH_ST_NER_MODEL, PATH_ST_NER)
        self.dep_parser = StanfordDependencyParser(PATH_ST_PARSER, PATH_ST_PARSER_MODEL)
        self.parser = StanfordParser(PATH_ST_PARSER, PATH_ST_PARSER_MODEL)
        self.ps = []
        # if ner_tagger error occur
        # export STANFORDTOOLSDIR= jars
        # export CLASSPATH=$STANFORDTOOLSDIR/stanford-ner-2015-12-09/stanford-ner.jar
        # export STANFORD_MODELS=$STANFORDTOOLSDIR/stanford-ner-2015-12-09/classifiers
        pass

    def analyze(self, document):
        self._analyze_tokens()

        self._analyze_sentence_feature()

        self._analyze_document_feature()
        self._analyze_named_entity()
        self._indexing()
        pass

    def _indexing(self):
        pass

    def _analyze_semantic(self, document):
        pass

    def _analyze_named_entity(self):
        pass

    def _analyze_sentence_feature(self, sentence):
        dep_parsed = {}
        parsed = []
        subjects, verbs, dobject, nmod, root = [], [], [], [], None
        for i, dep in enumerate(next(self.dep_parser.raw_parse(sentence.lower())).to_conll(4).split('\n')):
            if dep is not '':
                p = dep.split('\t')
                dep_parsed[i+1] = [p[0], map_tag('en-brown', 'universal', p[1]), p[2], p[3]]

        for i in range(1, len(dep_parsed)):
            p = dep_parsed[i]
            parsed.extend([(p[0], p[1])])
            if p[3] == 'nsubj':
                subjects.append([p[0], map_tag('en-brown', 'universal', p[1]), p[2], p[3]])
            elif p[3] == 'root':
                root = [p[0], map_tag('en-brown', 'universal', p[1]), p[2], p[3]]
            elif p[3] == 'dobj':
                dobject.append([p[0], map_tag('en-brown', 'universal', p[1]), p[2], p[3]])
            elif p[3] == 'nmod':
                nmod.append([p[0], map_tag('en-brown', 'universal', p[1]), p[2], p[3]])
            if p[1][0] == 'V':
                verbs.append([p[0], map_tag('en-brown', 'universal', p[1]), p[2], p[3]])
        # debug
        for i in range(1, len(dep_parsed)+1):
            if dep_parsed[i] in subjects:
                print(dep_parsed[i], '\tsubject', end="")
            elif dep_parsed[i] in verbs:
                print(dep_parsed[i], '\tverb', end="")
            elif dep_parsed[i] in dobject:
                print(dep_parsed[i], '\tdobject', end="")
            elif dep_parsed[i] in nmod:
                print(dep_parsed[i], '\tobject(nmod)', end="")
            else:
                print(dep_parsed[i], end="")
            if dep_parsed[i] is root:
                print('\troot', end="")
            print()

        parsed2 = []
        pos_tagged = [p for w, p in pos_tag(sentence.split())]
        for i, (w, p) in enumerate(parsed):
            univ_tag = map_tag('en-brown', 'universal', p)
            parsed2.extend([(w, univ_tag, pos_tagged[i])])
        print(parsed2)


    def _analyze_document_feature(self):
        pass

    def _analyze_tokens(self):
        pass

    def separate_sent(self, parsed):
        import nltk
        for p in [_p for _p in parsed if isinstance(_p, nltk.Tree)]:
            self.separate_sent(p)
            if p.label() == 'SBAR':
                p.set_label('SBAR'+str(len(self.ps)))
                self.ps.append((p.subtrees(lambda p: p.label()[:4] != 'SBAR')))

    def semantic(self, document):
        sents = sent_tokenize(document)
        nouns = []
        rels = []
        for sent in sents:
            tagged = pos_tag(self.tokenizer.tokenize(sent))
            parsed = self.parser.raw_parse(sent)
            self.separate_sent(parsed)
            for p in self.ps:
                print(next(p))
            dep_parsed = {}
            start = 0
            subj = []
            # for i, dep in enumerate(next(self.dep_parser.raw_parse(sent.lower())).to_conll(4).split('\n')):
            #     if dep is not '':
            #         p = dep.split('\t')
            #         dep_parsed[i+1] = [p[0], map_tag('en-brown', 'universal', p[1]), p[2], p[3]]
            #         if p[3] == 'root':
            #             start = i+1
            #         if p[3] == 'nsubj':
            #             subj.append(i+1)
            # for i in range(1, len(dep_parsed)+1):
            #     print(dep_parsed[i])

    def ger_basic_rel(self, sents):
        for sent in sents:
            # tokenize
            tagged = pos_tag(self.tokenizer.tokenize(sent))
            print(tagged)
            # identify verb or noun or sth
            universal_tagged = [(idx, word, map_tag('en-brown', 'universal', tag), tag) for idx, (word, tag) in
                                enumerate(tagged)]
            # former_tag = ' '
            # i=0
            # words, tags = [][]
            # for word, tag in tagged:
            #     uni_tag = map_tag('en-brown', 'universal', tag)
            #     if former_tag == uni_tag:


            nouns = [(idx, word) for idx, word, uni_tag, tag in universal_tagged if (uni_tag == 'NOUN') | (tag == 'PRP')]

            # gather nouns
            for idx, noun in enumerate(nouns):
                if idx + 1 < len(nouns):
                    rel = [(idx, word, uni_tag, tag)
                           for (idx, word, uni_tag, tag) in universal_tagged[nouns[idx][0] + 1:nouns[idx + 1][0]]]
                    flag = False
                    for i, word, uni_tag, tag in rel:
                        if uni_tag == 'VERB':
                            flag = True
                    if flag:
                        print(' '.join([word for i, word, uni_tag, tag, in rel]), " ( ", noun[1], ", ",nouns[idx+1][1], " ) ")

                elif nouns[idx][0]+1 < len(tagged):
                    rel = [word for word, tag in tagged[nouns[idx][0] + 1:]]
                    print(' '.join(rel), " ( ", noun[1], " ) ")

    def summarize(self, document):
        """ Summarize document

        :param document: daily diary
        :return: summarized diary
        """
        sentences = sent_tokenize(document)
        if len(sentences) > 1:
            frequency = self._calc_token_frequency(sentences)
            sent_list = self._rank_sent_by_words(sentences, frequency, int(len(sentences) / 3))
            # print("info \noriginal sents : ", len(sentences))
            # print("summary sents : ", len(sent_list))
            # print(sent_list)
            # print("frequency :\n", Counter(frequency).most_common(int(len(sentences)/3)))
            return ' '.join(sent_list)
        else:
            return document

    def _calc_token_frequency(self, sentences):
        """ calculate frequency

        :param sentences: sentences of dictionary
        :return: frequent words in dictionary format
        """

        tokenizer = RegexpTokenizer(r'\w+')
        frequency = defaultdict(int)
        for sentence in sentences:
            for token in tokenizer.tokenize(sentence):
                if token not in self.meaningless:
                    frequency[token] += 1
        return frequency

    def _rank_sent_by_words(self, sentences, frequency, limit):

        """ filter out less popular sentences

        :param sentences: sentences of document
        :param frequency: {(word: frequency)...} dictionary
        :param limit: limit
        :return:
        """
        rank = defaultdict(int)
        maximum = float(max(frequency.values()))
        for word in frequency.keys():
            frequency[word] /= maximum

        for idx, sentence in enumerate(sentences):
            for token in word_tokenize(sentence):
                rank[idx] += frequency[token]
        common = Counter(rank).most_common(min(len(sentences), limit))
        return [sentences[freq_idx] for idx, (freq_idx, freq) in enumerate(common)]

    def summarize_sent(self):
        pass

    def _analyze_date(self, sentence):
        pass

    def _analyze_place(self, sentence):
        pass

    def _analyze_weather(self, sentence):
        pass


class TextAnalyzer:
    def __init__(self):
        self.meaningless = stopwords.words('english')
        self.meaningless.extend(string.punctuation)
        self.emotionless = self.meaningless
        self.emotionless.extend([',','"',''',''',"""'s""", '.', 'at'])

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
                # 'love':
                #     {'n': 'love.n.04', 'v': 'love.v.03', 'a': 'loving.a.01', 's': 'adorable.s.01'},
                'hate':
                    {'n': 'hate.n.01', 'v': 'hate.v.01', 'a': 'disliked.a.01', 's': 'offensive.s.01'}
            }
        for _category in emotion_set.keys():
            emotion = {}
            for _type in emotion_set[_category].keys():
                emotion[_type] = wordnet.synset(emotion_set[_category][_type])
            self.emotions[_category] = emotion

    def analyze_text(self, text):
        result = []
        words = []
        for sent in sent_tokenize(text):
            features = {'sentence': sent}
            features.update(self.analyze_sent(sent))
            result.append(features)
        result.append(self.get_extra_feat(text))
        return result

    def analyze_sent(self, sent):
        # pre process
        res = {'summary': [], 'ne': {}, 'act': [], 'emotion': {}}
        dep_tagged = self._get_conll4(self.dep_parser.raw_parse(sent))
        pos_tagged = self._get_pos(dep_tagged, res['summary'])

        # named entity
        res['ne'] = self._analyze_ne(pos_tagged, res['summary'])

        # act
        res['act'] = self._analyze_act(pos_tagged, res['summary'])

        return res

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
                tags[i + 1 + key_offset] = dep
        return tags

    def _get_pos(self, conll4, summary):
        pos = []
        for key, value in conll4.items():
            summary.append(value[0:2])
            _set = tuple(value[0:2])
            pos.append(_set)
        return pos

    def _analyze_ne(self, pos_tagged, summary):
        # pre process
        tmp = []
        for idx, (token, pos) in enumerate(pos_tagged):
            if (pos[0] == 'N') | (pos[0] == 'J'):
                tmp.append(token[0].upper()+token[1:])
            else:
                tmp.append(token)
        tagged = self.ner_tagger.tag(word_tokenize(' '.join(tmp)))

        # analyze ne
        tmp, tag = [], ""
        nes = {'LOCATION': [], 'ORGANIZATION': [], 'PERSON': [], 'MONEY': [], 'PERCENT': [],
               'DATE': [], 'TIME': [], 'MISC': []}
        for i, (token, ne) in enumerate(tagged):
            summary[i].extend([ne])
            if ne == tag:
                tmp.append(token)
            elif (ne == 'LOCATION') | (ne == 'ORGANIZATION') | (ne == 'PERSON') | (ne == 'MONEY') \
                    | (ne == 'PERCENT') | (ne == 'DATE') | (ne == 'TIME') | (ne == 'MISC'):
                if len(tmp) > 0:
                    nes[tag].append(' '.join(tmp))
                    tmp = []
                tag = ne
                tmp.append(token)
        if len(tmp) > 0:
            nes[tag].append(' '.join(tmp))

        # TODO: additional chunk
        return nes

    def _analyze_act(self, pos_tagged, summary):
        clauses = self._get_clauses(pos_tagged)
        clause_deps, c_pos = [], 0
        for clause in reversed(clauses):
            clause_deps.append(self._get_conll4(self.dep_parser.tagged_parse(clause.pos()), c_pos))
            c_pos += len(clause.pos()) - 1
        acts = []
        for c_dep in reversed(clause_deps):
            acts.append(self._analyze_role(c_dep, summary))
        return acts

    def _get_clauses(self, pos):
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

    def _analyze_role(self, dep, summary):
        rel = {'Agent': [], 'V': [], 'Patient': [], 'Instrument': [], 'neg': [] }
        for key in reversed(sorted(dep.keys())):
            d = dep[key]
            if d[3] == 'nsubj':
                rel['Agent'].append(d[0])
                summary[key - 1].extend(['*'])
                # summary[key-1].extend(['Agent'])
            # object
            elif d[3] == 'nmod':
                rel['Instrument'].append(d[0])
                summary[key - 1].extend(['*'])
                # summary[key-1].extend(['Instrument'])
            elif d[3] == 'avmod':
                rel['Instrument'].append(d[0])
                summary[key - 1].extend(['*'])
                # summary[key-1].extend(['Instrument'])
            elif d[3] == 'dobj':
                rel['Patient'].append(d[0])
                summary[key - 1].extend(['*'])
                # summary[key-1].extend(['Patient'])
            # negation
            elif d[3] == 'neg':
                rel['neg'].append(d[0])
                summary[key - 1].extend(['*'])
                summary[key-1].extend(['neg'])
            # verbs
            elif d[1][0] == 'V':
                rel['V'].append(d[0])
                lemma = self.lemmatizer.lemmatize(d[0], pos='v')
                # v = verbnet.classids(lemma=lemma)
                if lemma in self.meaningless:
                    summary[key-1].extend('*')
                # elif len(v) > 0:
                #     summary[key-1].extend([','.join(v)])
                else:
                    summary[key-1].extend([lemma])
            else:
                summary[key-1].extend('*')
        return rel

    def get_extra_feat(self, text):
        sents = sent_tokenize(text)
        words = []
        for sent in sents:
            words.extend(word_tokenize(sent))
        doc = nltk.Text([token for token in words if words not in self.emotionless])
        from nltk import FreqDist
        freq_words = FreqDist(doc).most_common(50)
        word_sets = []
        for (token, freq) in freq_words:
            word_sets.extend([token for sent_w in list(sentiwordnet.senti_synsets(token))
                              if sent_w.obj_score() < 0.75])
        emotions = {}
        for category in self.emotions:
            emotions[category] = float(0)

        sentiment = float(0)
        emotions = {'joy': 0.0, 'hate': 0.0, 'anger': 0.0, 'fear': 0.0}

        from multiprocessing import Queue, Process
        workers = len(word_sets)
        work_queue = Queue()
        done_queue = Queue()
        process = []

        for word in word_sets:
            work_queue.put(word)
        for w in range(workers):
            p = Process(target=_calc_emotion, args=(work_queue, done_queue))
            p.start()
            process.append(p)
            work_queue.put('STOP')

        for p in process:
            p.join()

        done_queue.put('STOP')

        for result in iter(done_queue.get, 'STOP'):
            tmp_dict = eval(result)
            emotions['joy'] += tmp_dict['joy']
            emotions['anger'] += tmp_dict['anger']
            emotions['hate'] += tmp_dict['hate']
            emotions['fear'] += tmp_dict['fear']
            sentiment += tmp_dict['sentiment']

        emo = 0
        key = ""
        full = 0
        for k in emotions.keys():
            full += emotions[k]
            if emotions[k] > emo:
                emo = emotions[k]
                key = k
        if full == 0:
            return {'emotion': 'NO_TEXT', 'accuracy': '0', 'sentiment_value': 'NO_TEXT'}
        return {'emotion': key, 'accuracy': (full-emo) / full, 'sentiment_value': sentiment}


def _calc_emotion(work_queue, done_queue):
    emotion_target = {'joy': {'a': wordnet.synset('happy.a.01'), 'v': wordnet.synset('rejoice.v.01'), 's': wordnet.synset('glad.s.02'), 'n': wordnet.synset('joy.n.02')}, 'hate': {'a': wordnet.synset('disliked.a.01'), 'v': wordnet.synset('hate.v.01'), 's': wordnet.synset('offensive.s.01'), 'n': wordnet.synset('hate.n.01')}, 'anger': {'a': wordnet.synset('angry.a.02'), 'v': wordnet.synset('anger.v.02'), 's': wordnet.synset('angry.s.02'), 'n': wordnet.synset('anger.n.02')}, 'fear': {'a': wordnet.synset('restless.a.03'), 'v': wordnet.synset('worry.v.01'), 's': wordnet.synset('apprehensive.s.02'), 'n': wordnet.synset('concern.n.02')}}

    res = {'fear': 0.0, 'joy': 0.0, 'anger': 0.0, 'hate': 0.0, 'sentiment': 0.0}
    for token in iter(work_queue.get, 'STOP'):
        for word in list(sentiwordnet.senti_synsets(token)):
            for category in emotion_target.keys():
                try:
                    p = word.synset.pos()
                    similarity = word.synset.jcn_similarity(emotion_target[category][word.synset.pos()],
                                                            wordnet_ic.ic('ic-bnc.dat'))
                    if (category == 'hate') | (category == 'fear'):
                        res[category] += similarity * (1 - word.obj_score()) * word.neg_score()
                        res['sentiment'] -= word.neg_score()
                    else:
                        res[category] += similarity * (1 - word.obj_score()) * word.pos_score()
                        res['sentiment'] += word.pos_score()
                except Exception as e:
                    print(e)
                    res[category] += 0
    try:
        done_queue.put(str(res))
    except Exception as e:
        done_queue.put(str({'fail': e}))
    return True

def make_ptree(s):
    all_ptrees = []
    ptree = ParentedTree.convert(Tree.fromstring(s))
    all_ptrees.extend(t for t in ptree.subtrees()
                      if isinstance(t, Tree))
    return ptree


def replace_negations(sentence):
    """Replace negation with antonym

    :param sent : sentence
    :return sentence of which negations are replaced with antonym
    """
    i = 0
    sent = word_tokenize(sentence)
    tagged = pos_tag(sent)
    words = []
    while i < len(sent):
        word = sent[i]
        if word == 'not' and i + 1 < len(sent):
            ants = get_antonyms(sent[i + 1])
            if ants:
                words.append(ants[0])
                i += 2
                continue
        words.append(word)
        i += 1
    return ' '.join(words)


def get_synonyms(word, pos=None, limit=5):
    """ Find synonyms of word

    :param word: word
    :param pos: word net pos
    :param limit: an integer
    :return synonym list"""
    syns = defaultdict(int)
    for syn in wordnet.synsets(word, pos):
        for lemma in syn.lemmas():
            syns[lemma.name()] += 1
    most_close_syns = Counter(syns).most_common(min(len(syns), limit))
    return [syn for (syn, count) in most_close_syns]


def get_antonyms(word, pos=None, limit=5):
    """ Find antonyms of word

    :param word: word
    :param pos : word net pos
    :param limit: an integer
    :return antonym list"""
    ants = defaultdict(int)
    for syn in wordnet.synsets(word, pos):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                ants[antonym.name()] += 1
    most_close_ants = Counter(ants).most_common(min(len(ants), limit))
    return [syn for (syn, count) in most_close_ants]


def sent_tokenizer(document):
    """Split document into sentences
    :param document: one diary .. not list!
    :return list of sentences"""
    return sent_tokenize(document)


def pos_tagger(sentence):
    return pos_tag(word_tokenize(sentence))


def demo():
    """I met matthew cornell at new york last weekend. I will meet him again at seoul this week."""
    ta = TextAnalyzer()
    ret = ta.analyze_text("""
    We've encountered a couple of issues. We are faced with some issues.
    """)

    for result in ret[:-1]:
        print('sentence : ', result['sentence'])
        print('entitys')
        print('\tLOCATION     : ', result['ne']['LOCATION'])
        print('\tORGANIZATION : ', result['ne']['ORGANIZATION'])
        print('\tPERSON       : ', result['ne']['PERSON'])
        print('\tTIME         : ', result['ne']['TIME'])
        print('\tDATE         : ', result['ne']['DATE'])
        print('\tMISC         : ', result['ne']['MISC'])
        print('actions')
        for i, action in enumerate(result['act']):
            print(i)
            print('\tAgent        : ', action['Agent'])
            print('\tVerb         : ', action['V'])
            print('\tPatient      : ', action['Patient'])
            print('\tInstrument   : ', action['Instrument'])
        print('summary')
        for word in result['summary']:
            print(word)
    print(ret[-1])

if __name__ == '__main__':
    demo()

