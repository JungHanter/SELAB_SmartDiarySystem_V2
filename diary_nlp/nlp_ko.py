import math
from konlpy.tag import Twitter
from konlpy.tag import Kkma
from collections import Counter, namedtuple
import codecs
from pandas import DataFrame
import pickle
import jpype
import nltk
import os
import re
from smart_diary_system import database
from smart_diary_system import settings
import numpy as np

class SimilarityAnalyzer:
    def __init__(self, user_id='lhs'):
        self.tokenizer = Twitter()
        self.user_id = user_id
        with open(os.path.join(settings.BASE_DIR, 'diary_nlp', 'doc.model'), 'rb') as f1:  # 저장한 pickle 불러오기
            self.model = pickle.load(f1)
        with open(os.path.join(settings.BASE_DIR, 'diary_nlp', 'classifier'), 'rb') as f2:  # 저장한 pickle 불러오기
            self.classifier = pickle.load(f2)

        self.diary_db = database.DiaryManager()
        self.c_text_db = database.ConvertedTextManager()
        self.grammar = """
        NP: {<N.*>*<Suffix>?}   # Noun phrase
        VP: {<V.*>*}            # Verb phrase
        AP: {<A.*>*}            # Adjective phrase
        """
        self.parser = nltk.RegexpParser(self.grammar)

    def find_locations(self, diary_id):
        location_list = []
        # sentence_list = self.diary_db.retrive_setence_list_from_diary(diary_id)
        sentence_list = [["오빠닭에서 치킨을 먹었다 어제"], ["나무 위에서 노래를 부르는 새들은 기분이 좋은가 봐요."], ["사람들이 운동장에서 공을 차고 있어요."],["우리는 어제 중국식당에서 저녁을 먹었어요."], ["바닷가로 오십시오."]]
        for sentence in sentence_list:
            chunks = self.parser.parse(self.tokenizer.pos(sentence[0]))
            print("pos : ", Kkma().pos(sentence[0]))
            # chunks = self.parser.parse(self.tokenizer.pos(sentence['text']))
            for idx, chunk in enumerate(chunks):
                if chunk[0] == '에서' or chunk[0] == '에' or chunk[0] == '으로' or chunk[0] == '로':
                    if chunk[1] == 'Josa':
                        tmp = str()
                        for c in chunks[idx-1]:
                            tmp += " " + c[0]
                        location_list.append(tmp.strip())
        return location_list

    def find_most_similar_docs(self, query_sentence, limit=10):
        """ Find N most similar sentences
        N is min(<# of sentences>, limit).

        :param query_sentence: a sentence regarded as a basis for similarity calculation
        :param limit: an integer
        :return: a list of diary_id in diary_sentences
        """

        diary_list = self.c_text_db.get_converted_text_list({'user_id': self.user_id})
        similarity_list = []
        c_text_ids = []

        text = nltk.Text(self.tokenizer.nouns(query_sentence), name='NMSC')
        selected_words = [f[0] for f in text.vocab().most_common(3)]
        related_list = []
        for word in selected_words:
            ko_db = database.NLPkoDictManager()
            results = ko_db.retrieve_collection_dic(word)
            if results is not None:
                for result in results:
                    related_list.append(result['means'])

        related_list = list(set(related_list))
        query_sentence = self.text_to_vector(query_sentence)
        query_sentence = query_sentence + Counter(related_list)

        for diary in diary_list:
            similarity_score = self._calc_similarity(self.text_to_vector(diary['text']), query_sentence)
            similarity_list.append(similarity_score)
            c_text_ids.append(diary['c_text_id'])

        dt = DataFrame({'similarity': similarity_list, 'c_text_id': c_text_ids})
        return dt.sort_values(by='similarity', ascending=False)

    def find_sementic(self, diary_id):
        prob_list = []
        sentence_list = self.diary_db.retrieve_sentence_list_from_diary(diary_id)
        for sentence in sentence_list:
            tagged_sentence = self.tag_rename(sentence['text'])
            tagged_sentence = self.model.infer_vector(tagged_sentence)
            prob = self.classifier.predict_proba(tagged_sentence.reshape(1, -1))[0][1]
            if 0.4 <= prob <= 0.6:
                continue
            prob_list.append(prob)

        if 0.4 <= np.mean(prob_list) <= 0.6 or prob_list == []:
            return None
        return np.mean(prob_list)

    def _calc_similarity(self, s1, s2):
        """ Calculate similarity score among the two input sentences

        :param s1: sentence 1
        :param s2: sentence 2
        :return: a similarity score
        """
        intersection = set(s1.keys()) & set(s2.keys())
        numerator = sum([s1[x] * s2[x] for x in intersection])

        sum1 = sum([s1[x] ** 2 for x in s1.keys()])
        sum2 = sum([s2[x] ** 2 for x in s2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def tokenize(self, sentence):
        tokens_list = []
        p = 0
        tokens = self.tokenizer.pos(sentence)
        for idx, token in enumerate(tokens):
            if token[0] == r"[.?!]":
                tokens_list.append(tokens[p:idx+1])
                p = idx + 1
        return tokens_list

    def slice_sentence(self, sentence):
        sentence = re.split(r"[.?!]", sentence)
        if sentence[-1] == '':
            sentence = sentence[:-1]
        return sentence

    def text_to_vector(self, _text):
        words = self.tokenizer.nouns(_text)
        return Counter(words)

    def token(self, text):
        return ['/'.join(t) for t in self.tokenizer.pos(text)]

    def tag_rename(self, text):
        TaggedDocument = namedtuple('TaggedDocument', 'words')
        text = self.token(text)
        return TaggedDocument(text)[0]

def read_data(filename):
    with codecs.open(filename, "r") as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data



if __name__ == '__main__':
    # t = Twitter()
    # with open('doc.model', 'rb') as train_file:  # 저장한 pickle 불러오기
    #     model = pickle.load(train_file)
    # sentences = []
    # analyzer = SimilarityAnalyzer(t)
    # sentence_datas = read_data("train.txt")
    # for s in sentence_datas:
    #     sentences.append(analyzer.text_to_vector(s[0]))
    # test = "함부로 다른 사람 영역에 간섭하지 않는 건 다치지 않기 위해서다."
    # print("asfd : " , t.nouns(test))
    # text = nltk.Text(t.nouns(test), name='NMSC')
    # selected_words = [f[0] for f in text.vocab().most_common(3)]
    # related_list = []
    # print("Selected nouns : : ", selected_words)

    # similarity_list = analyzer.find_most_similar_docs(query_sentence=test, user_id='lhs')
    # print("sim_list : ", similarity_list[:3])
    # jpype.attachThreadToJVM()

    test = "함부로 다른 사람 영역에 간섭하지 않는 건 다치지 않기 위해서다."
    analyzer = SimilarityAnalyzer(user_id='lhs')
    # print("prob : ", analyzer.find_sementic(diary_id=222))
    # analyzer.find_locations(diary_id=223)
    print(analyzer.find_locations(diary_id=227))
    with open('study95model', 'rb') as train_file:  # 저장한 pickle 불러오기
        model = pickle.load(train_file)

    model.inf