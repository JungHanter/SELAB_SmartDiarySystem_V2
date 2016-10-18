import codecs
import nltk
import pickle
from konlpy.tag import Twitter
from pprint import pprint
from collections import namedtuple
from gensim.models import doc2vec
import multiprocessing
from sklearn.linear_model import LogisticRegression
import gensim
man = []

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def read_data(filename):
    with codecs.open(filename, "r", "utf-8") as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]   # header 제외
    return data

def term_exists(doc):
    return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}

pos_tagger = Twitter()

# train_data = read_data('ratings_train.txt')
# test_data = read_data('ratings_test.txt')
# print(len(train_data))      # nrows: 150000
# print(len(train_data[0]))   # ncols: 3
# print(len(test_data))       # nrows: 50000
# print(len(test_data[0]))     # ncols: 3
#
# train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
# test_docs = [(tokenize(row[1]), row[2]) for row in test_data]

# with open('train_data.txt', 'wb') as train_file:   # train set 저장 하는 코드
#     pickle.dump(train_docs, train_file)

# with open('test_data.txt', 'wb') as test_file:     # test set 저장 하는 코드
#     pickle.dump(test_docs, test_file)

with open('train_data.txt', 'rb') as train_file, open('test_data.txt', 'rb') as test_file:  #저장한 pickle 불러오기
    train_docs = pickle.load(train_file)
    test_docs = pickle.load(test_file)

print(train_docs[0])
tokens = []
for d in train_docs:
    for t in d[0]:
        tokens.append(t)
print(tokens[:13])
# tokens = [t for d in train_docs for t in d[0]]
# print(tokens[:13])

text = nltk.Text(tokens, name='NMSC')
print(text.similar('영화/Noun'))
# print(len(text.tokens))                 # returns number of tokens
# => 2194536
# print(len(set(text.tokens)))            # returns number of unique tokens
# => 48765
# pprint(text.vocab().most_common(10))    # returns frequency distribution


# 여기서는 최빈도 단어 2000개를 피쳐로 사용
# WARNING: 쉬운 이해를 위한 코드이며 time/memory efficient하지 않습니다
selected_words = [f[0] for f in text.vocab().most_common(2000)]
print("sele : ",  selected_words)
# 시간 단축을 위한 꼼수로 training corpus의 일부만 사용할 수 있음
train_docs = train_docs[:1000]
# train_docs = train_docs[:10000]
print("train : ", test_docs)
# for d, c in train_docs[:10]:
#     print("d : ", d)
#     print("c : ", c)
#     print(term_exists(d))
#     print(term_exists(d), c)
train_xy = [(term_exists(d), c) for d, c in train_docs]
test_xy = [(term_exists(d), c) for d, c in test_docs[:100]]
print("ok")
classifier = nltk.NaiveBayesClassifier.train(train_xy)
print(nltk.classify.accuracy(classifier, test_xy))
# => 0.80418
# classifier.show_most_informative_features(10)


######################################################################################
# print("시작")
# TaggedDocument = namedtuple('TaggedDocument', 'words')
# TaggedDocument = namedtuple('TaggedDocument', 'words tags')
# 여기서는 15만개 training documents 전부 사용함
# tagged_train_docs = [TaggedDocument(d) for d, c in train_docs[:10000]]
# tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs[:10000]]
# print(tagged_train_docs[0])

# for d, c in test_docs[:10]:
#     print(TaggedDocument(d, [c]))
# tagged_test_docs = [TaggedDocument(d) for d, c in test_docs[:100]]
# tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_docs[:100]]
# print("학습 완료")
#
# doc_vectorizer = doc2vec.Doc2Vec(size=300, min_alpha=0.025)
# doc_vectorizer.build_vocab(tagged_train_docs)
#
# for epoch in range(2):
#     doc_vectorizer.train(tagged_train_docs)
#     doc_vectorizer.alpha -= 0.002  # decrease the learning rate
#     doc_vectorizer.min_alpha = doc_vectorizer.alpha  # fix the learning rate, no decay
# # To save
# # doc_vectorizer.save('doc2vec.model')
#
# model = doc_vectorizer.load("doc2vec.model")
# with open('doc.model', 'wb') as train_file:   # train set 저장 하는 코드
#     pickle.dump(model, train_file)


# with open('doc.model', 'rb') as train_file:  #저장한 pickle 불러오기
#     model = pickle.load(train_file)


# print("model load ok")
# pprint(model.most_similar(positive='공포/Noun', topn=10))
# print("doc : ", tagged_train_docs[0])
# print("doc : ", tagged_train_docs[0].words)
# train_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
# train_y = [doc.tags[0] for doc in tagged_train_docs]
# print("train ok")
# # len(train_x)       # 사실 이 때문에 앞의 term existance와는 공평한 비교는 아닐 수 있다
# # => 150000
# # len(train_x[0])
# # => 300
# test_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_test_docs]
# test_y = [doc.tags[0] for doc in tagged_test_docs]
# print("test ok")
# # len(test_x)
# # # => 50000
# # len(test_x[0])
# # # => 300
# #
# classifier = LogisticRegression(random_state=1234)
# classifier.fit(train_x, train_y)
# print("score : ", classifier.score(test_x, test_y))
# # => 0.78246000000000004