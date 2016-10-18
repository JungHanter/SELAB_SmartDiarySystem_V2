import sys
from diary_nlp import nlp_ko
from konlpy.tag import Twitter

def tokenize(text):
    kor = nlp_ko.SimilarityAnalyzer(Twitter())

    # Parse into Sentence
    kor.slice_sentence(text)

    # Parse into Sent Element
    sent_list = kor.slice_sentence(text)
    token_list = kor.tokenize(text)

    print(sent_list)
    print(token_list)

if __name__ == '__main__':
    tokenize(sys.argv[1])
