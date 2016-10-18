import nltk
from nltk import load_parser
from nltk.parse import stanford

from diary_nlp.nlp_en_tmp import settings


class NLQuery:
    def __init__(self):
        self.dep_parser = nltk.parse.stanford.StanfordDependencyParser(
            path_to_jar=settings.PATH_ST_PARSER,
            path_to_models_jar=settings.PATH_ST_PARSER_MODEL)
        self.parser = nltk.parse.stanford.StanfordParser(
            path_to_jar=settings.PATH_ST_PARSER,
            path_to_models_jar=settings.PATH_ST_PARSER_MODEL,
            encoding='utf-8')

    def show_query(self, nl_query):
        query_parser = load_parser('grammars/book_grammars/sql0.fcfg')
        trees = list(query_parser.parse(nl_query.split()))
        answer = trees[0].label()['SEM']
        answer = [s for s in answer if s]
        q = ' '.join(answer)


def demo():
    nl_query = "What cities are located in China"
    nlq = NLQuery()
    nlq.show_query(nl_query)

if __name__ == '__main__':
    demo()