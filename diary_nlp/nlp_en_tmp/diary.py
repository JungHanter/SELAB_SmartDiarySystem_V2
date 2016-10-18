import nltk
from nltk.parse import stanford

from diary_nlp.nlp_en_tmp import settings


class DiaryManager:
    def __init__(self):
        self.parser = nltk.parse.stanford.StanfordParser(
            path_to_jar=settings.PATH_ST_PARSER,
            path_to_models_jar=settings.PATH_ST_PARSER_MODEL,
            encoding='utf-8')

    def get_similar_sentence(self, basic_sent, sent_obj_list):
        d = 1
