import nltk
from nltk.parse import stanford
from nltk.tokenize import word_tokenize

from diary_nlp.nlp_en_tmp import settings


class Similar:
    def __init__(self):
        import string
        self.dep_parser = nltk.parse.stanford.StanfordDependencyParser(
            path_to_jar=settings.PATH_ST_PARSER,
            path_to_models_jar=settings.PATH_ST_PARSER_MODEL)
        self.parser = nltk.parse.stanford.StanfordParser(
            path_to_jar=settings.PATH_ST_PARSER,
            path_to_models_jar=settings.PATH_ST_PARSER_MODEL,
            encoding='utf-8')
        self.meaningless = nltk.corpus.stopwords.words('english')
        self.meaningless.extend(string.punctuation)
        self.meaningless.append('')

    def get_similarity_using_nothing(self, sent1, sent2):
        # get tokens except for meaningless token
        tokens1 = word_tokenize(sent1)
        tokens2 = word_tokenize(sent2)
        meaningful1 = [token.lower() for token in tokens1
                       # if token not in self.meaningless
                       ]
        meaningful2 = [token.lower() for token in tokens2
                       # if token not in self.meaningless
                       ]
        # calculate
        common_words = (list(set(meaningful1) & set(meaningful2)))
        all_words = list(set(meaningful1) | set(meaningful2))
        ratio = (len(common_words) / len(all_words))
        # print
        print("\nsimilarity_using_nothing : ", ratio)
        print("common words: ", common_words, "\nall words   : ", all_words)

    def get_similarity_using_pos(self, sent1, sent2):
        # get tokens except for meaningless token
        tokens1 = word_tokenize(sent1)
        tokens2 = word_tokenize(sent2)
        pos1 = nltk.pos_tag(tokens1)
        pos2 = nltk.pos_tag(tokens2)
        meaningful1 = [(token, pos[0:2]) for token, pos in pos1 if token not in self.meaningless]
        meaningful1_lower = [(token.lower(), pos) for (token, pos) in meaningful1]
        meaningful2 = [(token, pos[0:2]) for token, pos in pos2 if token not in self.meaningless]
        meaningful2_lower = [(token.lower(), pos) for (token, pos) in meaningful2]

        common_words = (list(set(meaningful1_lower) & set(meaningful2_lower)))
        case_sensitive_common_words = (list(set(meaningful1) & set(meaningful2)))

        all_words = (list(set(meaningful1) | set(meaningful2)))

        ratio = (len(common_words) / len(all_words))
        case_sensitive_ratio = (len(case_sensitive_common_words) / len(all_words))
        print("\nsimilarity_using_pow_tagger : ", ratio)
        print("similarity_using_pow_tagger : ", case_sensitive_ratio, "<- case sensitivity")
        print("common words: ", common_words,
              "\ncommon words: ", case_sensitive_common_words, "<- case sensitivity",
              "\nall words   : ", all_words)

    def get_similarity_using_lemma(self, sent1, sent2):
        # get tokens except for meaningless token
        tokens1 = word_tokenize(sent1)
        tokens2 = word_tokenize(sent2)
        from nltk.stem import WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        meaningful1 = [wordnet_lemmatizer.lemmatize(token.lower()) for token in tokens1 if token not in self.meaningless]
        meaningful2 = [wordnet_lemmatizer.lemmatize(token.lower()) for token in tokens2 if token not in self.meaningless]
        # calculate
        common_words = (list(set(meaningful1) & set(meaningful2)))
        all_words = list(set(meaningful1) | set(meaningful2))
        ratio = (len(common_words) / len(all_words))
        # print
        print("\nsimilarity_using_wordnet_lemmatizer : ", ratio)
        print("common words: ", common_words, "\nall words   : ", all_words)

    def get_similarity_using_stem(self, sent1, sent2):
        # get tokens except for meaningless token
        tokens1 = word_tokenize(sent1)
        tokens2 = word_tokenize(sent2)
        from nltk.stem.porter import PorterStemmer
        from nltk.stem.lancaster import LancasterStemmer
        from nltk.stem import SnowballStemmer
        porter_stemmer = PorterStemmer()
        lancaster_stemmer = LancasterStemmer()

        snowball_stemmer = SnowballStemmer('english')
        meaningful1 = [token.lower() for token in tokens1 if
                       token not in self.meaningless]
        meaningful2 = [token.lower() for token in tokens2 if
                       token not in self.meaningless]

        stems1 = [snowball_stemmer.stem(token) for token in meaningful1]
        stems2 = [snowball_stemmer.stem(token) for token in meaningful2]
        # calculate
        common_words = (list(set(stems1) & set(stems2)))
        all_words = list(set(stems1) | set(stems2))
        ratio = (len(common_words) / len(all_words))
        # print
        print("\nsimilarity_using_stem : ", ratio)
        print("common words: ", common_words, "\nall words   : ", all_words)

    def get_similarity(self, sent1, sent2):

        self.get_similarity_using_nothing(sent1, sent2)
        self.get_similarity_using_pos(sent1, sent2)
        self.get_similarity_using_stem(sent1, sent2)
        self.get_similarity_using_lemma(sent1, sent2)
        # print("how it work?: common words / all words")

def demo():
    print("nltk version should be 3.1. current version is ", nltk.__version__, )
    file = open('/'.join([settings.PATH_BASE, 'example1']), mode='r', encoding='utf-8')
    text = file.read()
    sent1 = """When did I wake up yesterday?"""
    sent2 = """I woke up early"""

    similar = Similar()
    similar.get_similarity(sent1, sent2)


if __name__ == '__main__':
    demo()