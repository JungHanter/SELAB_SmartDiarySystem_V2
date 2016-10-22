from diary_analyzer import _configure as config
from nltk import StanfordTokenizer, StanfordPOSTagger
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag import StanfordNERTagger

tokenizer = StanfordTokenizer(path_to_jar=config.STANFORD_TAGGER_PATH)
pos_tagger = StanfordPOSTagger(path_to_jar=config.STANFORD_TAGGER_PATH,
                               model_filename=config.STANFORD_TAGGER_MODEL_PATH)
dep_parser = StanfordDependencyParser(path_to_jar=config.STANFORD_PARSER_PATH,
                                      path_to_models_jar=config.STANFORD_PARSER_MODEL_PATH)
ner_tagger = StanfordNERTagger(config.STANFORD_NER_CLASSIFIER_PATH,
                               config.STANFORD_NER_PATH, encoding='utf-8')

