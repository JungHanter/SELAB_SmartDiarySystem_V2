import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JAR_DIR = os.path.join(str(Path(__file__).parents[1]), 'stanford-nlp')
# JAR_DIR = "/Users/hanter/SEL/Smart Diary System/stanford_nlp/"

STANFORD_CORE_BASE_PATH = os.path.join(JAR_DIR, "stanford-corenlp-full-2015-12-09")
STANFORD_CORE_PATH = os.path.join(STANFORD_CORE_BASE_PATH, "stanford-corenlp-3.6.0.jar")

STANFORD_TAGGER_BASE_PATH = os.path.join(JAR_DIR, "stanford-postagger-full-2015-12-09")
STANFORD_TAGGER_PATH = os.path.join(STANFORD_TAGGER_BASE_PATH, 'stanford-postagger.jar')
STANFORD_TAGGER_MODEL_PATH = os.path.join(os.path.join(STANFORD_TAGGER_BASE_PATH, 'models'),
                                          'english-bidirectional-distsim.tagger')

STANFORD_PARSER_BASE_PATH = os.path.join(JAR_DIR, "stanford-parser-full-2015-12-09")
# STANFORD_PARSER_BASE_PATH = os.path.join(JAR_DIR, "stanford-parser-full-2015-04-20")
STANFORD_PARSER_PATH = os.path.join(STANFORD_PARSER_BASE_PATH, 'stanford-parser.jar')
STANFORD_PARSER_MODEL_PATH = os.path.join(STANFORD_PARSER_BASE_PATH, 'stanford-parser-3.6.0-models.jar')
# STANFORD_PARSER_MODEL_PATH = os.path.join(STANFORD_PARSER_BASE_PATH, 'stanford-parser-3.5.2-models.jar')1

STANFORD_NER_BASE_PATH = os.path.join(JAR_DIR, "stanford-ner-2015-12-09")
STANFORD_NER_PATH = os.path.join(STANFORD_NER_BASE_PATH, "stanford-ner.jar")
STANFORD_NER_CLASSIFIER_PATH = os.path.join(os.path.join(STANFORD_NER_BASE_PATH, 'classifiers'),
                                            "english.all.3class.distsim.crf.ser.gz")
