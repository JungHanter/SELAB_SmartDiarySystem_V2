import os
from os import path


PATH_BASE = os.path.join(path.dirname(__file__), '..')
PATH_JARS = os.path.join(PATH_BASE, 'jars')
# stanford parser, stanford dependency parser
PATH_ST_PARSER = \
    os.path.join(PATH_JARS, 'stanford-parser-full-2015-04-20', 'stanford-parser.jar')
PATH_ST_PARSER_MODEL = \
    os.path.join(PATH_JARS, 'stanford-parser-full-2015-04-20', 'stanford-parser-3.5.2-models.jar')

# stanford neural parser
PATH_ST_CORE = \
    os.path.join(PATH_JARS, 'stanford-corenlp-full-2015-12-09', 'stanford-corenlp-3.6.0.jar')
PATH_ST_CORE_MODEL = \
    os.path.join(PATH_JARS,  'stanford-corenlp-full-2015-12-09', 'stanford-corenlp-3.6.0-models.jar')

# stanford ner parser
PATH_ST_NER = \
    os.path.join(PATH_JARS,  'stanford-ner-2015-04-20', 'stanford-ner.jar')
PATH_ST_NER_MODEL = \
    os.path.join(PATH_JARS, 'stanford-ner-2015-04-20', 'classifiers', 'english.all.3class.distsim.crf.ser.gz')

# nrc emotion data set
PATH_NRC_DATA = \
    os.path.join(PATH_BASE, 'emotions', 'NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt')

PATH_SAMPLES = os.path.join(PATH_BASE, 'samples')
PATH_ST_TAGGER = os.path.join(PATH_JARS, 'stanford-postagger-full-2015-04-20')
