# KEY TERM SELECTION USING TOPIA
import json
import sys
from topia.termextract import tag
from topia.termextract import extract
import nltk


def uniqify(seq, idFun=None):
    # order preserving
    if idFun is None:
        def idFun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idFun(item)

    if marker in seen:
        pass
    seen[marker] = 1
    result.append(item)
    return result


def build(language='english'):
    # initialize the tagger with the required language
    tagger = tag.Tagger(language)
    tagger.initialize()

    # create the extractor with the tagger
    extractor = extract.TermExtractor(tagger=tagger)
    # invoke tagging the text
    s = nltk.data.load('corpora/operating/td1.txt', format='raw')
    extractor.tagger(s)
    # extract all the terms, even the &amp;quot;weak&amp;quot; ones
    extractor.filter = extract.DefaultFilter(singleStrengthMinOccur=1)
    # extract
    return extractor(s)


resultList = []

# get a results
result = build('english')
# or result = build('dutch')

for r in result:
    # discard the weights for now, not using them at this point and defaulting to lowercase keywords/tags
    resultList.append(r[0].lower())

# dump to JSON output
a = json.dumps(sorted(uniqify(resultList)))
fd = open("/home/jayakrishnan/finalout/a1term.txt", "w")
fd.write(a)
fd = open("/home/jayakrishnan/finalout/a1term.txt", "r")
fd1 = open("/home/jayakrishnan/finalout/a1term1.txt", "w")
m = fd.read()
y = [x.strip() for x in m.split(',')]
for w in y:
    w = w + '\n'
    print
    w
    fd1.write(w)
fd.close()
fd1.close()