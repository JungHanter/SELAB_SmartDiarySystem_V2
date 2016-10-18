# FIND WORD COLLOCATIONS FROM A FILE
# WORD COLLOCATIONS WILL GIVE THE MUTUALY RELATED WORDS

# !/usr/bin/python
from nltk.book import *
import nltk
from nltk.corpus import stopwords

f = open('sample1.txt', 'rU')
text = f.read()
text = ' '.join([word for word in text.split() if word not in (stopwords.words('english'))])
wordtext1 = text.split()
abstracts = nltk.Text(text1)
abstracts.collocations()