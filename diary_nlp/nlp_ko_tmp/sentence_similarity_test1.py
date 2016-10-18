import re, math
from konlpy.tag import Twitter
from collections import Counter
from nltk.corpus import stopwords

twitter = Twitter()
WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = twitter.nouns(text)
     print("wo : ", words)
     return Counter(words)

text1 = "이제 문장간 영향력을 행사하는 정도'가 정의 되었다. 두 문장에서 공통으로 등장하는 명사들을 멀티셋으로 만든 후 나오는 Jaccard Index의 값을 '서로에게 이만큼 영향을 준다'로 정의한 것이다."
text2 = "사실 이 말고도 여러가지 방법이 있다. 명사 말고 동사도 포함시켜도 되고, 레벤슈타인 거리를 사용해도 되고, TF-IDF를 계산해서 문장간의 각도를 사용해도 된다."

vector1 = text_to_vector(text1)
vector2 = text_to_vector(text2)

cosine = get_cosine(vector1, vector2)

print ('Cosine:', cosine)