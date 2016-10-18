from konlpy.tag import Twitter
from collections import Counter

twitter = Twitter()

bow1 = Counter(twitter.nouns("이제 '문장간 영향력을 행사하는 정도'가 정의 되었다. 두 문장에서 공통으로 등장하는 명사들을 멀티셋으로 만든 후 나오는 Jaccard Index의 값을 '서로에게 이만큼 영향을 준다'로 정의한 것이다."))
bow2 = Counter(twitter.nouns("사실 이 말고도 여러가지 방법이 있다. 명사 말고 동사도 포함시켜도 되고, 레벤슈타인 거리를 사용해도 되고, TF-IDF를 계산해서 문장간의 각도를 사용해도 된다."))

j_index = sum((bow1 & bow2).values()) / sum((bow1 | bow2).values())
print("유사도  : ", j_index)
print("  afsd  : ", twitter.nouns("이제 '문장간 영향력을 행사하는 정도'가 정의 되었다. 두 문장에서 공통으로 등장하는 명사들을 멀티셋으로 만든 후 나오는 Jaccard Index의 값을 '서로에게 이만큼 영향을 준다'로 정의한 것이다."))
