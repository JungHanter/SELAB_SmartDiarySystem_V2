from diary_analyzer import tagger

print("Start tagging eliz's diaries.\n")
with open("wordset/eliz.txt", "r") as file:
    cnt = 1
    while True:
        diary = file.readline()
        if not diary:
            break
        print("start tagging diary #%s" % cnt)
        diary_tags = tagger.tag_pos_doc(diary, True)
        print(diary_tags[1])
        print("create pickle for tags of diary #%s" % cnt)
        tagger.tags_to_pickle(diary_tags, "diary_pickles/eliz_" + str(cnt) + ".pkl")
        print()
        cnt += 1
    file.close()
