import os
import re

import datetime
import nltk
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from nltk.corpus import wordnet as wn
from nltk import StanfordTokenizer, StanfordPOSTagger, WordNetLemmatizer, PorterStemmer
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag.stanford import StanfordNERTagger

from diary_analyzer import tools
from smart_diary_system import settings

ELEMENT_SUBJECT = "Subject"
ELEMENT_VERB = "Verb"
ELEMENT_OBJECT = "Object"
ELEMENT_COMPLEMENT = "Complement"
ELEMENT_ADVERB = "Adverb"
ELEMENT_AUXILIARYVERB = "Auxiliary Verb"

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def tag_dep(tokens, parser):
    """ Determine roles of each word

    :param tokens: [[word, tag], ...]
    :return: [[word, tag, dir, dep, role], ...]
    """
    result = parser.tagged_parse(tokens)
    dep = next(result)
    roles = [morpheme.split('\t') for morpheme in dep.to_conll(4).split('\n')]

    tagsiter = iter(tokens)
    result = []
    for role in roles:
        while True:
            try:
                tag = next(tagsiter)
                if tag[0] == role[0]:
                    result.append([tag[0], tag[1], role[2], role[3]])
                    break
                else:
                    result.append([tag[0], tag[1], None, tag[1]])
            except Exception as e:
                break
    # return determine_role(result)
    return result


def determine_role(s):
    roles = []
    for w in s:
        if 'subj' in w[3]:
            roles.append(ELEMENT_SUBJECT)
        elif 'obj' in w[3]:
            roles.append(ELEMENT_OBJECT)
        elif w[1].startswith('VB') and w[3] == 'aux':
            roles.append(ELEMENT_AUXILIARYVERB)
        elif w[1].startswith('VB'):
            roles.append(ELEMENT_VERB)
        elif w[1].startswith('RB'):
            roles.append(ELEMENT_ADVERB)
        elif w[2] != 'compound' and 'comp' in w[3] or 'root' == w[3]:
            roles.append(ELEMENT_COMPLEMENT)
        else:
            roles.append(None)
    return roles


def extract_activities(diaries, diary_dates, activity_word_set):
    """ Extract activities of each word based on WordNet

    # :param diaries: [[[word, tag, dir, dep, role, lemma, named], ...]]
    :param diaries: [[[word, tag, dir, dep, named], ...]]
    :return:
    """

    def determine_activity(word, pos):
        """Compare the words of diaries to the sets of activity-related words including WordNet and Wikipedia Hobby List
        """
        if word is None:
            return None
        if word in ['set', 'make', 'have', 'take', 'get', 'use']:
            return False
        if type(word) is str:
            synsets = wn.synsets(word, pos=pos)
        if len(set(activity_word_set).intersection(set(synsets))) > 0:
            return True
        return False

    occurrence_list = {}  # activity -> list of date-time instances
    for d, t in zip(diaries, diary_dates):
        last_time_words = set([])
        for s in d:
            s = np.array(s)

            # --------------------
            # Skip the sentence as it is not for today
            tomorrow_words = s[np.char.lower(s[:, 0].tolist()) == 'tomorrow']
            date_words = s[s[:, -1] == 'DATE']
            if len(date_words) > 0 or len(tomorrow_words):
                continue

            # --------------------
            # Identify day timezones
            is_new_occurrence = True
            time_words = s[s[:, -1] == 'TIME']
            if len(time_words) > 0:
                _time_word_set = set(time_words[:, 0].tolist())
                _time_word_set = {time_word.lower() for time_word in _time_word_set}
                if len(_time_word_set.intersection(last_time_words)) > 0:
                    last_time_words = time_words
                    is_new_occurrence = False

            # --------------------
            # Identify subjects, verbs, and objects
            subject_mask = np.array(['subj' in dep for dep in s[:, 3]], dtype=bool)
            subject_words = s[subject_mask]
            subject_idxs = np.add(np.nonzero(subject_mask)[0], 1)
            subject_articles = []
            for subject_idx in subject_idxs:
                subject_articles.append(s[np.logical_and(np.in1d(s[:, 3], ['det']), np.in1d(s[:, 2], [subject_idx]))].tolist())
            subject_articles = np.array(subject_articles)
            # Identify active and passive subjects
            subject_active_words = []
            subject_pass_words = []
            subject_form = None
            if subject_words.shape[0] > 0:
                subject_pass_mask = np.array(['pass' in dep for dep in subject_words[:, 3]], dtype=bool)
                subject_pass_words =  subject_words[subject_pass_mask]
                subject_pass_idxs = subject_idxs[subject_pass_mask]
                subject_pass_articles = subject_articles[np.logical_not(subject_pass_mask)]
                subject_active_words = subject_words[np.logical_not(subject_pass_mask)]
                subject_active_idxs = subject_idxs[np.logical_not(subject_pass_mask)]
                subject_active_articles = subject_articles[np.logical_not(subject_pass_mask)]
                subject_form = 'passive' if len(subject_pass_words) > 0 else 'active'
            # Identify verbs
            roles = np.array(determine_role(s))
            verb_mask = roles == ELEMENT_VERB
            verb_words = s[verb_mask]
            verb_idxs = np.add(np.nonzero(verb_mask)[0], 1)
            # Identify objects
            object_mask = roles == ELEMENT_OBJECT
            object_words = s[object_mask]
            object_idxs = np.add(np.nonzero(object_mask)[0], 1)
            object_articles = s[np.logical_and(np.in1d(s[:, 3], ['det']), np.in1d(s[:, 2], object_idxs))]

            # --------------------
            # Identify verb-object pairs
            verb_object_pairs = {}
            for idx, word in zip(verb_idxs, verb_words):
                pos = wn.NOUN
                if 'VB' in word[1]:
                    pos = wn.VERB
                _word = stemmer.stem(word[0]).lower()
                _word = lemmatizer.lemmatize(_word, pos=pos).lower()
                if determine_activity(_word, pos=pos):
                    verb_object_pairs['%s' % (idx)] = _word
            for word in s:
                if word[2] is not None and int(word[2]) - 1 in verb_idxs.tolist() and 'mod' in word[3]:
                    if '%s' % word[2] in verb_object_pairs:
                        verb_object_pairs['%s' % word[2]] += ' ' + _word

            # --------------------
            # Identify phrases
            phrase_words = [w for w in s if w[1] == 'NNP' and w[3] == 'root']

            # --------------------
            # Nothing is identified. Consider nouns.
            # Probably they are about the writer's activities.
            noun_words = []
            if len(subject_words) == 0 and len(verb_words) == 0 and len(object_words) == 0:
                noun_words = s[np.in1d(s[:, 3], ['nmod'])]

            # --------------------
            # Extract activities with object if the sentence is active-form and the subject is one of 'I' and 'We'
            if subject_form is None or subject_form == 'active':
                # Validate active subjects
                subject_valid = False
                if subject_form == 'active':
                    for word in subject_words:
                        subj = word[0].lower()
                        pos = word[1]
                        dep = word[3]
                        if subj in ['he', 'she', 'they', 'it']:
                            break
                        if (subj == 'i' or subj == 'we' or
                                ('csubj' in dep and 'had' != subj and 'having' != subj)):
                            subject_valid = True
                            break
                if subject_form is None or subject_valid:
                    for word in subject_active_words:
                        pos = wn.NOUN
                        if 'VB' in word[1]:
                            pos = wn.VERB
                        _word = stemmer.stem(word[0]).lower()
                        _word = lemmatizer.lemmatize(_word, pos=pos).lower()
                        if determine_activity(_word, pos):
                            if _word not in occurrence_list:
                                occurrence_list[_word] = [t]
                            elif is_new_occurrence and 'the' not in [article.lower() for articles in subject_pass_articles.tolist() for article in articles]:
                                occurrence_list[_word].append(t)

                    for word in object_words:
                        pos = wn.NOUN
                        if 'VB' in word[1]:
                            pos = wn.VERB
                        _word = stemmer.stem(word[0]).lower()
                        _word = lemmatizer.lemmatize(_word, pos=pos).lower()
                        if determine_activity(_word, pos):
                            if _word not in occurrence_list:
                                occurrence_list[_word] = [t]
                            elif is_new_occurrence and 'the' not in [article.lower() for articles in object_articles.tolist() for article in articles]:
                                occurrence_list[_word].append(t)

            # --------------------
            # Extract activities with subject if the sentence is passive-form
            elif subject_form == 'passive':
                for word in subject_pass_words:
                    pos = wn.NOUN
                    if 'VB' in word[1]:
                        pos = wn.VERB
                    _word = stemmer.stem(word[0]).lower()
                    _word = lemmatizer.lemmatize(_word, pos=pos).lower()
                    if determine_activity(_word, pos):
                        if _word not in occurrence_list:
                            occurrence_list[_word] = [t]
                        elif is_new_occurrence and 'the' not in [article.lower() for articles in subject_pass_articles.tolist() for article in articles]:
                            occurrence_list[_word].append(t)

            # --------------------
            # Extract activities with verb-object pairs
            for _, word in verb_object_pairs.items():
                if word not in occurrence_list:
                    occurrence_list[word] = [t]
                elif is_new_occurrence:
                    occurrence_list[word].append(t)

            # --------------------
            # Extract activities with phrases
            for word in phrase_words:
                pos = wn.VERB
                _word = stemmer.stem(word[0]).lower()
                _word = lemmatizer.lemmatize(_word, pos=pos).lower()
                if determine_activity(_word, pos):
                    if _word not in occurrence_list:
                        occurrence_list[_word] = [t]
                    elif is_new_occurrence:
                        occurrence_list[_word].append(t)

            # --------------------
            # Extract activities with nouns
            for word in noun_words:
                pos = wn.NOUN
                if 'VB' in word[1]:
                    pos = wn.VERB
                _word = stemmer.stem(word[0]).lower()
                _word = lemmatizer.lemmatize(_word, pos=pos).lower()
                if determine_activity(_word, pos):
                    if _word not in occurrence_list:
                        occurrence_list[_word] = [t]
                    elif is_new_occurrence:
                        occurrence_list[_word].append(t)

    # Return the activity-related words extracted from the diaries
    return occurrence_list


class ActivityPatternAnalyzer(object):

    def preprocess(self, diaries):
        # tokenizer = StanfordTokenizer(path_to_jar=os.path.join(settings.BASE_DIR,
        #                                                        'jars/stanford-postagger-full-2015-04-20/stanford-postagger-3.5.2.jar'))
        # pos_tagger = StanfordPOSTagger(path_to_jar=os.path.join(settings.BASE_DIR,
        #                                                     'jars/stanford-postagger-full-2015-04-20/stanford-postagger-3.5.2.jar'),
        #                            model_filename=os.path.join(settings.BASE_DIR,
        #                                                        'jars/stanford-postagger-full-2015-04-20/models/english-bidirectional-distsim.tagger'))
        # dep_parser = StanfordDependencyParser(
        #     path_to_jar=os.path.join(settings.BASE_DIR, 'jars/stanford-parser-full-2015-04-20/stanford-parser.jar'),
        #     path_to_models_jar=os.path.join(settings.BASE_DIR,
        #                                     'jars/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar'))
        # ner_tagger = StanfordNERTagger(os.path.join(settings.BASE_DIR,
        #                                             'jars/stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz'),
        #                                os.path.join(settings.BASE_DIR, 'jars/stanford-ner-2015-12-09/stanford-ner.jar'))
        tokenizer, pos_tagger, dep_parser, ner_tagger = tools.tokenizer, tools.pos_tagger, tools.dep_parser, tools.ner_tagger

        preprocessed_diaries = []
        for diary in diaries:
            # 1. Sentence Tokenization
            sent_list = nltk.sent_tokenize(diary)

            # 2. Word Tokenization
            token_list_2d = [tokenizer.tokenize(sent) for sent in sent_list]

            # 3. POS Tagging
            token_list_2d_pos = [pos_tagger.tag(token_list) for token_list in token_list_2d]

            # 4. Dependency Parsing
            token_list_2d_dep = [tag_dep(token_list, dep_parser) for token_list in token_list_2d_pos]

            # 5. Lemmatization
            # token_list_2d_lemma = [lemmatize(token_list) for token_list in token_list_2d_role]

            # 6. Named Entity Recognition
            token_list_2d_named = []

            word_list = []
            # for token_list in token_list_2d_lemma:
            for token_list in token_list_2d_dep:
                for token in token_list:
                    word_list.append(token[0])
                named_list = ner_tagger.tag(word_list)
                for token, (_, named_entity) in zip(token_list, named_list):
                    named_entity = named_entity if named_entity != 'O' else None
                    token.append(named_entity)
                token_list_2d_named.append(token_list)

            preprocessed_diaries.append(token_list_2d_named)

            return preprocessed_diaries

    def analyze_diaries(self, preprocessed_diaries, diary_dates):
        # ---------------------------------------------------
        # Step 1. Extracting Activities
        # ---------------------------------------------------
        print("Step 1. Extracting activity-related words ---------------------------")
        activity_word_set_path = os.path.join(settings.BASE_DIR, os.path.join('datasets', 'activity_word_set.pickle'))

        activity_word_set = set()
        if False and os.path.exists(activity_word_set_path):
            with open(activity_word_set_path, 'rb') as f:
                activity_word_set = pickle.load(f)
                activity_word_set = [wn.synset(word) for word in activity_word_set]
        else:
            with open(os.path.join(settings.BASE_DIR, 'datasets/activity_word_set.txt'), 'r') as f:
                activity_word_set = f.read().splitlines()
            activity_word_set = set([wn.synset(activity) for activity in activity_word_set])
            activity_word_set_ext = set([word.name() for word in activity_word_set])
            # for activity in activity_word_set:
            #     activity_word_set_ext.update(set([hyp.name() for hyp in activity.hyponyms()]))
            activity_word_set = activity_word_set_ext
            with open(activity_word_set_path, 'wb') as f:
                pickle.dump(activity_word_set_ext, f)
            activity_word_set = [wn.synset(word) for word in activity_word_set]

        occurrence_list = extract_activities(preprocessed_diaries, diary_dates, activity_word_set)
        print(occurrence_list)

        # ---------------------------------------------------
        # Step 2. Recurrence Analysis
        # ---------------------------------------------------
        print("Step 2. Analyzing recurrence -----------------------------------------")
        rec = {}
        # period = (diary_dates[-1] - diary_dates[0]).days
        period = len(preprocessed_diaries)
        print("\tPeriod: %s" % period)
        for activity, occurrences in occurrence_list.items():
            rec[activity] = len(occurrences) / period
        print(rec)

        # ---------------------------------------------------
        # Step 3. Frequency Analysis
        # ---------------------------------------------------
        print("Step 3. Analyzing frequencies -----------------------------------------")
        freq = {}
        for activity, occurrences in occurrence_list.items():
            interval_list = []
            for i in range(len(occurrences) - 1):
                interval_list.append((occurrences[i + 1] - occurrences[i]).days)
            if len(interval_list) > 0:
                if len(interval_list) == 1:
                    freq[activity] = interval_list[0]
                    # print(activity, interval_list)
                else:
                    Z = linkage(np.array(interval_list).reshape(len(interval_list), 1), 'ward')
                    interval_groups = pd.DataFrame(columns=['group', 'interval'])
                    for group, interval in zip(fcluster(Z, t=7., criterion='distance'), interval_list):
                        interval_groups = interval_groups.append({'group': group, 'interval': interval},
                                                                 ignore_index=True)
                    freq_values = interval_groups.groupby('group').mean()
                    freq[activity] = sorted(freq_values['interval'].tolist())
                    # print(activity, interval_list, interval_groups)
        print(freq)

        # ---------------------------------------------------
        # Step 4. Regularity Analysis
        # ---------------------------------------------------
        print("Step 4. Analyzing Regularity -----------------------------------------")
        reg = {}
        for activity, occurrences in occurrence_list.items():
            n_occurrences = len(occurrences)
            interval_matrix = np.zeros((n_occurrences, n_occurrences))
            for i in range(n_occurrences - 1):
                for j in range(i + 1, n_occurrences):
                    interval_matrix[i, j] = abs((occurrences[j] - occurrences[i]).days)

            interval_list = []
            for i in range(n_occurrences - 1):
                for j in range(i + 1, n_occurrences):
                    interval_list.append(interval_matrix[i, j])

            # K-Means clustering
            # if len(date_diff_list) > 0:
            #     kmeans = KMeans(n_clusters=min(5, len(date_diff_list)))
            #     kmeans.fit(np.array(date_diff_list).reshape(len(date_diff_list), 1))
            #     clusters = pd.DataFrame(columns=['cluster', 'date_diff'])
            #     for cluster, date_diff in zip(kmeans.labels_, date_diff_list):
            #         clusters = clusters.append({'cluster': cluster, 'date_diff': date_diff}, ignore_index=True)
            #     regularity_values = clusters.groupby('cluster').mean()
            #     activity_regularity[activity] = regularity_values['date_diff'].tolist()
            # else:
            #     activity_regularity[activity] = np.nan

            # Hierarchical clustering
            if len(interval_list) > 1:
                Z = linkage(np.array(interval_list).reshape(len(interval_list), 1), 'ward')
                interval_groups = pd.DataFrame(columns=['group', 'interval'])
                for group, interval in zip(fcluster(Z, t=7., criterion='distance'), interval_list):
                    interval_groups = interval_groups.append({'group': group, 'interval': interval}, ignore_index=True)
                regularity_values = interval_groups.groupby('group').mean()
                reg[activity] = sorted(regularity_values['interval'].tolist())
                # print(activity, interval_list, interval_groups)

        regularity_df = pd.DataFrame(columns=['activity', 'regularity'])
        for activity, regularity in reg.items():
            regularity_df = regularity_df.append({'activity': activity, 'regularity': regularity}, ignore_index=True)
        print(reg)

        return rec, freq, reg