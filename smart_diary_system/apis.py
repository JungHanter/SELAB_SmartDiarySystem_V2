import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from smart_diary_system import database
from smart_diary_system import security
import logging
from diary_nlp import nlp_ko
from konlpy.tag import Twitter
from konlpy.tag import Kkma
from diary_nlp import nlp_en
from langdetect import detect
import os
import shutil
import jpype
from django.http import QueryDict
from django.conf import settings
from django.http import HttpResponse
import tarfile

import subprocess
from pydub import AudioSegment
import shlex

from pprint import pprint

logging.basicConfig(
    format="[%(name)s][%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()],
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

salt = 'lovejesus'


@csrf_exempt
def manage_user(request, option=None):
    logger.debug(request)
    if request.method == 'POST':
            if option is None:  # Register
                try:
                    # USER INFO from APP
                    data = json.loads(request.body.decode('utf-8'))
                    logger.debug("INPUT %s", data)
                    if not ('timestamp' in data):
                        data['timestamp'] = ''
                    if not ('gender' in data):
                        data['gender'] = ''
                    user_manager = database.UserManager()
                    # Encrypting Password
                    cipher = security.AESCipher(salt)
                    enc = cipher.encrypt(data.get('password'))
                    # DB Transaction
                    result = user_manager.add_user(user_id=data.get('user_id'), password=enc, name=data.get('name'),
                                                   timestamp=data.get('timestamp'), gender=data.get('gender'))

                    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                    os.mkdir(ROOT_DIR + '/uploaded/' + str(data.get('user_id')))

                    if result:
                        return JsonResponse({'register': True})
                    else:
                        return JsonResponse({'register': False})
                except Exception as exp:
                    logger.exception(exp)
                    return JsonResponse({'register': False})

            elif option == 'login':  # Login
                try:
                    # PASSWORD(plain) from APP
                    data = json.loads(request.body.decode('utf-8'))
                    logger.debug("INPUT %s", data)
                    user_manager = database.UserManager()
                    password_from_user = data.get('password')

                    # USER_ID CHECK from DB
                    user_info = user_manager.get_user(data.get('user_id'))
                    if user_info is None:
                        return JsonResponse({'login': False})
                    else:
                        # PASSWORD(encrypted) from DB
                        cipher = security.AESCipher(salt)
                        plain = cipher.decrypt(user_info['password'])

                        if password_from_user is not None:
                            if password_from_user == plain:
                                return JsonResponse({'login': True, 'name': user_info['name'],
                                                     'timestamp': user_info['timestamp'], 'gender': user_info['gender'],
                                                     })
                            else:
                                return JsonResponse({'login': False})
                        else:
                            return JsonResponse({'login': False})
                except Exception as exp:
                    logger.exception(exp)
                    return JsonResponse({'login': False})

    if request.method == 'PUT':
        if option is None:  # Update User Info
            try:
                # Update info from APP
                put = QueryDict(request.body)
                logger.debug("INPUT : %s", put)

                # Encrypting Password
                cipher = security.AESCipher(salt)
                enc = cipher.encrypt(put.get('password'))

                user_info = {'user_id': put.get('user_id'), 'name': put.get('name'), 'gender': put.get('gender'),
                             'timestamp': put.get('timestamp'), 'password': enc}

                # DB Transaction
                user_manager = database.UserManager()
                user_manager.update_user(user_info)
                return JsonResponse({'update_user': True})


            except Exception as exp:
                logger.exception(exp)
                return JsonResponse({'update_user': False})


@csrf_exempt
def manage_diary(request, option=None):
    logger.debug(request)
    if request.method == 'POST':  # create diary

        if option == None:
            try:
                # Diary Info from APP
                data = json.loads(request.POST['json'])  # at POST
                # data = json.loads(request.body.decode('utf-8'))  # at BODY
                logger.debug("INPUT : %s", data)
                if not ('location' in data):
                    data['location'] = ''

                diary_id, c_text_id = insert_new_diary(data, request)

                if diary_id is False:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'create_diary': False})
                else:
                    logger.debug("RETURN : diary_id : %s c_text_id %s ", diary_id,  c_text_id)
                    return JsonResponse({'create_diary': True, 'diary_id': diary_id, 'c_text_id': c_text_id})
            except Exception as exp:
                logger.exception(exp)
                logger.debug("RETURN : FALSE - EXCEPTION")
                return JsonResponse({'create_diary': False})

        if option == 'delete':
            try:
                # input From APP
                # data = json.loads(request.POST['json'])  # at POST
                data = json.loads(request.body.decode('utf-8'))  # at BODY
                logger.debug("INPUT : %s", data)
                diary_id = data['diary_id']
                user_id = data['user_id']

                # Delete Diary
                diary_manager = database.DiaryManager()
                result = diary_manager.delete_diary(diary_id)

                # Delete Diary Attachment Files
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                dest_path = os.path.join(ROOT_DIR, 'uploaded', user_id, str(diary_id))
                if os.path.isdir(dest_path):
                    shutil.rmtree(dest_path)
                if result is False:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'delete_diary': False})
                else:
                    logger.debug("RETURN : TRUE")
                    return JsonResponse({'delete_diary': True})

            except Exception as exp:
                logger.exception(exp)
                logger.debug("RETURN : FALSE - EXCEPTION")
                return JsonResponse({'delete_diary': False})

        if option == 'update':
            try:
                # input From APP
                data = json.loads(request.POST['json'])  # at POST
                # data = json.loads(request.body.decode('utf-8'))  # at BODY
                logger.debug("INPUT : %s", data)
                diary_id = data['diary_id']
                user_id = data['user_id']

                # Delete c_text
                diary_manager = database.DiaryManager()
                diary_manager.delete_c_text_from_diary(diary_id)

                data = {'user_id': data['user_id'], 'diary_id': diary_id, 'title': data['title'],
                        'text': data['text'], 'timestamp': data['timestamp'], 'location': data['location'],
                        'annotation': data['annotation']}

                # Delete Delete Diary Attachment Files
                # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                # dest_path = ROOT_DIR + '/uploaded/' + data['user_id'] + '/' + str(diary_id)
                # if os.path.isdir(dest_path):
                #     shutil.rmtree(dest_path)

                # Insert New c_text, sentence, sent_element and FILES
                diary_id, c_text_id = insert_new_diary(data, request)

                logger.debug("RETURN : c_text_id : %s", c_text_id)
                return JsonResponse({'update_diary': True, 'c_text_id': c_text_id})

            except Exception as exp:
                logger.exception(exp)
                logger.debug("RETURN : FALSE - EXCEPTION")
                return JsonResponse({'update_diary': False})

    elif request.method == 'GET':  # retrieve diary
        try:
            if option is None:
                # Diary Info from APP
                data = json.loads(json.dumps(request.GET))
                logger.debug("INPUT : %s", data)
                # DB Transaction
                converted_manager = database.ConvertedTextManager()
                result = converted_manager.get_converted_text_list(data)
                if result is False or result is None:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'retrieve_diary': False})
                else:
                    logger.debug("RETURN : result : %s", result)
                    return JsonResponse({'retrieve_diary': True, 'result': result})

            if option == 'match':
                data = json.loads(json.dumps(request.GET))
                logger.debug("INPUT :%s", data)
                diary_manager = database.DiaryManager()
                result = diary_manager.retrieve_diary_with_keyword(data)

                if result is None:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'retrieve_diary': False})
                else:
                    logger.debug("RETURN : result : %s", result)
                    return JsonResponse({'retrieve_diary': True, 'result': result})

            if option == 'detail':
                data = json.loads(json.dumps(request.GET))
                logger.debug("INPUT :%s", data)

                diary_manager = database.DiaryManager()
                result = diary_manager.retrieve_diary_detail(data)

                if result is None:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'retrieve_diary': False})

                else:  # result is exist
                    logger.debug("RETURN : result : %s", result)
                    return JsonResponse({'retrieve_diary': True, 'result': result})

        except Exception as exp:
            logger.exception(exp)
            logger.debug("RETURN : FALSE - EXCEPTION")
            return JsonResponse({'retrieve_diary': False})

    elif request.method == 'PUT':
        try:
            # input From APP
            data = json.loads(request.POST['json'])
            logger.debug("INPUT : %s", data)
            diary_id = data.get('diary_id')
            user_id = data.get('user_id')

            # Delete c_text
            diary_manager = database.DiaryManager()
            diary_manager.delete_c_text_from_diary(diary_id)

            data = {'user_id': data.get('user_id'), 'diary_id': diary_id, 'title': data.get('title'),
                    'text': data.get('text'), 'timestamp': data.get('timestamp'), 'location': data.get('location')}

            # Delete Delete Diary Attachment Files
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            dest_path = ROOT_DIR + '/uploaded/' + data['user_id'] + '/' + diary_id
            if os.path.isdir(dest_path):
                shutil.rmtree(dest_path)

            # Insert New c_text, sentence, sent_element and FILES
            diary_id, c_text_id = insert_new_diary(data, request)

            return JsonResponse({'update_diary': True, 'c_text_id': c_text_id})

        except Exception as exp:
            logger.exception(exp)
            return JsonResponse({'update_diary': False})

    elif request.method == 'DELETE':  # delete diary
        try:
            # input From APP
            data = QueryDict(request.body)
            logger.debug("INPUT : %s", data)
            diary_id = data.get('diary_id')
            user_id = data.get('user_id')

            # Delete Diary
            diary_manager = database.DiaryManager()
            diary_manager.delete_diary(diary_id)

            # Delete Diary Attachment Files
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            dest_path = ROOT_DIR + '/uploaded/' + user_id + '/' + diary_id
            if os.path.isdir(dest_path):
                shutil.rmtree(dest_path)

            return JsonResponse({'delete_diary': True})

        except Exception as exp:
            logger.exception(exp)
            return JsonResponse({'delete_diary': False})


@csrf_exempt
def manage_c_text(request, option=None):
    print(request)
    if request.method == 'POST':
        try:
            if option is None:
                # converted text Info from APP
                data = json.loads(request.body.decode('utf-8'))

                # DB Transaction
                c_text_manager = database.ConvertedTextManager()
                c_text_id = c_text_manager.add_converted_text(data)

                if c_text_id is False:
                    return JsonResponse({'add_c_text': False})
                else:
                    return JsonResponse({'add_c_text': True, 'c_text_id': c_text_id})

        except Exception as exp:
            logger.exception(exp)
            return JsonResponse({'add_c_text': False})

    if request.method == 'GET':
        try:
            if option is None:
                # converted text Info from APP
                data = json.loads(json.dumps(request.GET))

                # DB Transaction
                c_text_manager = database.ConvertedTextManager()
                c_text = c_text_manager.get_converted_text(data['c_text_id'])

                if c_text is False or c_text is None:
                    return JsonResponse({'get_c_text': False})
                else:
                    return JsonResponse({'get_c_text': True, 'c_text': c_text})

            elif option == 'list':
                # converted text Info from APP
                data = json.loads(request.body.decode('utf-8'))

                # DB Transaction
                c_text_manager = database.ConvertedTextManager()
                c_text_list = c_text_manager.get_converted_text_list(data)

                if c_text_list is False or c_text_list is None:
                    return JsonResponse({'get_c_text_list': False})
                else:
                    return JsonResponse({'get_c_text_list': True, 'c_text_list': c_text_list})


        except Exception as exp:
            logger.exception(exp)
            return JsonResponse({'get_c_text': False})


# @csrf_exempt
# def manage_sentence(request, option=None):
#     print(request)
#     if request.method == 'POST':
#         try:
#             if option is None:
#                 # parsed sentence from NLP
#                 data = json.loads(request.body.decode('utf-8'))
#
#                 # DB Transaction
#                 sentence_manager = database.SentenceManager()
#                 sentence_id = sentence_manager.add_sentence(data)
#                 if sentence_id is False:
#                     return JsonResponse({'add_sentence': False})
#                 else:
#                     return JsonResponse({'add_sentence': True, 'sentence_id': sentence_id})
#             if option == 'tokenizing':
#                 data = json.loads(request.body.decode('utf-8'))
#                 kor = nlp_ko.SimilarityAnalyzer(user_id=data['user_id'])
#                 sentence_manager = database.SentenceManager
#                 # sentence_manager.add_sentence()
#                 return JsonResponse({'add_sentence': True, 'tokenized': kor.tokenizer(data['text'])})
#
#         except Exception as exp:
#             logger.exception(exp)
#             return JsonResponse({'add_sentence': False})


@csrf_exempt
def manage_nlp_ko_dict(request, option=None):
    logger.debug(request)
    if request.method == 'GET':
        try:
            # word Info from APP
            data = json.loads(json.dumps(request.GET))
            logger.debug("INPUT : %s", data)

            # DB Transaction
            nlp_ko_db = database.NLPkoDictManager()
            result = nlp_ko_db.retrieve_collection_dic(data['word'])

            return JsonResponse({'get_nlp_ko_dict': True, 'data': result})
        except Exception as exp:
            logger.exception(exp)
            return JsonResponse({'get_nlp_ko_dict': False})


@csrf_exempt
def analyze_semantic(request, option=None):
    logger.debug(request)
    if request.method == 'GET':
        if option is None:
            try:
                data = json.loads(json.dumps(request.GET))
                logger.debug("INPUT : %s", data)

                diary_manager = database.DiaryManager()
                diary_info = diary_manager.retrieve_diary(data['diary_id'])
                try:
                    lang = detect(diary_info['text'])
                except Exception as e:  # cho-sung only text Exception Handling
                    lang = 'ko'
                logger.debug("detected lang : %s", lang)

                if lang == 'ko':
                    kor = nlp_ko.SimilarityAnalyzer(user_id='')
                    result = kor.find_sementic(data['diary_id'])
                    if result is None:
                        return JsonResponse({'find_semantic': True, 'result': 'neutrality'})
                    elif result < 0.4:
                        return JsonResponse({'find_semantic': True, 'result': 'bad'})
                    elif result >= 0.6:
                        return JsonResponse({'find_semantic': True, 'result': 'good'})

                elif lang == 'en':
                    en = nlp_en.TextAnalyzer()
                    result = en.analyze_text(diary_info['text'])
                    for idx in result:
                        print(idx)
                    return JsonResponse({'find_semantic': True, 'result': 'good'})




            except Exception as exp:
                logger.exception(exp)
                return JsonResponse({'find_semantic': False})

        if option == 'keyword':
            try:
                data = json.loads(json.dumps(request.GET))
                kor = nlp_ko.SimilarityAnalyzer(user_id=data['user_id'])
                converted_manager = database.ConvertedTextManager()
                diary_manager = database.DiaryManager()

                try:
                    lang = detect(data['keyword'])
                except Exception as e:  # cho-sung only text Exception Handling
                    lang = 'ko'
                logger.debug("detected lang : %s", lang)
                result = {}

                if lang == 'ko':
                    list_k = kor.find_most_similar_docs(query_sentence=data['keyword'])
                    c_text_id_list_k = list_k.values.tolist()
                    result = diary_manager.retrieve_diary_list_wit_c_text_list(c_text_id_list_k)
                    # result = converted_manager.get_converted_text(int(list_k['c_text_id'].iloc[0]))
                elif lang == 'en':
                    eng = nlp_en.SimilarityAnalyzer()
                    list_e = eng.find_most_similar_docs(query_sentence=data['keyword'], user_id=data['user_id'])
                    c_text_id_list_e = list_e.values.tolist()
                    result = diary_manager.retrieve_diary_list_wit_c_text_list(c_text_id_list_e)
                    # result = converted_manager.get_converted_text(int(list_e['c_text_id'].iloc[0]))
                if result is False or result is None:
                    return JsonResponse({'retrieve_diary_by_semantic': False})
                else:
                    return JsonResponse({'retrieve_diary_by_semantic': True, 'result': result})
            except Exception as exp:
                logger.exception(exp)
                return JsonResponse({'retrieve_diary_by_semantic': False})


@csrf_exempt
def find_diary(query_sentence):
    t = Twitter()
    kor = nlp_ko.SimilarityAnalyzer(t)
    similarity_list = kor.find_most_similar_docs(query_sentence=query_sentence,)
    for similarity in similarity_list:
        if similarity['similarity'] != 0:
            print("d_id : ", similarity['c_text_id'])


@csrf_exempt
def download(request):
    logger.debug(request)
    if request.method == 'GET':
        try:
            # USER INFO from APP
            data = json.loads(json.dumps(request.GET))
            logger.debug("INPUT :%s", data)
            if data['user_id'] is not '' and data['diary_id'] is not '':
                file_path = os.path.join(settings.MEDIA_ROOT, data['user_id'], data['diary_id'])
                # zipped_path = os.path.join(file_path, 'audio_files.tar')
                file_list = os.listdir(file_path)
                if os.path.exists(file_path) and file_list:
                    mp3_path = os.path.join(file_path, file_list[0])
                    # write tar on mulitpart
                    with open(mp3_path, 'rb') as fh:
                        response = HttpResponse(fh.read(), content_type="application/force-download")
                        response['GET'] = {"title": 'test', 'text': 'is it works?'}
                        response['Content-Disposition'] = 'inline; filename=' + file_list[0]
                        # plug_cleaning_into_stream(fh, zipped_path)
                        return response
                    # # create tar
                    # with tarfile.open(zipped_path, "w") as tar:
                    #     tar.add(name=file_path, arcname='')
                    #
                    # # write tar on mulitpart
                    # with open(zipped_path, 'rb') as fh:
                    #     response = HttpResponse(fh.read(), content_type="application/force-download")
                    #     response['GET'] = {"title": 'test', 'text': 'is it works?'}
                    #     response['Content-Disposition'] = 'inline; filename=' + 'audio_files.tar'
                    #     plug_cleaning_into_stream(fh, zipped_path)
                    #     return response
                else:
                    return JsonResponse({'download': False})
            else:
                return JsonResponse({'download': False})

        except Exception as exp:
            logger.exception(exp)
            return JsonResponse({'download': False})


# @csrf_exempt
# def manage_file(request):
#     print(request)
#     if request.method == 'POST':
#         data = json.loads(request.body.decode('utf-8'))  # at BODY
#         if request.FILES.items():
#             ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#             dest_path = ROOT_DIR + '/uploaded/' + str(data['user_id']) + '/' + str(data['diary_id'])
#             if not (os.path.isdir(dest_path)):
#                 os.mkdir(dest_path)
#             # request.POST['count']
#             for key, value in request.FILES.items():
#                 file_path = dest_path + '/' + str(request.FILES[key].name)
#                 with open(file_path, 'wb') as destination:
#                     for chunck in request.FILES[key]:
#                         destination.write(chunck)
#         else:
#             print('no file')
#
#         return JsonResponse({'file_upload': True})
#     return JsonResponse({'file_upload': False})


def insert_new_diary(data, request):
    # init DB Modules
    diary_manager = database.DiaryManager()
    sentence_manager = database.SentenceManager()
    s_element_manager = database.SentElementManager()
    # DB Transaction
    if not ('annotation' in data):
        data['annotation'] = ""
    if not ('text' in data) or data['text'] is '' or data['text'] is None:
        return JsonResponse({'create_diary': False})

    if request.FILES.get('file0', False):  # file checking
        data['audio'] = True
    else:
        data['audio'] = False

    # INSERT Text INTO DB
    if not ('diary_id' in data):
        diary_id = diary_manager.create_diary(data)
        data['diary_id'] = diary_id
    else:
        diary_manager.update_diary(data)
        diary_id = data['diary_id']

    c_text_manager = database.ConvertedTextManager()
    c_text_id = c_text_manager.add_converted_text(data)



    # FILE UPLOAD LOGIC-------------------------------------------------------------------------------------------------
    # Audio File Uploading
    if data['audio']:
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        DIARY_DIR = os.path.join(ROOT_DIR, 'uploaded', str(data['user_id']), str(diary_id))
        if not (os.path.isdir(DIARY_DIR)):
            os.mkdir(DIARY_DIR)

        for key, value in request.FILES.items():
            # Saving UPLOADED Files
            file_path = os.path.join(DIARY_DIR, str(request.FILES[key].name))
            with open(file_path, 'wb') as destination:
                for chunck in request.FILES[key]:
                    destination.write(chunck)

            # Converting
            command = 'ffmpeg -i \"' + file_path + '\" -ar 22050 ' + '\"' + file_path.replace('m4a', '') + 'mp3\"'
            subprocess.call(shlex.split(command), shell=True)

            # Deleting .m4a(AMR)
            os.remove(file_path)

        # Merging
        file_list = os.listdir(DIARY_DIR)
        merge_list = []

        for file in file_list:
            song = AudioSegment.from_file(os.path.join(DIARY_DIR, file), 'mp3')
            merge_list.append(song)

        merged = AudioSegment.empty()
        for file in merge_list:
            merged = merged + file

        # Saving Merged File
        merged.export(os.path.join(DIARY_DIR, str(diary_id) + '.mp3'), format('mp3'))

        # Deleting mp3 files AFTER Merging
        for file in file_list:
            os.remove(os.path.join(DIARY_DIR, file))
        # FILE UPLOAD LOGIC END-----------------------------------------------------------------------------------------



    # Parse INTO sentence, sent element LOGIC---------------------------------------------------------------------------
    if data['text'] is '':
        pass
    else:
        try:
            lang = detect(data['text'])
        except Exception as e:  # cho-sung only text Exception Handling
            lang = 'ko'

        if lang == 'ko':  # When sentence written in KOREAN
            k = Twitter()
            kor = nlp_ko.SimilarityAnalyzer(k)

            # Parse into Sentence
            sentence_info = {'c_text_id': c_text_id, 'text': kor.slice_sentence(data['text'])}
            sentence_id = sentence_manager.add_sentence(sentence_info)

            # Parse into Sent Element
            s_element_list = kor.tokenize(data['text'])
            s_element_json_list = []
            if s_element_list:
                for s_element_line in s_element_list:
                    element_no = 0
                    for s_element_tuple in s_element_line:
                        s_element_json = {'sentence_id': sentence_id, 'text': s_element_tuple[0],
                                          'pos': s_element_tuple[1],
                                          'role': '', 'element_no': element_no,
                                          'ne': ''
                                          }
                        element_no += 1
                        s_element_json_list.append(s_element_json)
                    sentence_id += 1
                s_element_manager.add_s_element(s_element_json_list)

        elif lang == 'en':  # Parse to Sentence & Save in DB
            sentence_info = {'c_text_id': c_text_id, 'text': nlp_en.sent_tokenizer(data['text'])}
            sentence_id = sentence_manager.add_sentence(sentence_info)
            s_element_json_list = []
            for sentence in sentence_info['text']:
                # Parse into Sent Element
                s_element_line = nlp_en.pos_tagger(sentence)
                element_no = 0
                for s_element_tuple in s_element_line:
                    s_element_json = {'sentence_id': sentence_id, 'text': s_element_tuple[0], 'pos': s_element_tuple[1],
                                      'role': '', 'element_no': element_no}  # must adding ne
                    element_no += 1
                    s_element_json_list.append(s_element_json)
                sentence_id += 1
            s_element_manager.add_s_element(s_element_json_list)

    return diary_id, c_text_id


def plug_cleaning_into_stream(stream, filename):  # for deleting tar
    try:
        closer = getattr(stream, 'close')
        # define a new function that still uses the old one

        def new_closer():
            closer()
            os.remove(filename)
            # any cleaning you need added as well
        # substitute it to the old close() function
        setattr(stream, 'close', new_closer)
    except:
        raise
