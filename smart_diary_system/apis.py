import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from smart_diary_system import database
from smart_diary_system import security
import logging
from django.http import Http404

# from diary_nlp import nlp_en
from langdetect import detect
import os
import shutil

from django.http import QueryDict
from django.conf import settings
from django.http import HttpResponse


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
                    if not ('birthday' in data):
                        data['birthday'] = 0
                    if not ('email' in data):
                        data['email'] = ''
                    if not ('phone' in data):
                        data['phone'] = ''
                    user_manager = database.UserManager()
                    # Encrypting Password
                    cipher = security.AESCipher(salt)
                    enc = cipher.encrypt(data.get('password'))
                    # DB Transaction
                    result = user_manager.create_user(user_id=data.get('user_id'), password=enc, name=data.get('name'),
                                                   birthday=data.get('birthday'), gender=data.get('gender'),
                                                   email=data.get('email'), phone=data.get('phone'))


                    if result:
                        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                        os.mkdir(os.path.join(ROOT_DIR, 'uploaded', str(data.get('user_id'))))
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
                                                     'birthday': user_info['birthday'], 'gender': user_info['gender'],
                                                     'email': user_info['email'], 'phone': user_info['phone']
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
                put = json.loads(request.body.decode('utf-8'))
                print(put)
                logger.debug("INPUT : %s", put)

                # Encrypting Password
                cipher = security.AESCipher(salt)
                enc = cipher.encrypt(put.get('password'))

                # DB Transaction
                user_manager = database.UserManager()
                user_manager.update_user(user_id=put.get('user_id'), name=put.get('name'), gender=put.get('gender'),
                                         birthday=put.get('birthday'), password=enc, email=put.get('email'),
                                         phone=put.get('phone'))
                return JsonResponse({'update_user': True})

            except Exception as exp:
                logger.exception(exp)
                return JsonResponse({'update_user': False})

    if request.method == 'DELETE':
        if option is None:
            try:
                delete = json.loads(request.body.decode('utf-8'))
                logger.debug("INPUT : %s", delete)

                user_manager = database.UserManager()
                user_id = delete.get('user_id')
                result = user_manager.delete_user(user_id=user_id)

                # delete Files
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                shutil.rmtree(os.path.join(ROOT_DIR, 'uploaded', user_id))

                if result:
                    return JsonResponse({'delete_user': True})
                else:
                    return JsonResponse({'delete_user': False})

            except Exception as exp:
                logger.exception(exp)
                return JsonResponse({'delete_user': False})


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

                audio_diary_id, text_diary_id = insert_new_diary(data, request)

                if audio_diary_id is False:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'create_diary': False})
                else:
                    logger.debug("RETURN : audio_diary_id : %s text_diary_id %s ", audio_diary_id,  text_diary_id)
                    return JsonResponse({'create_diary': True, 'audio_diary_id': audio_diary_id, 'text_diary_id': text_diary_id})
            except Exception as exp:
                logger.exception(exp)
                logger.debug("RETURN : FALSE - EXCEPTION")
                return JsonResponse({'create_diary': False})

    elif request.method == 'GET':  # retrieve diary
        try:
            if option is None:
                # Diary Info from APP
                data = json.loads(json.dumps(request.GET))
                logger.debug("INPUT : %s", data)
                # DB Transaction
                text_diary_maanger = database.TextDiaryManager()
                result = text_diary_maanger.retrieve_text_diary_list(data)
                if result is False:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'retrieve_diary': False})
                else:
                    logger.debug("RETURN : result : %s", result)
                    return JsonResponse({'retrieve_diary': True, 'result': result})

            if option == 'match':
                data = json.loads(json.dumps(request.GET))
                logger.debug("INPUT :%s", data)
                audio_diary_manager = database.AudioDiaryManager()
                result = audio_diary_manager.retri(data)

                if result is None:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'retrieve_diary': False})
                else:
                    logger.debug("RETURN : result : %s", result)
                    return JsonResponse({'retrieve_diary': True, 'result': result})

            if option == 'detail':
                data = json.loads(json.dumps(request.GET))
                logger.debug("INPUT :%s", data)

                audio_diary_manager = database.AudioDiaryManager()
                diary_context_manager = database.DiaryContextManager()
                result = audio_diary_manager.retrieve_audio_diary_detail_by_audio_diary_id(data['user_id'], data['audio_diary_id'])

                if result is False:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'retrieve_diary': False})
                else:  # result is exist
                    logger.debug("RETURN : result : %s", result)
                    if result is not None:
                        result_context = diary_context_manager.retrieve_diary_context_by_audio_diary_id(data['audio_diary_id'])
                    else:
                        result_context = []
                    return JsonResponse({'retrieve_diary': True, 'result_detail': result, 'result_context': result_context})

        except Exception as exp:
            logger.exception(exp)
            logger.debug("RETURN : FALSE - EXCEPTION")
            return JsonResponse({'retrieve_diary': False})

    elif request.method == 'PUT':
        try:
            if option is None:
                # input From APP
                data = json.loads(request.body.decode('utf-8'))  # at BODY
                logger.debug("INPUT : %s", data)
                audio_diary_id = data.get('audio_diary_id')
                user_id = data.get('user_id')

                # Delete text_diary
                text_diary_manger = database.TextDiaryManager()
                text_diary_manger.delete_text_diary_by_audio_diary_id(audio_diary_id)

                # Delete Delete Diary Attachment Files
                # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                # dest_path = ROOT_DIR + '/uploaded/' + data['user_id'] + '/' + audio_diary_id
                # if os.path.isdir(dest_path):
                #     shutil.rmtree(dest_path)

                # Insert New text_diary, sentence, sent_element
                audio_diary_id, text_diary_id = insert_new_diary(data, request)

                return JsonResponse({'update_diary': True, 'text_diary_id': text_diary_id})

        except Exception as exp:
            logger.exception(exp)
            return JsonResponse({'update_diary': False})

    elif request.method == 'DELETE':  # delete diary
        try:
            # input From APP
            data = json.loads(request.body.decode('utf-8'))
            logger.debug("INPUT : %s", data)
            audio_diary_id = data.get('audio_diary_id')
            user_id = data.get('user_id')

            # Delete Diary
            audio_diary_manager = database.AudioDiaryManager()
            audio_diary_manager.delete_audio_diary(audio_diary_id)

            # Delete Diary Attachment Files
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            dest_path = ROOT_DIR + '/uploaded/' + user_id + '/' + audio_diary_id
            if os.path.isdir(dest_path):
                shutil.rmtree(dest_path)

            return JsonResponse({'delete_diary': True})

        except Exception as exp:
            logger.exception(exp)
            return JsonResponse({'delete_diary': False})


# @csrf_exempt
# def analyze_semantic(request, option=None):
#     logger.debug(request)
#     if request.method == 'GET':
#         if option is None:
#             try:
#                 data = json.loads(json.dumps(request.GET))
#                 logger.debug("INPUT : %s", data)
#
#                 diary_manager = database.DiaryManager()
#                 diary_info = diary_manager.retrieve_diary(data['audio_diary_id'])
#                 try:
#                     lang = detect(diary_info['content'])
#                 except Exception as e:  # cho-sung only text Exception Handling
#                     lang = 'ko'
#                 logger.debug("detected lang : %s", lang)
#
#                 if lang == 'en':
#                     en = nlp_en.TextAnalyzer()
#                     result = en.analyze_text(diary_info['content'])
#                     for idx in result:
#                         print(idx)
#                     return JsonResponse({'find_semantic': True, 'result': 'good'})
#
#                 # elif lang == 'ko':
#                 #     kor = nlp_ko.SimilarityAnalyzer(user_id='')
#                 #     result = kor.find_sementic(data['audio_diary_id'])
#                 #     if result is None:
#                 #         return JsonResponse({'find_semantic': True, 'result': 'neutrality'})
#                 #     elif result < 0.4:
#                 #         return JsonResponse({'find_semantic': True, 'result': 'bad'})
#                 #     elif result >= 0.6:
#                 #         return JsonResponse({'find_semantic': True, 'result': 'good'})
#                 else:
#                     logger.debug('LANG ERROR')
#
#
#
#
#             except Exception as exp:
#                 logger.exception(exp)
#                 return JsonResponse({'find_semantic': False})
#
#         if option == 'keyword':
#             try:
#                 data = json.loads(json.dumps(request.GET))
#
#                 converted_manager = database.ConvertedTextManager()
#                 diary_manager = database.DiaryManager()
#
#                 try:
#                     lang = detect(data['keyword'])
#                 except Exception as e:  # cho-sung only text Exception Handling
#                     lang = 'ko'
#                 logger.debug("detected lang : %s", lang)
#                 result = {}
#
#                 if lang == 'en':
#                     eng = nlp_en.SimilarityAnalyzer()
#                     list_e = eng.find_most_similar_docs(query_sentence=data['keyword'], user_id=data['user_id'])
#                     text_diary_id_list_e = list_e.values.tolist()
#                     result = diary_manager.retrieve_diary_list_wit_c_text_list(text_diary_id_list_e)
#                     # result = converted_manager.get_converted_text(int(list_e['text_diary_id'].iloc[0]))
#                 # elif lang == 'ko':
#                 #     kor = nlp_ko.SimilarityAnalyzer(user_id=data['user_id'])
#                 #     list_k = kor.find_most_similar_docs(query_sentence=data['keyword'])
#                 #     text_diary_id_list_k = list_k.values.tolist()
#                 #     result = diary_manager.retrieve_diary_list_wit_c_text_list(text_diary_id_list_k)
#                 #     # result = converted_manager.get_converted_text(int(list_k['text_diary_id'].iloc[0]))
#                 else:
#                     logger.debug('LANG ERROR')
#                 if result is False or result is None:
#                     return JsonResponse({'retrieve_diary_by_semantic': False})
#                 else:
#                     return JsonResponse({'retrieve_diary_by_semantic': True, 'result': result})
#             except Exception as exp:
#                 logger.exception(exp)
#                 return JsonResponse({'retrieve_diary_by_semantic': False})


@csrf_exempt
def download(request):
    logger.debug(request)
    if request.method == 'GET':
        try:
            # USER INFO from APP
            data = json.loads(json.dumps(request.GET))
            logger.debug("INPUT :%s", data)
            if data['user_id'] is not '' and data['audio_diary_id'] is not '':
                file_path = os.path.join(settings.MEDIA_ROOT, data['user_id'], data['audio_diary_id'])
                file_list = os.listdir(file_path)
                if os.path.exists(file_path) and file_list:
                    audio_file_path = os.path.join(file_path, file_list[0])
                    with open(audio_file_path, 'rb') as fh:
                        response = HttpResponse(fh.read(), content_type="audio/wav")
                        response['Content-Disposition'] = 'inline; filename=' + file_list[0]
                        return response
                else:
                    raise Http404
            else:
                raise Http404

        except Exception as exp:
            logger.exception(exp)
            raise Http404

    if request.method == 'POST':
        try:
            # USER INFO from APP
            data = json.loads(request.body.decode('utf-8'))
            logger.debug("INPUT :%s", data)
            if data['user_id'] is not '' and data['audio_diary_id'] is not '':
                file_path = os.path.join(settings.MEDIA_ROOT, data['user_id'], str(data['audio_diary_id']))
                file_list = os.listdir(file_path)
                if os.path.exists(file_path) and file_list:
                    audio_file_path = os.path.join(file_path, file_list[0])
                    with open(audio_file_path, 'rb') as fh:
                        response = HttpResponse(fh.read(), content_type="audio/wav")
                        response['Content-Disposition'] = 'inline; filename=' + file_list[0]
                        return response
                else:
                    raise Http404
            else:
                raise Http404

        except Exception as exp:
            logger.exception(exp)
            raise Http404


def insert_new_diary(data, request):
    # init DB Modules
    audio_diary_manager = database.AudioDiaryManager()
    sentence_manager = database.SentenceManager()
    s_element_manager = database.SentenceElementManager()
    diary_context_manager = database.DiaryContextManager()

    # DB Transaction
    # KerError Exception Handling
    if not ('content' in data) or data['content'] is '' or data['content'] is None:
        return JsonResponse({'create_diary': False})

    # INSERT Text INTO DB
    if not ('audio_diary_id' in data):  # NEW NOTE
        audio_diary_id = audio_diary_manager.create_audio_diary(data['user_id'], data['title'], data['created_date'])
        data['audio_diary_id'] = audio_diary_id
    else:  # EDIT NOTE
        audio_diary_manager.update_audio_diary(data)
        audio_diary_id = data['audio_diary_id']
        diary_context_manager.delete_diary_context_by_audio_diary_id(audio_diary_id)

    text_diary_manager = database.TextDiaryManager()
    text_diary_id = text_diary_manager.create_text_diary(data['audio_diary_id'], data['content'], data['created_date'])

    if 'diary_context' in data and data['diary_context'] is not None:
        diary_context_manager.create_diary_context(audio_diary_id=data['audio_diary_id'], diary_context_info=data['diary_context'])

    # FILE UPLOAD LOGIC-------------------------------------------------------------------------------------------------
    # Audio File Uploading
    if request.FILES.get('file0', False):  # file checking
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        DIARY_DIR = os.path.join(ROOT_DIR, 'uploaded', str(data['user_id']), str(audio_diary_id))
        if not (os.path.isdir(DIARY_DIR)):
            os.mkdir(DIARY_DIR)

        for key, value in request.FILES.items():
            # Saving UPLOADED Files
            file_path = os.path.join(DIARY_DIR, str(request.FILES[key].name))
            with open(file_path, 'wb') as destination:
                for chunck in request.FILES[key]:
                    destination.write(chunck)

    # FILE UPLOAD LOGIC END-----------------------------------------------------------------------------------------



    # Parse INTO sentence, sent element LOGIC---------------------------------------------------------------------------
    # if data['content'] is '':
    #     pass
    # else:
    #     try:
    #         lang = detect(data['content'])
    #     except Exception as e:  # cho-sung only text Exception Handling
    #         lang = 'ko'
    #
    #     if lang == 'en':  # Parse to Sentence & Save in DB
    #         sentence_info = {'text_diary_id': text_diary_id, 'content': nlp_en.sent_tokenizer(data['content'])}
    #         sentence_id = sentence_manager.add_sentence(sentence_info)
    #         s_element_json_list = []
    #         for sentence in sentence_info['content']:
    #             # Parse into Sent Element
    #             s_element_line = nlp_en.pos_tagger(sentence)
    #             element_no = 0
    #             for s_element_tuple in s_element_line:
    #                 s_element_json = {'sentence_id': sentence_id, 'content': s_element_tuple[0], 'pos': s_element_tuple[1],
    #                                   'role': '', 'element_no': element_no}  # must adding ne
    #                 element_no += 1
    #                 s_element_json_list.append(s_element_json)
    #             sentence_id += 1
    #         s_element_manager.add_s_element(s_element_json_list)

        # elif lang == 'ko':  # When sentence written in KOREAN
        #     k = Twitter()
        #     kor = nlp_ko.SimilarityAnalyzer(k)
        #
        #     # Parse into Sentence
        #     sentence_info = {'text_diary_id': text_diary_id, 'content': kor.slice_sentence(data['content'])}
        #     sentence_id = sentence_manager.add_sentence(sentence_info)
        #
        #     # Parse into Sent Element
        #     s_element_list = kor.tokenize(data['content'])
        #     s_element_json_list = []
        #     if s_element_list:
        #         for s_element_line in s_element_list:
        #             element_no = 0
        #             for s_element_tuple in s_element_line:
        #                 s_element_json = {'sentence_id': sentence_id, 'content': s_element_tuple[0],
        #                                   'pos': s_element_tuple[1],
        #                                   'role': '', 'element_no': element_no,
        #                                   'ne': ''
        #                                   }
        #                 element_no += 1
        #                 s_element_json_list.append(s_element_json)
        #             sentence_id += 1
        #         s_element_manager.add_s_element(s_element_json_list)
        # else:
        #     logger.debug('LANG ERROR')

    return audio_diary_id, text_diary_id

