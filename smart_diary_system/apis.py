import json
import timeit
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from smart_diary_system import database
from smart_diary_system import security
import logging
from django.http import Http404
from diary_analyzer import tagger
from diary_analyzer import lifestyles
import operator

# from diary_nlp import nlp_en
from langdetect import detect
import os
import shutil
import threading

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
                        os.mkdir(os.path.join(ROOT_DIR, 'pickles', str(data.get('user_id'))))
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
                                tag_manager = database.TagManager()
                                tag_list = tag_manager.retrieve_tag_by_user_id(user_info['user_id'])
                                return JsonResponse({'login': True, 'name': user_info['name'],
                                                     'birthday': user_info['birthday'], 'gender': user_info['gender'],
                                                     'email': user_info['email'], 'phone': user_info['phone'], 'tag_list': tag_list
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
                shutil.rmtree(os.path.join(ROOT_DIR, 'pickles', user_id))

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

                audio_diary_id, text_diary_id, mc_id_list = insert_new_diary(data, request)

                if audio_diary_id is False:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'create_diary': False})
                else:
                    logger.debug("RETURN : audio_diary_id : %s text_diary_id %s mc_id_list %s", audio_diary_id,  text_diary_id, mc_id_list)
                    return JsonResponse({'create_diary': True, 'audio_diary_id': audio_diary_id, 'text_diary_id': text_diary_id, 'media_context_id_list': mc_id_list})
            except Exception as exp:
                logger.exception(exp)
                logger.debug("RETURN : FALSE - EXCEPTION")
                return JsonResponse({'create_diary': False})

        if option == 'delete':
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
                # Delete Diary Attachment Files
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                audio_path = os.path.join(ROOT_DIR, 'uploaded', user_id, str(audio_diary_id))
                pickle_path = os.path.join(ROOT_DIR, 'pickles', user_id, str(audio_diary_id))
                if os.path.isdir(audio_path):
                    shutil.rmtree(audio_path)
                if os.path.isdir(pickle_path):
                    shutil.rmtree(pickle_path)

                return JsonResponse({'delete_diary': True})

            except Exception as exp:
                logger.exception(exp)
                return JsonResponse({'delete_diary': False})

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

            if option == 'search':
                data = json.loads(json.dumps(request.GET))
                logger.debug("INPUT :%s", data)
                audio_diary_manager = database.AudioDiaryManager()
                result = audio_diary_manager.retrieve_text_diary_list_by_keyword(data)

                if result is None:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'retrieve_diary': False})
                else:
                    logger.debug("RETURN : result : %s", result)
                    return JsonResponse({'retrieve_diary': True, 'result': result})

            if option == 'tag':
                data = json.loads(json.dumps(request.GET))
                logger.debug("INPUT :%s", data)
                tag_manager = database.TagManager()
                result = tag_manager.retrieve_tag_by_keyword(data['user_id'], data['tag'])

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
                e_context_manager = database.EnvironmentalContextManager()
                tag_manager = database.TagManager()
                result = audio_diary_manager.retrieve_audio_diary_detail_by_audio_diary_id(data['user_id'], data['audio_diary_id'])
                tag_result = tag_manager.retrieve_tag_by_audio_diary_id(data['audio_diary_id'])
                if result is False:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'retrieve_diary': False})
                else:  # result is exist
                    logger.debug("RETURN : result : %s", result)
                    if result is not None:
                        result_context = e_context_manager.retrieve_environmental_context_by_audio_diary_id(data['audio_diary_id'])
                    else:
                        result_context = []
                    return JsonResponse({'retrieve_diary': True, 'result_detail': result, 'result_environmental_context': result_context,
                                         'result_tag_list': tag_result})

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

    # elif request.method == 'DELETE':  # delete diary
    #     try:
    #         # input From APP
    #         data = json.loads(request.body.decode('utf-8'))
    #         logger.debug("INPUT : %s", data)
    #         audio_diary_id = data.get('audio_diary_id')
    #         user_id = data.get('user_id')
    #
    #         # Delete Diary
    #         audio_diary_manager = database.AudioDiaryManager()
    #         audio_diary_manager.delete_audio_diary(audio_diary_id)
    #
    #         # Delete Diary Attachment Files
    #         ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    #         audio_path = os.path.join(ROOT_DIR, 'uploaded', user_id, str(audio_diary_id))
    #         pickle_path = os.path.join(ROOT_DIR, 'pickles', user_id, str(audio_diary_id))
    #         if os.path.isdir(audio_path):
    #             shutil.rmtree(audio_path)
    #         if os.path.isdir(pickle_path):
    #             shutil.rmtree(pickle_path)
    #
    #         return JsonResponse({'delete_diary': True})
    #
    #     except Exception as exp:
    #         logger.exception(exp)
    #         return JsonResponse({'delete_diary': False})


@csrf_exempt
def manage_analyze(request, option=None):
    logger.debug(request)
    if request.method == 'POST':  # create diary
        if option == None:
            try:
                pass

            except Exception as exp:
                logger.exception(exp)
                logger.debug("RETURN : FALSE - EXCEPTION")
                return JsonResponse({'create_diary': False})

        if option == 'delete':
            try:
                pass

            except Exception as exp:
                logger.exception(exp)
                return JsonResponse({'delete_diary': False})

    elif request.method == 'GET':

            if option is None:
                pass

            if option == 'check_pos_tagging':
                try:
                    data = json.loads(json.dumps(request.GET))
                    logger.debug("INPUT :%s", data)

                    audio_diary_manager = database.AudioDiaryManager()
                    audio_diary_list = audio_diary_manager.retrieve_state_flags(data['audio_diary_id'])

                    if audio_diary_list is None:
                        return JsonResponse({'check_pos_tagging': False, 'reason': 'AUDIO_DIARY_NOT_EXIST'})
                    else:
                        state = audio_diary_list['pickle']
                        if state == 0:
                            return JsonResponse({'check_pos_tagging': False, 'reason': 'MAKING_FAILED'})
                        elif state == 1:
                            return JsonResponse({'check_pos_tagging': False, 'reason': 'WORKING'})
                        else:
                            return JsonResponse({'check_pos_tagging': True})

                except Exception as exp:
                    logger.exception(exp)
                    logger.debug("RETURN : FALSE - EXCEPTION")
                    return JsonResponse({'parsing_is_done': False})

            if option == 'lifestyle':
                try:
                    data = json.loads(json.dumps(request.GET))
                    logger.debug("INPUT :%s", data)

                    thing_type = data['thing_type']
                    defined_thing_type = ['food', 'sport']
                    if not (thing_type in defined_thing_type):
                        # Can not Anaylzed, THING TYPE is wrong, which is not defined
                        logger.debug("RETURN : FALSE - INVALID THING TYPE")
                        return JsonResponse({'lifestyle': False, 'reason': 'INVALID THING TYPE'})

                    # init DB Mangagers
                    audio_diary_manager = database.AudioDiaryManager()
                    life_style_manager = database.LifeStyleManager()

                    # retrieve audio diaries which will be analyzed
                    audio_diary_list = audio_diary_manager.retrieve_audio_diary_list_by_timestamp(data)  # user_id, timestamp_from, timestamp_to
                    if audio_diary_list is None:
                        # Nothing to show
                        logger.debug("RETURN : TRUE - NO DIARY AVAILABLE")
                        return JsonResponse({'lifestyle': True, 'result': []})
                    else:
                        diary_tag_list = []
                        analyzed_audio_diary_id_list = []
                        ranking_audio_diary_id_list = []
                        for audio_diary in audio_diary_list:  # load pickles
                            ranking_audio_diary_id_list.append(audio_diary['audio_diary_id'])

                            # check if required thing_type already analyzed
                            analyzed_types = parse_lifetype(audio_diary['lifestyle_analyzed'])
                            if thing_type in analyzed_types:
                                logger.debug('lifestyle : id(%s) already Analyzed', audio_diary['audio_diary_id'])
                            else:
                                logger.debug('lifestyle : id(%s) Will be Analyzed FOR %s',
                                             audio_diary['audio_diary_id'], thing_type)
                                # load pickles
                                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                                PICKLE_DIR = os.path.join(ROOT_DIR, 'pickles', audio_diary['user_id'],
                                                          str(audio_diary['audio_diary_id']), 'pos_texts.pkl')
                                diary_tag_list.append(tagger.pickle_to_tags(PICKLE_DIR))

                                if audio_diary['lifestyle_analyzed'] is None:
                                    lifestyle_analyzed = thing_type + ','
                                else:
                                    lifestyle_analyzed = thing_type + ',' + audio_diary['lifestyle_analyzed']
                                temp = {'audio_diary_id': audio_diary['audio_diary_id'], 'lifestyle_analyzed': lifestyle_analyzed}
                                analyzed_audio_diary_id_list.append(temp)

                        if not diary_tag_list and not analyzed_audio_diary_id_list:
                            logger.debug('lifestyle : NOTHING TO INSERT INTO DB')

                        # making lifestyle DB record
                        lifestyles_dict_list = []
                        for audio_diary_id, diary_tags in zip(analyzed_audio_diary_id_list, diary_tag_list):
                            # analyze lifestyle
                            if thing_type == 'food':
                                lifestyle_analyze_result = lifestyles.analyzer.analyze_food(diary_tags[1])
                            elif thing_type == 'sport':
                                # lifestyle_analyze_result = lifestyles.analyzer.sport_collect(diary_tags[1])
                                return JsonResponse({'lifestyle': False, 'reason': 'NOT YET IMPLEMENTED'})

                            for lifestyle_item in lifestyle_analyze_result.keys():
                                lifestyles_dict = {'audio_diary_id': audio_diary_id['audio_diary_id'], 'thing_type': thing_type,
                                                   'thing': lifestyle_item, 'score': lifestyle_analyze_result[lifestyle_item]}
                                lifestyles_dict_list.append(lifestyles_dict)

                        if lifestyles_dict_list:  # insert lifestyle record into DB
                            life_style_manager.create_lifestyle_by_list(lifestyles_dict_list)

                        # statistic analyze
                        lifestyle_item_list = life_style_manager.retrieve_lifestyle(ranking_audio_diary_id_list, thing_type)
                        if lifestyle_item_list:
                            if str(data['option']).lower() == 'like':
                                like = True
                            elif str(data['option']).lower() == 'dislike':
                                like = False
                            else:
                                logger.debug("RETURN : FALSE - INVALID OPTION TYPE")
                                return JsonResponse({'lifestyle': False, 'reason': 'INVALID OPTION TYPE'})
                            final_result = lifestyles.ranking_lifestyle(lifestyle_list=lifestyle_item_list, like=like)
                            logger.debug("RETURN : %s", final_result)

                            # updating lifestyle_analyzed flag in audio_diary table
                            audio_diary_manager.update_lifestyle_analyzed_state(analyzed_audio_diary_id_list)
                            return JsonResponse({'lifestyle': True, 'result': final_result})
                        else:
                            return JsonResponse({'lifestyle': True, 'result': []})


                except Exception as exp:
                    logger.exception(exp)
                    logger.debug("RETURN : FALSE - EXCEPTION")
                    return JsonResponse({'lifestyle': False, 'reason': 'INTERNAL SERVER ERROR'})

    elif request.method == 'PUT':
        try:
            if option is None:
                pass

        except Exception as exp:
            logger.exception(exp)
            return JsonResponse({'update_diary': False})

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
                    if not('media_context_id' in data):
                        for file in file_list:
                            if '.wav' in file:
                                file_path = os.path.join(file_path, file)
                                with open(file_path, 'rb') as fh:
                                    response = HttpResponse(fh.read(), content_type="audio/wav")
                                    response['Content-Disposition'] = 'inline; filename=' + file
                                    return response
                            else:
                                raise Http404
                    else:
                        mc_manager = database.MediaContextManager()
                        mc_info = mc_manager.retrieve_media_context_by_mc_id(data['media_context_id'])

                        if mc_info['type'] == 'picture':
                            with open(mc_info['path'], 'rb') as fh:
                                response = HttpResponse(fh.read(), content_type="image")
                                response['Content-Disposition'] = 'inline; filename=' + str(os.path.basename(mc_info['path']))
                                return response
                        elif mc_info['type'] == 'video':
                            with open(mc_info['path'], 'rb') as fh:
                                response = HttpResponse(fh.read(), content_type="video")
                                response['Content-Disposition'] = 'inline; filename=' + str(
                                    os.path.basename(mc_info['path']))
                                return response
                        elif mc_info['type'] == 'music':
                            with open(mc_info['path'], 'rb') as fh:
                                response = HttpResponse(fh.read(), content_type="audio")
                                response['Content-Disposition'] = 'inline; filename=' + str(
                                    os.path.basename(mc_info['path']))
                                return response
                        elif data['type'] == 'handdrawn':
                            pass
                        else:
                            raise Http404
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
                    file_path = os.path.join(file_path, file_list[0])
                    with open(file_path, 'rb') as fh:
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


@csrf_exempt
def uploading_test(request):
    if request.FILES.get('file0', False):  # file checking
        audio_diary_id = 400
        mc_id_list = []
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        DIARY_DIR = os.path.join(ROOT_DIR, 'uploaded', 'lhs', str(audio_diary_id))
        MEDIA_DIR = os.path.join(DIARY_DIR, 'media')
        if not (os.path.isdir(DIARY_DIR)):
            os.mkdir(DIARY_DIR)
        if not (os.path.isdir(MEDIA_DIR)):
            os.mkdir(MEDIA_DIR)

        picture_extension_list = ['.ani', '.bmp', '.cal', '.fax', '.gif', '.img', '.jbg', '.jpe', '.jpeg', '.jpg',
                                  '.mac', '.pbm', '.pcd', '.pcx', '.pct', '.pgm', '.png', '.ppm', '.psd', '.ras',
                                  '.tga', '.tiff', '.wmf']
        music_extension_list = ['.3gp', '.aa', '.aac', '.aax', '.act', '.aiff', '.amr', '.ape', '.au', '.awb', '.dct',
                                '.dss', '.dvf', '.flac', '.gsm', '.iklax', '.ivs', '.m4a', '.m4b', '.m4p', '.mmf',
                                '.mp3', '.mpc', '.msv', '.ogg', '.oga', 'mogg', '.opus', '.ra', '.rm', '.raw', '.sln',
                                '.tta', '.vox', '.wav', '.wma', '.wv', '.webm']
        video_extension_list = ['.webm', '.mkv', '.flv', '.flv', '.vob', '.ogv', '.ogg', '.gif', '.gifv', '.mng',
                                '.avi', '.mov', '.qt', '.wmv', '.yuv', '.rm', '.rmvb', '.asf', '.amv', '.mp4', '.m4p',
                                '.m4v', '.mpg', '.mp2', '.mpeg', '.mpe', '.mpv', '.mpg', '.mpeg', '.m2v', '.m4v',
                                '.svi', '.3gp', '.3g2', '.flv', '.f4v', '.f4p', '.f4a', '.f4b']

        mc_manager = database.MediaContextManager()
        for key, value in request.FILES.items():
            # Saving UPLOADED Files
            if key == 'file0':
                file_path = os.path.join(DIARY_DIR, str(request.FILES[key].name))
                with open(file_path, 'wb') as destination:
                    for chunck in request.FILES[key]:
                        destination.write(chunck)
            else:
                file_name = str(request.FILES[key].name)
                file_path = os.path.join(MEDIA_DIR, file_name)
                mc_data = {'audio_diary_id': audio_diary_id, 'path': file_path}
                if any(tp in file_name for tp in picture_extension_list):
                    mc_data['type'] = 'picture'

                elif any(tp in file_name for tp in music_extension_list):
                    mc_data['type'] = 'music'

                elif any(tp in file_name for tp in video_extension_list):
                    mc_data['type'] = 'video'

                mc_id = mc_manager.create_media_context(audio_diary_id, mc_data)
                tmp = {'file_name': file_name, 'media_context_id': mc_id}
                mc_id_list.append(tmp)

                with open(file_path, 'wb') as destination:
                    for chunck in request.FILES[key]:
                        destination.write(chunck)

        return JsonResponse({'result': True})
    else:
        return JsonResponse({'result': False})


def insert_new_diary(data, request):
    # init DB Modules
    audio_diary_manager = database.AudioDiaryManager()
    sentence_manager = database.SentenceManager()
    e_context_manager = database.EnvironmentalContextManager()
    tag_manager = database.TagManager()
    mc_id_list = []

    # DB Transaction
    # KerError Exception Handling
    if not ('content' in data) or data['content'] is '' or data['content'] is None:
        return JsonResponse({'create_diary': False})

    # INSERT Text INTO DB
    if not ('audio_diary_id' in data):  # NEW NOTE
        audio_diary_id = audio_diary_manager.create_audio_diary(data['user_id'], data['title'], data['created_date'])
        data['audio_diary_id'] = audio_diary_id
        if not('tag' in data):
            data['tag'] = []
        tag_manager.create_tag(audio_diary_id, data['tag'])
    else:  # EDIT NOTE
        audio_diary_manager.update_audio_diary(data)
        audio_diary_id = data['audio_diary_id']
        e_context_manager.delete_environmental_by_audio_diary_id(audio_diary_id)

    text_diary_manager = database.TextDiaryManager()
    text_diary_id = text_diary_manager.create_text_diary(data['audio_diary_id'], data['content'], data['created_date'])

    if 'environmental _context' in data and data['environmental _context']:
        e_context_manager.create_environmental_context(audio_diary_id=data['audio_diary_id'], diary_context_info=data['environmental _context'])

    # FILE UPLOAD LOGIC-------------------------------------------------------------------------------------------------
    # Audio File Uploading
    if request.FILES.get('file0', False):  # file checking
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        DIARY_DIR = os.path.join(ROOT_DIR, 'uploaded', str(data['user_id']), str(audio_diary_id))
        MEDIA_DIR = os.path.join(DIARY_DIR, 'media')
        if not (os.path.isdir(DIARY_DIR)):
            os.mkdir(DIARY_DIR)
        if not (os.path.isdir(MEDIA_DIR)):
            os.mkdir(MEDIA_DIR)

        picture_extension_list = ['.ani', '.bmp', '.cal', '.fax', '.gif', '.img', '.jbg', '.jpe', '.jpeg', '.jpg',
                                  '.mac', '.pbm', '.pcd', '.pcx', '.pct', '.pgm', '.png', '.ppm', '.psd', '.ras',
                                  '.tga', '.tiff', '.wmf']
        music_extension_list = ['.3gp', '.aa', '.aac', '.aax', '.act', '.aiff', '.amr', '.ape', '.au', '.awb', '.dct',
                                '.dss', '.dvf', '.flac', '.gsm', '.iklax', '.ivs', '.m4a', '.m4b', '.m4p', '.mmf',
                                '.mp3', '.mpc', '.msv', '.ogg', '.oga', 'mogg', '.opus', '.ra', '.rm', '.raw', '.sln',
                                '.tta', '.vox', '.wav', '.wma', '.wv', '.webm']
        video_extension_list = ['.webm', '.mkv', '.flv', '.flv', '.vob', '.ogv', '.ogg', '.gif', '.gifv', '.mng',
                                '.avi', '.mov', '.qt', '.wmv', '.yuv', '.rm', '.rmvb', '.asf', '.amv', '.mp4', '.m4p',
                                '.m4v', '.mpg', '.mp2', '.mpeg', '.mpe', '.mpv', '.mpg', '.mpeg', '.m2v', '.m4v',
                                '.svi', '.3gp', '.3g2', '.flv', '.f4v', '.f4p', '.f4a', '.f4b']

        mc_manager = database.MediaContextManager()
        import pprint
        pprint.pprint(request.FILES)
        for key, value in request.FILES.items():
            # Saving UPLOADED Files
            if key == 'file0':
                file_path = os.path.join(DIARY_DIR, str(request.FILES[key].name))
                with open(file_path, 'wb') as destination:
                    for chunck in request.FILES[key]:
                        destination.write(chunck)
            else:
                file_name = str(request.FILES[key].name)
                file_path = os.path.join(MEDIA_DIR, file_name)
                mc_data = {'audio_diary_id': audio_diary_id, 'path': file_path}
                if any(tp in file_name for tp in picture_extension_list):
                    mc_data['type'] = 'picture'

                elif any(tp in file_name for tp in music_extension_list):
                    mc_data['type'] = 'music'

                elif any(tp in file_name for tp in video_extension_list):
                    mc_data['type'] = 'video'
                mc_id = mc_manager.create_media_context(audio_diary_id, mc_data)
                tmp = {'file_name': file_name, 'media_context_id': mc_id}
                mc_id_list.append(tmp)

                with open(file_path, 'wb') as destination:
                    for chunck in request.FILES[key]:
                        destination.write(chunck)



    # FILE UPLOAD LOGIC END---------------------------------------------------------------------------------------------

    # NLP PICKLING LOGIC------------------------------------------------------------------------------------------------

    # pickling in thread
    pickle_th = threading.Thread(target=pickling, args=(str(data['user_id']), audio_diary_id, data['content']))
    pickle_th.start()

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

    return audio_diary_id, text_diary_id, mc_id_list


def pickling(user_id, audio_diary_id, content):
    # START : for calculating execution time
    start = timeit.default_timer()
    logger.debug("PICKLE : START")
    # init DB
    # 0 for False, 1 for doing, 2 for complete
    audio_diary_manager = database.AudioDiaryManager()
    try:
        # making pickles------------------------------------------------------------------------------------------------
        # making path
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        PICKLE_DIR = os.path.join(ROOT_DIR, 'pickles', user_id, str(audio_diary_id))
        if not (os.path.isdir(PICKLE_DIR)):
            os.mkdir(PICKLE_DIR)

        # set state
        audio_diary_manager.update_pickle_state(audio_diary_id, 1)  # pickling is on going

        # making pickles
        pos_texts = tagger.tag_pos_doc(content, True)

        # saving pickles
        save_result = tagger.tags_to_pickle(pos_texts, os.path.join(PICKLE_DIR, "pos_texts.pkl"))

        # update audio_diary.pickle
        if save_result:
            audio_diary_manager.update_pickle_state(audio_diary_id, 2)  # pickling is successfully finished
            stop = timeit.default_timer()
            logger.debug("PICKLE : SUCCESSFUL - Execution Time : %s", stop - start)
            return True
        else:
            raise Exception('SAVING PICKLES FAILED')

    except Exception as exp:
        logger.exception(exp)
        logger.debug("PICKLE : FAIL")
        audio_diary_manager.update_pickle_state(audio_diary_id, 0)
        return False


def parse_lifetype(lifetype):
    parsed = str(lifetype).split(',')
    return parsed



