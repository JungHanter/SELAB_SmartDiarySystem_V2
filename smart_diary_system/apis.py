import json
import timeit
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from smart_diary_system import database
from smart_diary_system import security
import logging
from django.http import Http404
from diary_analyzer import tagger
from diary_analyzer import tendency
from diary_analyzer import activity_pattern
import datetime
import pprint
import operator
import random

# from diary_nlp import nlp_en
from langdetect import detect
import os
import shutil
import threading
import uuid
from django.template import loader
from django.shortcuts import render

from django.http import QueryDict
from django.conf import settings
from django.http import HttpResponse
from django.core.mail import send_mail

logging.basicConfig(
    format="[%(name)s][%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()],
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

salt = 'lovejesus'
email_auth_url = 'http://203.253.23.7:8000/auth?auth_key='

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

                    # generate auth key
                    auth_key = uuid.uuid4().hex

                    # DB Transaction
                    result = user_manager.create_user(user_id=data.get('user_id'), password=enc, name=data.get('name'),
                                                   birthday=data.get('birthday'), gender=data.get('gender'),
                                                   email=data.get('email'), phone=data.get('phone'), auth_key=auth_key)

                    logger.debug('CREATE USER RESULT : %s', result)
                    if (result != 'PK') and (result != 'EMAIL') and result:
                        # Making Uploading Folders
                        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                        os.mkdir(os.path.join(ROOT_DIR, 'uploaded', str(data.get('user_id'))))
                        os.mkdir(os.path.join(ROOT_DIR, 'pickles', str(data.get('user_id'))))

                        # Sending Authorization Email to User's Email
                        auth_email_th = threading.Thread(target=send_auth_email, args=(auth_key, data.get('email')))
                        auth_email_th.start()


                        return JsonResponse({'register': True})
                    elif result == 'PK':
                        logger.debug("SIGN UP FAILED (EXISTED ID) : %s", data.get('user_id'))
                        return JsonResponse({'register': False, 'reason': 'EXISTED ID'})
                    elif result == 'EMAIL':
                        logger.debug("SIGN UP FAILED (EXISTED EMAIL) : %s", data.get('email'))
                        return JsonResponse({'register': False, 'reason': 'EXISTED EMAIL'})
                    else:
                        logger.debug("SIGN UP FAILED (INTERNAL SERVER ERROR) : %s", data)
                        return JsonResponse({'register': False, 'reason': 'INTERNAL SERVER ERROR'})
                except Exception as exp:
                    logger.exception(exp)
                    return JsonResponse({'register': False, 'reason': 'INTERNAL SERVER ERROR'})

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
                        logger.debug("LOGIN FAILED (UNREGISTERED ID) : %s", data.get('user_id'))
                        return JsonResponse({'login': False, 'reason': 'UNREGISTERED ID'})
                    else:
                        # PASSWORD(encrypted) from DB
                        cipher = security.AESCipher(salt)
                        plain = cipher.decrypt(user_info['password'])

                        if password_from_user is not None:
                            if password_from_user == plain:
                                if user_info['auth_key'] == '0':
                                    tag_manager = database.TagManager()
                                    tag_list = tag_manager.retrieve_tag_by_user_id(user_info['user_id'])
                                    logger.debug("LOGIN SUCCESS : %s", data.get('user_id'))
                                    return JsonResponse({'login': True, 'name': user_info['name'],
                                                         'birthday': user_info['birthday'], 'gender': user_info['gender'],
                                                         'email': user_info['email'], 'phone': user_info['phone'], 'tag_list': tag_list
                                                         })
                                else:
                                    logger.debug("LOGIN FAILED (EMAIL AUTH) : %s", data.get('user_id'))
                                    return JsonResponse({'login': False, 'reason': 'EMAIL AUTH REQUIRED', 'email': user_info['email']})
                            else:
                                logger.debug("LOGIN FAILED (MISMATCH PW) : %s", data.get('user_id'))
                                return JsonResponse({'login': False, 'reason': 'MISMATCH PW'} )
                        else:
                            logger.debug("LOGIN FAILED (UNREGISTERD ID) : %s", data.get('user_id'))
                            return JsonResponse({'login': False, 'reason': 'INVALID PW'})
                except Exception as exp:
                    logger.exception(exp)
                    return JsonResponse({'login': False})

            elif option == 'recovery':
                try:
                    data = json.loads(request.body.decode('utf-8'))
                    logger.debug("INPUT %s", data)
                    user_manager = database.UserManager()
                    email = data.get('email')
                    result = user_manager.get_id_pw_by_email(email)
                    if result:
                        cipher = security.AESCipher(salt)
                        plain = cipher.decrypt(result['password'])

                        email_th = threading.Thread(target=send_recovery_email, args=(result['user_id'], plain, email))
                        email_th.start()

                        return JsonResponse({'recovery': True})
                    else:
                        return JsonResponse({'recovery': False, 'reason': 'NOT REGISTERED'})
                except Exception as exp:
                    logger.exception(exp)
                    return JsonResponse({'recovery': False, 'reason': 'INTERNAL SERVER ERROR'})

            elif option == 'withdraw':
                try:
                    data = json.loads(request.body.decode('utf-8'))
                    logger.debug("INPUT %s", data)
                    user_manager = database.UserManager()
                    user_id = data['user_id']
                    result = user_manager.delete_user(user_id=user_id)

                    if result:
                        # delete Files
                        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                        shutil.rmtree(os.path.join(ROOT_DIR, 'uploaded', user_id))
                        shutil.rmtree(os.path.join(ROOT_DIR, 'pickles', user_id))
                        return JsonResponse({'delete_user': True})
                    else:
                        return JsonResponse({'delete_user': False})

                except Exception as exp:
                    logger.exception(exp)
                    return JsonResponse({'delete_user': False})

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
                                         birthday=put.get('birthday'), password=enc,
                                         phone=put.get('phone'))
                return JsonResponse({'update_user': True})

            except Exception as exp:
                logger.exception(exp)
                return JsonResponse({'update_user': False})

        elif option == 'email':  # Update User Info
            try:
                # Update info from APP
                put = json.loads(request.body.decode('utf-8'))
                logger.debug("INPUT : %s", put)

                # Encrypting Password
                cipher = security.AESCipher(salt)
                enc = cipher.encrypt(put.get('password'))

                # generate auth key
                auth_key = uuid.uuid4().hex

                # DB Transaction
                user_manager = database.UserManager()
                result = user_manager.update_email(put.get('user_id'), enc, put.get('email'), auth_key)

                if result:
                    # Sending Authorization Email to User's Email
                    auth_email_th = threading.Thread(target=send_auth_email, args=(auth_key, put.get('email')))
                    auth_email_th.start()

                    return JsonResponse({'update_email': True})
                elif result is None:
                    return JsonResponse({'update_email': False, 'reason': 'EXIST EMAIL'})
                else:
                    return JsonResponse({'update_email': False, 'reason': 'USER NOT FOUND'})

            except Exception as exp:
                logger.exception(exp)
                return JsonResponse({'update_email': False})


@csrf_exempt
def auth_email(request):
    if request.method == 'GET':
        try:
            data = json.loads(json.dumps(request.GET))
            logger.debug("INPUT : %s", data)

            user_manager = database.UserManager()
            result = user_manager.auth_user_mail(data['auth_key'])

            if result:
                logger.debug("EMAIL AUTH SUCCESSFUL : %s, %s", result['user_id'], result['email'])
                return render(request, 'auth_success.html', {'user_id': result['user_id'], 'email': result['email']}, content_type='text/html')
            else:
                logger.debug("EMAIL AUTH FAILED")
                return render(request, 'auth_fail.html', content_type='text/html')

        except Exception as exp:
            logger.exception(exp)
            logger.debug("RETURN : FALSE - EXCEPTION")
            return JsonResponse({'retrieve_diary': False})


@csrf_exempt
def manage_diary(request, option=None):
    logger.debug(request)
    if request.method == 'POST':
        if option is None:  # create diary
            try:
                # Diary Info from APP (multipart)
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

        elif option == 'update':   # update diary
            try:
                # input From APP
                data = json.loads(request.body.decode('utf-8'))  # at BODY
                logger.debug("INPUT : %s", data)

                updated = update_diary(data)
                if updated:
                    return JsonResponse({'update_diary': True})
                else:
                    return JsonResponse({'update_diary': False})

            except Exception as exp:
                logger.exception(exp)
                return JsonResponse({'update_diary': False})

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
                mc_manager = database.MediaContextManager()


                result = audio_diary_manager.retrieve_audio_diary_detail_by_audio_diary_id(data['user_id'], data['audio_diary_id'])
                tag_result = tag_manager.retrieve_tag_by_audio_diary_id(data['audio_diary_id'])
                mc_result = mc_manager.retrieve_media_context_by_audio_diary_id(data['audio_diary_id'])

                # cutting address
                final_mc_result = []
                for mc in mc_result:
                    tmp = {}
                    tmp['type'] = mc ['type']
                    tmp['media_context_id'] = mc['media_context_id']
                    tmp['file_name'] = os.path.basename(mc['path'])
                    final_mc_result.append(tmp)

                if result is False:
                    logger.debug("RETURN : FALSE")
                    return JsonResponse({'retrieve_diary': False})
                else:  # result is exist
                    if result is not None:
                        result_context = e_context_manager.retrieve_environmental_context_by_audio_diary_id(data['audio_diary_id'])
                    else:
                        result_context = []
                    logger.debug(
                        "RETURN : result_detail : %s \n result_environmental_context : %s \n result_tag_list : %s \n result_media_context_list %s" %
                        (result, result_context, tag_result, final_mc_result))
                    return JsonResponse({'retrieve_diary': True, 'result_detail': result, 'result_environmental_context': result_context,
                                         'result_tag_list': tag_result, 'result_media_context_list': final_mc_result})

        except Exception as exp:
            logger.exception(exp)
            logger.debug("RETURN : FALSE - EXCEPTION")
            return JsonResponse({'retrieve_diary': False})


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

        if option == 'tendency' or option == 'activity_pattern':
            try:
                data = json.loads(json.dumps(request.GET))
                logger.debug("INPUT :%s", data)

                # init DB Mangagers
                audio_diary_manager = database.AudioDiaryManager()
                life_style_manager = database.TendencyManager()

                # retrieve audio diaries which will be analyzed
                audio_diary_list = audio_diary_manager.retrieve_audio_diary_list_by_timestamp(data)  # user_id, timestamp_from, timestamp_to
                if audio_diary_list is None:
                    # Nothing to show
                    logger.debug("RETURN : TRUE - NO DIARY AVAILABLE")
                    return JsonResponse({'analyzed': True, 'result': {'pos': [], 'neg': []}})
                else:
                    diary_tag_list = []
                    diary_date_list = []

                    for audio_diary in audio_diary_list:  # load pickles
                        # pprint.pprint(audio_diary)
                        diary_tags_tuple = None

                        if audio_diary['pickle'] == 0:
                            # do pickling again...
                            pass
                        elif audio_diary['pickle'] == 1:
                            # wait for pickling
                            pass
                        else: #2
                            # load pickles
                            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                            PICKLE_DIR = os.path.join(ROOT_DIR, 'pickles', audio_diary['user_id'],
                                                      str(audio_diary['audio_diary_id']), 'pos_texts.pkl')
                            diary_tags_tuple = tagger.pickle_to_tags(PICKLE_DIR)

                        if diary_tags_tuple:
                            diary_tag_list.append(tagger.pickle_to_tags(PICKLE_DIR)[1])
                            if option == 'activity_pattern':
                                diary_date_list.append(datetime.datetime.fromtimestamp(int(audio_diary['created_date']/1000)))
                        else:
                            # privious picklingis failed. do pickling again.
                            pass

                    if option == 'tendency':
                        # analyze tendency
                        pos_tend, neg_tend = tendency.tend_analyzer.analyze_diaries(diary_tag_list)
                        # pprint.pprint(pos_tend)
                        # pprint.pprint(neg_tend)
                        return JsonResponse({'analyzed': True, 'result': {'pos': pos_tend, 'neg': neg_tend}})
                    elif option == 'activity_pattern':
                        # pprint.pprint(diary_date_list)

                        interval = data.get('interval')
                        if interval is None:
                            interval = 0
                        else:
                            interval = int(interval)
                        if interval == 0:
                            if len(audio_diary_list) > 2:
                                period_millis = audio_diary_list[0]['created_date'] \
                                        - audio_diary_list[len(audio_diary_list)-1]['created_date']
                                period_days = period_millis / (1000*60*60*24)
                                if period_days >= (365 * 5): interval = 30;
                                elif period_days >= 365: interval = 14;
                                elif period_days >= 90: interval = 7;
                                else: interval = 3;
                            else:
                                interval = 3

                        recurrent, frequency, regularity = activity_pattern.activity_pattern_analyzer\
                            .analyze_diaries(diary_tag_list, diary_date_list, interval)
                        recurrent, frequency, regularity = activity_pattern.rank_result(recurrent, frequency, regularity)
                        return JsonResponse({'analyzed': True, 'result':
                            {'recurrent': recurrent, 'frequency': frequency, 'regularity': regularity}})

                    # making tendency DB record
                    # tendencys_dict_list = []
                    # for audio_diary_id, diary_tags in zip(analyzed_audio_diary_id_list, diary_tag_list):
                    #     # analyze tendency
                    #     if thing_type == 'food':
                    #         tendency_analyze_result = lifestyles.analyzer.analyze_food(diary_tags[1])
                    #     elif thing_type == 'sport':
                    #         # tendency_analyze_result = tendencys.analyzer.sport_collect(diary_tags[1])
                    #         return JsonResponse({'tendency': False, 'reason': 'NOT YET IMPLEMENTED'})
                    #
                    #     for tendency_item in tendency_analyze_result.keys():
                    #         tendency_dict = {'audio_diary_id': audio_diary_id['audio_diary_id'], 'thing_type': thing_type,
                    #                            'thing': tendency_item, 'score': tendency_analyze_result[tendency_item]}
                    #         tendencys_dict_list.append(tendency_dict)
                    #
                    # if tendencys_dict_list:  # insert tendency record into DB
                    #     life_style_manager.create_tendency_by_list(tendencys_dict_list)

                    # statistic analyze
                    # tendency_item_list = life_style_manager.retrieve_tendency(ranking_audio_diary_id_list, thing_type)
                    # if tendency_item_list:
                    #     if str(data['option']).lower() == 'like':
                    #         like = True
                    #     elif str(data['option']).lower() == 'dislike':
                    #         like = False
                    #     else:
                    #         logger.debug("RETURN : FALSE - INVALID OPTION TYPE")
                    #         return JsonResponse({'tendency': False, 'reason': 'INVALID OPTION TYPE'})
                    #     final_result = lifestyles.ranking_lifestyle(lifestyle_list=tendency_item_list, like=like)
                    #     logger.debug("RETURN : %s", final_result)
                    #
                    #     # updating tendency_analyzed flag in audio_diary table
                    #     audio_diary_manager.update_tendency_analyzed_state(analyzed_audio_diary_id_list)
                    #     return JsonResponse({'tendency': True, 'result': final_result})
                    # else:
                    #     return JsonResponse({'tendency': True, 'result': []})

            except Exception as exp:
                logger.exception(exp)
                logger.debug("RETURN : FALSE - EXCEPTION")
                return JsonResponse({'analyzed': False, 'reason': 'INTERNAL SERVER ERROR'})

    elif request.method == 'PUT':
        try:
            if option is None:
                pass

        except Exception as exp:
            logger.exception(exp)
            return JsonResponse({'update_diary': False})


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
                                    logger.debug("SENDING FILE : %s" & file)
                                    return response
                            else:
                                raise Http404
                    else:
                        mc_manager = database.MediaContextManager()
                        mc_info = mc_manager.retrieve_media_context_by_mc_id(data['media_context_id'])

                        if mc_info['type'] == 'picture':
                            with open(mc_info['path'], 'rb') as fh:
                                response = HttpResponse(fh.read(), content_type="image")
                                response['Content-Disposition'] = 'inline; filename=' + str(
                                    os.path.basename(mc_info['path']))
                                logger.debug("SENDING FILE : %s" & fh.name)
                                return response
                        elif mc_info['type'] == 'video':
                            with open(mc_info['path'], 'rb') as fh:
                                response = HttpResponse(fh.read(), content_type="video")
                                response['Content-Disposition'] = 'inline; filename=' + str(
                                    os.path.basename(mc_info['path']))
                                logger.debug("SENDING FILE : %s" & fh.name)
                                return response
                        elif mc_info['type'] == 'music':
                            with open(mc_info['path'], 'rb') as fh:
                                response = HttpResponse(fh.read(), content_type="audio")
                                response['Content-Disposition'] = 'inline; filename=' + str(
                                    os.path.basename(mc_info['path']))
                                logger.debug("SENDING FILE : %s" & fh.name)
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
                    if not('media_context_id' in data):
                        for file in file_list:
                            if '.wav' in file:
                                file_path = os.path.join(file_path, file)
                                with open(file_path, 'rb') as fh:
                                    response = HttpResponse(fh.read(), content_type="audio/wav")
                                    response['Content-Disposition'] = 'inline; filename=' + file
                                    logger.debug("SENDING FILE : %s" % fh.name)
                                    return response
                    else:
                        mc_manager = database.MediaContextManager()
                        mc_info = mc_manager.retrieve_media_context_by_mc_id(data['media_context_id'])

                        if mc_info['type'] == 'picture':
                            with open(mc_info['path'], 'rb') as fh:
                                response = HttpResponse(fh.read(), content_type="image")
                                response['Content-Disposition'] = 'inline; filename=' + str(
                                    os.path.basename(mc_info['path']))
                                logger.debug("SENDING FILE : %s" % fh.name)
                                return response
                        elif mc_info['type'] == 'video':
                            with open(mc_info['path'], 'rb') as fh:
                                response = HttpResponse(fh.read(), content_type="video")
                                response['Content-Disposition'] = 'inline; filename=' + str(
                                    os.path.basename(mc_info['path']))
                                logger.debug("SENDING FILE : %s" % fh.name)
                                return response
                        elif mc_info['type'] == 'music':
                            with open(mc_info['path'], 'rb') as fh:
                                response = HttpResponse(fh.read(), content_type="audio")
                                response['Content-Disposition'] = 'inline; filename=' + str(
                                    os.path.basename(mc_info['path']))
                                logger.debug("SENDING FILE : %s" % fh.name)
                                return response
                        elif data['type'] == 'handdrawn':
                            pass
                        else:
                            logger.debug("NOT VAILD TYPE OF MEDIA CONTEXT")
                            raise Http404
                else:
                    logger.debug("NO FILE IN diary directory")
                    raise Http404
            else:
                logger.debug("NO INPUT")
                raise Http404

        except Exception as exp:
            logger.exception(exp)
            raise Http404


@csrf_exempt
def uploading_test(request):
    if request.FILES.get('file0', False):  # file checking
        MAX_FILE_SIZE = 104857600
        # MAX_FILE_SIZE = 1048570
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

        logger.debug(pprint.pformat(request.FILES))
        i = 0
        while True:
            key = 'file' + str(i)
            if request.FILES.get(key, False):
                # for key, value in request.FILES.items():
                # Saving UPLOADED Files
                if key == 'file0':
                    file_path = os.path.join(DIARY_DIR, str(request.FILES[key].name))
                    with open(file_path, 'wb') as destination:
                        for chunck in request.FILES[key]:
                            destination.write(chunck)
                else:
                    # File Size Checking
                    file = request.FILES[key]
                    file.seek(0, os.SEEK_END)
                    file_size = file.tell()
                    if file_size > MAX_FILE_SIZE:
                        logger.debug("FILE ( %s ) exceed 100MB" % file.name)
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
                i += 1
            else:
                break

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
    # if not ('audio_diary_id' in data):  # NEW NOTE
    audio_diary_id = audio_diary_manager.create_audio_diary(
        data['user_id'], data['title'], data['created_date'])
    data['audio_diary_id'] = audio_diary_id
    if not('tag' in data):
        data['tag'] = []
    tag_manager.create_tag(audio_diary_id, data['tag'])
    # else:  # EDIT NOTE
    #     audio_diary_manager.update_audio_diary(data)
    #     audio_diary_id = data['audio_diary_id']
    #     e_context_manager.delete_environmental_by_audio_diary_id(audio_diary_id)

    text_diary_manager = database.TextDiaryManager()
    text_diary_id = text_diary_manager.create_text_diary(data['audio_diary_id'], data['content'])

    if 'environmental_context' in data and data['environmental_context']:
        logger.debug('ec_context : %s ' % data['environmental_context'])
        e_context_manager.create_environmental_context(audio_diary_id=data['audio_diary_id'], diary_context_info=data['environmental_context'])

    # FILE UPLOAD LOGIC-------------------------------------------------------------------------------------------------
    # Audio File Uploading
    MAX_FILE_SIZE = 104857600  # 100MB in byte
    logger.debug('UPLOADED FILES')
    logger.debug(pprint.pformat(request.FILES))
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
        i = 0
        while True:
            key = 'file' + str(i)
            if request.FILES.get(key, False):
                if key == 'file0':
                    file_path = os.path.join(DIARY_DIR, str(request.FILES[key].name))
                    with open(file_path, 'wb') as destination:
                        for chunck in request.FILES[key]:
                            destination.write(chunck)
                else:
                    # File Size Checking
                    file = request.FILES[key]
                    file.seek(0, os.SEEK_END)
                    file_size = file.tell()
                    if file_size > MAX_FILE_SIZE:
                        logger.debug("FILE ( %s ) exceed 100MB" % file.name)
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
                i += 1
            else:
                break
    # FILE UPLOAD LOGIC END---------------------------------------------------------------------

    # pickling in thread
    pickle_th = threading.Thread(target=pickling, args=(str(data['user_id']), audio_diary_id, data['content']))
    pickle_th.start()

    return audio_diary_id, text_diary_id, mc_id_list


def update_diary(data):
    audio_diary_manager = database.AudioDiaryManager()
    # DB Transaction
    # KerError Exception Handling
    if not ('content' in data) or data['content'] is '' or data['content'] is None:
        return False

    # UPDATE Text INTO DB
    audio_diary_id = data['audio_diary_id']
    updated = audio_diary_manager.update_audio_diary(
        data['user_id'], data['user_id'], data['title'], data['created_date'])
    if not updated:
        return False

    text_diary_manager = database.TextDiaryManager()
    updated = text_diary_manager.update_text_diary(
        audio_diary_id, data['content'])

    if not updated:
        return False

    pickle_th = threading.Thread(target=pickling, args=(str(data['user_id']), audio_diary_id, data['content']))
    pickle_th.start()
    return True


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


def send_auth_email(auth_key, email):
    start = timeit.default_timer()
    logger.debug("SENDING AUTH E-MAIL TO : %s", email)
    auth_link = email_auth_url + auth_key
    html_message = loader.render_to_string('auth_email_form.html', {
            'auth_link': auth_link
        }
    )
    send_mail('Authorization Email for Smart Diary', '', 'test@abc.com', [email], html_message=html_message)
    stop = timeit.default_timer()
    logger.debug("SENDING AUTH E-MAIL : SUCCESSFUL - Execution Time : %s", stop - start)


def send_recovery_email(user_id, pw, email):
    start = timeit.default_timer()
    html_message = loader.render_to_string('recovery_email_form.html', {
        'user_id': user_id,
        'password': pw
    })
    logger.debug("SENDING RECOVER E-MAIL TO : %s", email)
    send_mail('Recovery Email for Smart Diary', '', 'test@abc.com', [email], html_message=html_message)
    stop = timeit.default_timer()
    logger.debug("SENDING RECOVER E-MAIL : SUCCESSFUL - Execution Time : %s", stop - start)


def parse_lifetype(lifetype):
    parsed = str(lifetype).split(',')
    return parsed



