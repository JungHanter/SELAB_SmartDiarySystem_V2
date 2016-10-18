import logging
import pymysql
import time
import timeit
import datetime

logging.basicConfig(
    format="[%(name)s][%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()],
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class DBManager(object):
    def __init__(self):
        """DB Connection Class for Smart Diary (SD) Database

        """
        try:
            self.conn = pymysql.connect(host='203.253.23.17', port=3306, user='root', passwd='lovejesus',
                                        db='smartdiary', charset='utf8', use_unicode=True)
            self.connected = True
        except pymysql.Error:
            self.connected = False


class UserManager(DBManager):
    def __init__(self):
        """DB Model Class for smarydiary.user table

        """
        DBManager.__init__(self)

    def add_user(self, user_id, password, name, timestamp, gender):
        """Adding new user to SD DB
        Usually, this method be called
        When User register to SD

        :param age:
        :param gender:
        :param name:
        :param password:
        :param user_id: id of newly added user
        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        query_for_add_user = "INSERT INTO user " \
                             "(user_id, password, name, timestamp, gender) " \
                             "VALUES (%s, %s, %s, %s, %s)"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_add_user,
                            (user_id, password, name, timestamp, gender))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : add_user() - Execution Time : %s", stop - start)
            return True

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At add_user()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def get_user(self, user_id):  # for retrieving user data
        """retrieving new user from H.I.S DB
        Usually, this method be called
        When User Information need to be display

        :param user_id: id of user of Retrieving Target
        :rtype: dict contains user's inforamtion
        """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected  # Connection Check Flag
        query_for_get_user = "SELECT * FROM user WHERE user_id = %s"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_get_user, user_id)
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : get_user() - Execution Time : %s", stop - start)
                return cur.fetchone()
        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At get_user()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)

    def update_user(self, user_info):
        """retrieving new user from H.I.S DB
            Usually, this method be called
            When User Information need to be display

            :param user_id: id of user of Retrieving Target
            :rtype: dict contains user's inforamtion
            """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected  # Connection Check Flag
        query_for_update_user = "UPDATE user SET name = %s, timestamp = %s, gender = %s, password = %s WHERE user_id = %s"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_update_user, (user_info['name'], user_info['timestamp'],
                                                    user_info['gender'], user_info['password'],
                                                    user_info['user_id']))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : update_user() - Execution Time : %s", stop - start)
                return cur.fetchone()

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At update_user()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)


class DiaryManager(DBManager):
    def __init__(self):
        """DB Model Class for smarydiary.diary table

        """
        DBManager.__init__(self)

    def create_diary(self, diary_info):
        """Creating new diary to SD DB
        Usually, this method be called
        When User creating new diary


        :param diary_info:
        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        query_for_create_diary = "INSERT INTO diary " \
                             "(user_id, location, timestamp, title, annotation, audio) " \
                             "VALUES (%s, %s, %s, %s, %s, %s)"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_create_diary,
                            (diary_info['user_id'],
                             diary_info['location'], diary_info['timestamp'],
                             diary_info['title'],
                             diary_info['annotation'],
                             diary_info['audio']))
                cur.execute("SELECT LAST_INSERT_ID()")
                self.conn.commit()
                diary_id = cur.fetchone()['LAST_INSERT_ID()']
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : create_diary() - Execution Time : %s", stop - start)
            return diary_id

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At create_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_diary(self, diary_id):
        """Creating new diary to SD DB
        Usually, this method be called
        When User retrieving diary


        :param diary_info:
        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_create_diary = "SELECT diary.diary_id, diary.user_id, diary.title, diary.timestamp, diary.location, diary.annotation, " \
                              "IF(audio, 'True', 'False') as audio, converted_text.text FROM diary, converted_text WHERE diary.diary_id = %s AND diary.diary_id = converted_text.diary_id"
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_create_diary, diary_id)
                result = cur.fetchone()
                if result:
                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : retrieve_user() - Execution Time : %s", stop - start)
                    logger.debug('DB RESULT : %s', result) 
                    return result
                else:
                    return False

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_diary_detail(self, diary_info):
        """Creating new diary to SD DB
        Usually, this method be called
        When User retrieving diary


        :param diary_info:
        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_diary = "SELECT diary.diary_id, diary.user_id, diary.title, diary.timestamp, diary.annotation, " \
                              "IF(audio, 'True', 'False') as audio, converted_text.text as text, " \
                              "diary.location, " \
                              "converted_text.c_text_id as c_text_id " \
                                     "FROM diary, converted_text " \
                                     "WHERE diary.user_id = %s and diary.diary_id = %s " \
                                     "and diary.diary_id = converted_text.diary_id"
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_diary, (diary_info['user_id'], diary_info['diary_id']))
                diary_result = cur.fetchone()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : retrieve_diary_detail() - Execution Time : %s", stop - start)
            if diary_result:
                pass
                # query_for_sentence = "SELECT text as sentence, sentence_id FROM sentence WHERE c_text_id = %s"
                # with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                #     cur.execute(query_for_sentence, diary_result['c_text_id'])
                #     sentence_result = cur.fetchall()
                # if sentence_result:
                #     query_for_s_element = "SELECT text, pos, element_no FROM s_element WHERE sentence_id = %s"
                #     for idx, sentence in enumerate(sentence_result):
                #         with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                #             cur.execute(query_for_s_element, sentence['sentence_id'])
                #             s_element_result = cur.fetchall()
                #         sentence_result[idx]['s_element_list'] = s_element_result
                #     diary_result['sentence_list'] = sentence_result
                # else:
                #     pass
            else:
                diary_result = False

            return diary_result

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_diary_detail()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_diary_list_wit_c_text_list(self, c_text_id_list):
        """Creating new diary to SD DB
            Usually, this method be called
            When User retrieving diary


            :param diary_info:
            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_list = "SELECT converted_text.text, diary.title, diary.timestamp, diary.user_id, " \
                             "diary.diary_id, converted_text.c_text_id, diary.annotation, " \
                              "IF(audio, 'True', 'False') as audio, diary.location FROM diary, converted_text " \
                             "WHERE diary.diary_id = converted_text.diary_id AND converted_text.c_text_id IN ( "
            flag = True
            for c_text_id in c_text_id_list:
                if c_text_id[1] == 0.0:
                    break
                query_for_list = query_for_list + str(int(c_text_id[0])) + ','
                flag = False
            if flag:
                return None
            query_for_list = query_for_list[:-1]
            query_for_list = query_for_list + " ) LIMIT 10 "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_list)
                result = cur.fetchall()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : retrieve_diary_list_wit_c_text_list() - Execution Time : %s", stop - start)
                if result:
                    logger.debug('DB RESULT : %s', result) 
                    return result
                else:
                    return None

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_c_text_list_from_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_sentence_list_from_diary(self, diary_id):
        """Creating new diary to SD DB
            Usually, this method be called
            When User retrieving diary


            :param diary_info:
            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_list = "SELECT sentence.text FROM sentence INNER JOIN converted_text " \
                             "WHERE converted_text.diary_id = %s "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_list, diary_id)
                result = cur.fetchall()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : retrieve_sentence_list_from_diary() - Execution Time : %s", stop - start)

                if result:
                    logger.debug('DB RESULT : %s', result)

                    return result
                else:
                    return None

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrive_setence_list_from_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_diary_with_keyword(self, diary_info):
        """Creating new diary to SD DB
            Usually, this method be called
            When User retrieving diary


            :param diary_info:
            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_list = "SELECT diary.diary_id, diary.user_id, diary.title, diary.timestamp, diary.annotation, " \
                              "IF(audio, 'True', 'False') as audio, converted_text.text, diary.location FROM diary, converted_text " \
                             "WHERE diary.diary_id = converted_text.diary_id AND converted_text.text LIKE %s " \
                             "AND diary.user_id = %s "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                if 'limit' in diary_info:
                    if 'offset' in diary_info:
                        query_for_list += " ORDER BY diary.timestamp DESC LIMIT %s OFFSET %s"
                        cur.execute(query_for_list, ('%'+diary_info['keyword']+'%', diary_info["user_id"],
                                                     int(diary_info['limit']), int(diary_info['offset'])))
                    else:
                        query_for_list += " ORDER BY diary.timestamp DESC LIMIT %s"
                        cur.execute(query_for_list, ('%'+diary_info['keyword']+'%', diary_info["user_id"],
                                                     int(diary_info['limit'])))
                else:
                    query_for_list += " ORDER BY diary.timestamp DESC"
                    cur.execute(query_for_list, ('%'+diary_info['keyword']+'%', diary_info["user_id"]))

                result = cur.fetchall()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : retrieve_diary_with_keyword() - Execution Time : %s", stop - start)
                if result:

                    logger.debug('DB RESULT : %s', result) 
                    return result
                else:
                    return None

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_diary_with_keyword()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def update_diary(self, diary_info):
        """Creating new diary to SD DB
            Usually, this method be called
            When User retrieving diary


            :param diary_info:
            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_delete = "UPDATE diary SET title = %s, location = %s, annotation = %s, audio = %s, timestamp = %s WHERE diary_id = %s "

            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_delete, (diary_info['title'], diary_info['location'], diary_info['annotation'],
                                               diary_info['audio'], int(diary_info['timestamp']),
                                               int(diary_info['diary_id'])))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : update_diary() - Execution Time : %s", stop - start)

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At update_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def delete_diary(self, diary_id):
        """Creating new diary to SD DB
        Usually, this method be called
        When User retrieving diary


        :param diary_info:
        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_delete = "DELETE FROM diary WHERE diary_id  = %s "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_delete, int(diary_id))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : update_diary() - Execution Time : %s", stop - start)

            return True

        except Exception as exp:
            logger.exception(exp)
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At delete_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def delete_c_text_from_diary(self, diary_id):
        """Creating new diary to SD DB
        Usually, this method be called
        When User retrieving diary


        :param diary_info:
        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_delete = "DELETE FROM converted_text WHERE diary_id  = %s "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_delete, int(diary_id))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : delete_c_text_from_diary() - Execution Time : %s", stop - start)
            return True

        except Exception as exp:
            logger.exception(exp)
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At delete_c_text_from_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False


class ConvertedTextManager(DBManager):
    def __init__(self):
        """DB Model Class for smarydiary.converted_text table

        """
        DBManager.__init__(self)

    def add_converted_text(self, c_text_info):
        """Adding new diary to SD DB
        Usually, this method be called
        When User creating new diary

        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected
        query_for_add_c_text = "INSERT INTO converted_text " \
                             "(diary_id, text) " \
                             "VALUES (%s, %s) "
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_add_c_text, (c_text_info['diary_id'], c_text_info['text']))
                cur.execute("SELECT LAST_INSERT_ID()")
                self.conn.commit()

                c_text_id = cur.fetchone()['LAST_INSERT_ID()']

                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : add_converted_text() - Execution Time : %s", stop - start)
            return c_text_id
        except pymysql.MySQLError as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At add_converted_text()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def get_converted_text(self, c_text_id):
        """retrieving converted text from SD DB
        Usually, this method be called
        When ...

        :param c_text_id:
        :rtype: dict contains user's inforamtion
        """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected  # Connection Check Flag
        query_for_c_text = "SELECT * FROM converted_text WHERE c_text_id = %s"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_c_text, c_text_id)
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : get_converted_text() - Execution Time : %s", stop - start)
                result = cur.fetchone()
                if result:
                    logger.debug('DB RESULT : %s', result) 
                    return result
                else:
                    return None
        except pymysql.MySQLError as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At get_converted_text()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def get_converted_text_list(self, diary_info):
        """Creating new diary to SD DB
        Usually, this method be called
        When User creating new diary



        :type diary_info: dict contains user_id, timestamp_from, timestamp_to
        :rtype query result[list] or false:
        """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected
        query_for_get_c_text_list = "SELECT d.c_text_id, d.text, m.user_id, d.diary_id, m.timestamp, m.title, m.annotation FROM converted_text AS d " \
                                    "INNER JOIN diary as m " \
                                    "ON m.diary_id = d.diary_id " \
                                    "WHERE m.user_id = %s"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                if 'timestamp_from' in diary_info and 'timestamp_to' in diary_info:
                    if 'limit' in diary_info:
                        if 'offset' in diary_info:
                            query_for_get_c_text_list += " AND m.timestamp>=%s AND m.timestamp<=%s ORDER BY m.timestamp DESC LIMIT %s OFFSET %s"
                            cur.execute(query_for_get_c_text_list, (diary_info["user_id"], diary_info["timestamp_from"],
                                        diary_info["timestamp_to"], int(diary_info['limit']),
                                        int(diary_info['offset'])))
                        else:
                            query_for_get_c_text_list += " AND m.timestamp>=%s AND m.timestamp<=%s ORDER BY m.timestamp DESC LIMIT %s"
                            cur.execute(query_for_get_c_text_list, (diary_info["user_id"], diary_info["timestamp_from"],
                                                                    diary_info["timestamp_to"], int(diary_info['limit'])))
                    else:
                        query_for_get_c_text_list += " AND m.timestamp>=%s AND m.timestamp<=%s ORDER BY m.timestamp DESC"
                        cur.execute(query_for_get_c_text_list, (diary_info["user_id"], diary_info["timestamp_from"],
                                                                diary_info["timestamp_to"]))
                elif 'timestamp_from' in diary_info:
                    if 'limit' in diary_info:
                        if 'offset' in diary_info:
                            query_for_get_c_text_list += " AND m.timestamp>=%s ORDER BY m.timestamp DESC LIMIT %s OFFSET %s"
                            cur.execute(query_for_get_c_text_list, (diary_info["user_id"], diary_info["timestamp_from"],
                                        int(diary_info['limit']), int(diary_info['offset'])))
                        else:
                            query_for_get_c_text_list += " AND m.timestamp>=%s ORDER BY m.timestamp DESC LIMIT %s"
                            cur.execute(query_for_get_c_text_list,
                                        (diary_info["user_id"], diary_info["timestamp_from"],
                                        int(diary_info['limit'])))
                    else:
                        query_for_get_c_text_list += " AND m.timestamp>=%s ORDER BY m.timestamp DESC"
                        cur.execute(query_for_get_c_text_list, (diary_info["user_id"], diary_info["timestamp_from"]))

                elif 'timestamp_to' in diary_info:
                    if 'limit' in diary_info:
                        if 'offset' in diary_info:
                            query_for_get_c_text_list += " AND m.timestamp<=%s ORDER BY m.timestamp DESC LIMIT %s OFFSET %s"
                            cur.execute(query_for_get_c_text_list, (diary_info["user_id"], diary_info["timestamp_to"],
                                        int(diary_info['limit']), int(diary_info['offset'])))
                        else:
                            query_for_get_c_text_list += " AND m.timestamp<=%s ORDER BY m.timestamp DESC LIMIT %s"
                            cur.execute(query_for_get_c_text_list, (diary_info["user_id"], diary_info["timestamp_to"],
                                        int(diary_info['limit'])))
                    else:
                        query_for_get_c_text_list += " AND m.timestamp<=%s ORDER BY m.timestamp DESC"
                        cur.execute(query_for_get_c_text_list, (diary_info["user_id"], diary_info["timestamp_to"]))
                else:
                    if 'limit' in diary_info:
                        if 'offset' in diary_info:
                            query_for_get_c_text_list += " ORDER BY m.timestamp DESC LIMIT %s OFFSET %s"
                            cur.execute(query_for_get_c_text_list, (diary_info["user_id"], int(diary_info['limit']),
                                                                    int(diary_info['offset'])))
                        else:
                            query_for_get_c_text_list += " ORDER BY m.timestamp DESC LIMIT %s"
                            cur.execute(query_for_get_c_text_list, (diary_info["user_id"], int(diary_info['limit'])))
                    else:
                        query_for_get_c_text_list += " ORDER BY m.timestamp DESC"
                        cur.execute(query_for_get_c_text_list, diary_info["user_id"])
                result = cur.fetchall()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : get_converted_text_list() - Execution Time : %s", stop - start)
                if result:
                    logger.debug('DB RESULT : %s', result) 
                    return result
                else:
                    return None
        except pymysql.MySQLError as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At get_converted_text_list()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False


class SentenceManager(DBManager):
    def __init__(self):
        """DB Model Class for smartdiary.sentence table

        """
        DBManager.__init__(self)

    def add_sentence(self, sentence_info):
        """Adding new diary to SD DB
        Usually, this method be called
        converted text has been parsed

        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected

        if type(sentence_info['text']) is list:
            query_for_add_sentence = "INSERT INTO sentence " \
                                   "(c_text_id, text) " \
                                   "VALUES"
            for sentence_line in sentence_info['text']:
                query_for_add_sentence += " ('%s', '%s')," % (sentence_info['c_text_id'], sentence_line.replace('\'','\\\''))
            query_for_add_sentence = query_for_add_sentence[:-1]
            try:
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(query_for_add_sentence)
                    cur.execute("SELECT LAST_INSERT_ID()")
                    self.conn.commit()

                    sentence_id = cur.fetchone()['LAST_INSERT_ID()']

                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : add_sentence() - Execution Time : %s", stop - start)

                return sentence_id
            except pymysql.MySQLError as exp:
                logger.error(">>>MYSQL ERROR<<<")
                logger.error("At add_sentence()")
                num, error_msg = exp.args
                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
                return False

        else:
            query_for_add_sentence = "INSERT INTO sentence " \
                                   "(c_text_id, text) " \
                                   "VALUES (%s, %s)"
            try:
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(query_for_add_sentence, (sentence_info['c_text_id'], sentence_info['text'].replace('\'','\\\'')))
                    cur.execute("SELECT LAST_INSERT_ID()")
                    self.conn.commit()

                    sentence_id = cur.fetchone()['LAST_INSERT_ID()']

                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : add_sentence() - Execution Time : %s", stop - start)
                return sentence_id
            except pymysql.MySQLError as exp:
                logger.error(">>>MYSQL ERROR<<<")
                logger.error("At add_sentence()")
                num, error_msg = exp.args
                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
                return False


class SentElementManager(DBManager):
    def __init__(self):
        """DB Model Class for smartdiary.sentence table

        """
        DBManager.__init__(self)

    def add_s_element(self, s_element_info):
        """Adding new diary to SD DB
            Usually, this method be called
            converted text has been parsed

            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected

        if type(s_element_info) is list:
            query_for_add_s_element = "INSERT INTO s_element " \
                                     "(sentence_id, text, pos, role, element_no, ne) " \
                                     "VALUES"
            for s_element_line in s_element_info:
                query_for_add_s_element += " (%s, '%s', '%s', '%s', %s)," \
                                          % (s_element_line['sentence_id'],
                                             s_element_line['text'].replace('\'', '\\\''),
                                             s_element_line['pos'], s_element_line['role'],
                                             s_element_line['element_no'], s_element_line['ne'])
            query_for_add_s_element = query_for_add_s_element[:-1]
            try:
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(query_for_add_s_element)
                    cur.execute("SELECT LAST_INSERT_ID()")
                    self.conn.commit()

                    s_element_id = cur.fetchone()['LAST_INSERT_ID()']

                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : add_s_element(list) - Execution Time : %s", stop - start)
                return s_element_id
            except pymysql.MySQLError as exp:
                logger.error(">>>MYSQL ERROR<<<")
                logger.error("At add_s_element()")
                num, error_msg = exp.args
                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
                return False


class NLPkoDictManager(DBManager):
    def __init__(self):
        """DB Model Class for smartdiary.nlp_ko_dict & nlp_ko_dict_net table

        """
        DBManager.__init__(self)

    def retrieve_collection_dic(self, word):
        """Creating new diary to SD DB
        Usually, this method be called
        When User creating new diary

        :type diary_info: dict contains user_id, timestamp_from, timestamp_to
        :rtype query result[list] or false:
        """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected

        query = "SELECT DISTINCT means, idx from nlp_ko_dict as F INNER  JOIN (" \
                "SELECT stddicidx from nlp_ko_dict_net as B INNER JOIN (SELECT m.wordnetidx from nlp_ko_dict AS d " \
                "INNER JOIN nlp_ko_dict_net as m " \
                "ON d.idx = m.stddicidx WHERE means = %s) as R " \
                "ON B.wordnetidx = R.wordnetidx ) as SF " \
                "ON F.idx = SF.stddicidx "


        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query, word)
                result = cur.fetchall()

                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : retrieve_collection_dic() - Execution Time : %s", stop - start)
                if result:
                    logger.debug('DB RESULT : %s', result) 
                    return result
                else:
                    return None
        except pymysql.MySQLError as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_collection_dic()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

