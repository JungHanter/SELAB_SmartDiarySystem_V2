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
        """DB Connection Class for Smart audio_diary (SD) Database

        """
        try:
            self.conn = pymysql.connect(host='203.253.23.17', port=3306, user='root', passwd='lovejesus',
                                        db='new_smartdiary', charset='utf8', use_unicode=True)
            self.connected = True
        except pymysql.Error:
            self.connected = False


class UserManager(DBManager):
    def __init__(self):
        """DB Model Class for smaryaudio_diary.user table

        """
        DBManager.__init__(self)

    def create_user(self, user_id, password, name, birthday, gender, email, phone):
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
        query_for_create_user = "INSERT INTO user " \
                             "(user_id, password, name, birthday, gender, email, phone) " \
                             "VALUES (%s, %s, %s, %s, %s, %s ,%s)"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_create_user,
                            (user_id, password, name, birthday, gender, email, phone))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : create_user() - Execution Time : %s", stop - start)
            return True

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At create_user()")
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

    def update_user(self, user_id, name, password, gender, email, phone, birthday):
        """retrieving new user from H.I.S DB
            Usually, this method be called
            When User Information need to be display

            :param user_id: id of user of Retrieving Target
            :rtype: dict contains user's inforamtion
            """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected  # Connection Check Flag
        query_for_update_user = "UPDATE user SET name = %s, birthday = %s, gender = %s, password = %s," \
                                " email = %s, phone = %s WHERE user_id = %s"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                affected_rows = cur.execute(query_for_update_user, (name, birthday,
                                                    gender, password, email, phone,
                                                    user_id))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : update_user() - Execution Time : %s", stop - start)
                logger.debug("DB : AFFECTED ROWS : %s rows", affected_rows)
                return cur.fetchone()

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At update_user()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)

    def delete_user(self, user_id):
        """retrieving new user from H.I.S DB
            Usually, this method be called
            When User Information need to be display

            :param user_id: id of user of Retrieving Target
            :rtype: dict contains user's inforamtion
            """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected  # Connection Check Flag
        query_for_delete_user = "DELETE FROM user WHERE user_id = %s "
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                affected_rows = cur.execute(query_for_delete_user, user_id)
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : delete_user() - Execution Time : %s", stop - start)
                logger.debug("DB : AFFECTED ROWS : %s rows", affected_rows)
                return True

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At delete_user()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False


class AudioDiaryManager(DBManager):
    def __init__(self):
        """DB Model Class for smaryaudio_diary.audio_diary table

        """
        DBManager.__init__(self)

    def create_audio_diary(self, user_id, title, created_date):
        """Creating new audio_diary to SD DB
        Usually, this method be called
        When User creating new audio_diary


        :param audio_diary_info:
        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        query_for_create_audio_diary = "INSERT INTO audio_diary " \
                             "(user_id, title, created_date) " \
                             "VALUES (%s, %s, %s)"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_create_audio_diary,
                            (user_id, title, created_date))
                cur.execute("SELECT LAST_INSERT_ID()")
                self.conn.commit()
                audio_diary_id = cur.fetchone()['LAST_INSERT_ID()']
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : create_audio_diary() - Execution Time : %s", stop - start)
            return audio_diary_id

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At create_audio_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_audio_diary_by_user_id(self, user_id):
        """Creating new audio_diary to SD DB
        Usually, this method be called
        When User retrieving audio_diary


        :param audio_diary_info:
        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_retrieve_audio_diary = "SELECT * FROM audio_diary WHERE user_id = %s  "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_retrieve_audio_diary, user_id)
                result = cur.fetchall()
                if result:
                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : retrieve_audio_diary_by_user_id() - Execution Time : %s", stop - start)
                    logger.debug('DB RESULT : %s', result) 
                    return result
                else:
                    return False

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_audio_diary_by_user_id()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_audio_diary_list_by_timestamp(self, audio_diary_info):
        """Creating new audio_diary to SD DB
        Usually, this method be called
        When User creating new audio_diary
                                    "ON m.audio_diary_id = d.audio_diary_id " \



        :type audio_diary_info: dict contains user_id, timestamp_from, timestamp_to
        :rtype query result[list] or false:
        """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected
        query_for_get_c_text_list = "SELECT d.content, m.* FROM audio_diary as m " \
                                    "INNER JOIN text_diary AS d USING(audio_diary_id)" \
                                    "WHERE m.user_id = %s"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                if 'timestamp_from' in audio_diary_info and 'timestamp_to' in audio_diary_info:
                    query_for_get_c_text_list += " AND m.created_date>=%s AND m.created_date<=%s ORDER BY m.created_date DESC"
                    cur.execute(query_for_get_c_text_list,
                                (audio_diary_info["user_id"], audio_diary_info["timestamp_from"],
                                 audio_diary_info["timestamp_to"]))

                elif 'timestamp_from' in audio_diary_info:
                    query_for_get_c_text_list += " AND m.created_date>=%s ORDER BY m.created_date DESC"
                    cur.execute(query_for_get_c_text_list,
                                (audio_diary_info["user_id"], audio_diary_info["timestamp_from"]))

                elif 'timestamp_to' in audio_diary_info:
                    query_for_get_c_text_list += " AND m.created_date<=%s ORDER BY m.created_date DESC"
                    cur.execute(query_for_get_c_text_list,
                                (audio_diary_info["user_id"], audio_diary_info["timestamp_to"]))
                else:
                    query_for_get_c_text_list += " ORDER BY m.created_date DESC"
                    cur.execute(query_for_get_c_text_list, audio_diary_info["user_id"])
                result = cur.fetchall()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : get_text_diary_list() - Execution Time : %s", stop - start)
                if result:
                    logger.debug('DB RESULT : %s', result)
                    return result
                else:
                    logger.debug('DB RESULT : %s', result)
                    return []
        except pymysql.MySQLError as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At get_text_diary_list()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_audio_diary_detail_by_audio_diary_id(self, user_id, audio_diary_id, timestamp_from=None, timestamp_to=None):
        """Creating new audio_diary to SD DB
        Usually, this method be called
        When User retrieving audio_diary


        :param audio_diary_info:
        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_audio_diary = "SELECT audio_diary.*, text_diary.content " \
                                     "FROM audio_diary, text_diary " \
                                     "WHERE audio_diary.audio_diary_id = %s AND audio_diary.user_id = %s " \
                                     "AND audio_diary.audio_diary_id = text_diary.audio_diary_id "

            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_audio_diary, (audio_diary_id, user_id))
                audio_diary_result = cur.fetchone()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : retrieve_audio_diary_detail_by_audio_diary_id() - Execution Time : %s", stop - start)
            if audio_diary_result:
                return audio_diary_result
            else:
                return None

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_audio_diary_detail_by_audio_diary_id()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_sentence_list_from_audio_diary(self, audio_diary_id):
        """Creating new audio_diary to SD DB
            Usually, this method be called
            When User retrieving audio_diary


            :param audio_diary_info:
            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_list = "SELECT sentence.text FROM sentence INNER JOIN text_diary " \
                             "WHERE text_diary.audio_diary_id = %s "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_list, audio_diary_id)
                result = cur.fetchall()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : retrieve_sentence_list_from_audio_diary() - Execution Time : %s", stop - start)

                if result:
                    logger.debug('DB RESULT : %s', result)

                    return result
                else:
                    return None

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrive_setence_list_from_audio_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_audio_diary_by_title(self, audio_diary_info):
        """Creating new audio_diary to SD DB
            Usually, this method be called
            When User retrieving audio_diary


            :param audio_diary_info:
            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_list = "SELECT audio_diary.audio_diary_id, audio_diary.user_id, audio_diary.title, audio_diary.timestamp, audio_diary.annotation, " \
                              "IF(audio, 'True', 'False') as audio, text_diary.text, audio_diary.location FROM audio_diary, text_diary " \
                             "WHERE audio_diary.audio_diary_id = text_diary.audio_diary_id AND text_diary.text LIKE %s " \
                             "AND audio_diary.user_id = %s "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                if 'limit' in audio_diary_info:
                    if 'offset' in audio_diary_info:
                        query_for_list += " ORDER BY audio_diary.timestamp DESC LIMIT %s OFFSET %s"
                        cur.execute(query_for_list, ('%'+audio_diary_info['keyword']+'%', audio_diary_info["user_id"],
                                                     int(audio_diary_info['limit']), int(audio_diary_info['offset'])))
                    else:
                        query_for_list += " ORDER BY audio_diary.timestamp DESC LIMIT %s"
                        cur.execute(query_for_list, ('%'+audio_diary_info['keyword']+'%', audio_diary_info["user_id"],
                                                     int(audio_diary_info['limit'])))
                else:
                    query_for_list += " ORDER BY audio_diary.timestamp DESC"
                    cur.execute(query_for_list, ('%'+audio_diary_info['keyword']+'%', audio_diary_info["user_id"]))

                result = cur.fetchall()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : retrieve_audio_diary_with_keyword() - Execution Time : %s", stop - start)
                if result:

                    logger.debug('DB RESULT : %s', result) 
                    return result
                else:
                    return None

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_audio_diary_with_keyword()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def update_audio_diary(self, audio_diary_info):
        """Creating new audio_diary to SD DB
            Usually, this method be called
            When User retrieving audio_diary


            :param audio_diary_info:
            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_delete = "UPDATE audio_diary SET title = %s, created_date = %s WHERE audio_diary_id = %s "

            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                affected_rows = cur.execute(query_for_delete, (audio_diary_info['title'], audio_diary_info['created_date'], int(audio_diary_info['audio_diary_id'])))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : update_audio_diary() - Execution Time : %s", stop - start)
                logger.debug("DB : AFFECTED ROWS : %s rows", affected_rows)
                return True

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At update_audio_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def delete_audio_diary(self, audio_diary_id):
        """Creating new audio_diary to SD DB
        Usually, this method be called
        When User retrieving audio_diary


        :param audio_diary_info:
        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_delete = "DELETE FROM audio_diary WHERE audio_diary_id  = %s "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                affected_rows = cur.execute(query_for_delete, int(audio_diary_id))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : delete_audio_diary() - Execution Time : %s", stop - start)
                logger.debug("DB : AFFECTED ROWS : %s rows", affected_rows)
            return True

        except Exception as exp:
            logger.exception(exp)
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At delete_audio_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def delete_c_text_from_audio_diary(self, audio_diary_id):
        """Creating new audio_diary to SD DB
        Usually, this method be called
        When User retrieving audio_diary


        :param audio_diary_info:
        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_delete = "DELETE FROM text_diary WHERE audio_diary_id  = %s "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_delete, int(audio_diary_id))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : delete_c_text_from_audio_diary() - Execution Time : %s", stop - start)
            return True

        except Exception as exp:
            logger.exception(exp)
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At delete_c_text_from_audio_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def update_pickle_state(self, audio_diary_id, state):
        """Creating new audio_diary to SD DB
            Usually, this method be called
            When User retrieving audio_diary


            :param audio_diary_info:
            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_updated = "UPDATE audio_diary SET pickle= %s WHERE audio_diary_id = %s "

            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                affected_rows = cur.execute(query_for_updated, (state, audio_diary_id))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : update_pickle_state() - Execution Time : %s", stop - start)
                logger.debug("DB : AFFECTED ROWS : %s rows", affected_rows)
                return True

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At update_audio_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def update_lifestyle_analyzed_state(self, audio_diary_id, type_updated = None):
        """Creating new audio_diary to SD DB
            Usually, this method be called
            When User retrieving audio_diary


            :param audio_diary_info:
            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_updated = "UPDATE audio_diary SET lifestyle_analyzed= %s WHERE audio_diary_id =%s; "
            if type(audio_diary_id) is list:
                affected_rows = 0
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    for adi in audio_diary_id:
                        affected_rows = affected_rows + cur.execute(query_for_updated, (adi['lifestyle_analyzed'], adi['audio_diary_id']))
                    self.conn.commit()
                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : update_lifstyle_analyze_state() - Execution Time : %s", stop - start)
                    logger.debug("DB : AFFECTED ROWS : %s rows", affected_rows)
                    return True
                # query_for_updated = query_for_updated + ' IN ('
                # for adi in audio_diary_id:
                #     query_for_updated = query_for_updated + str(adi['audio_diary_id']) + ','
                # query_for_updated = query_for_updated[:-1]
                # query_for_updated = query_for_updated + ')'



            else:
                # query_for_updated = query_for_updated + '= %s '
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    affected_rows = cur.execute(query_for_updated, (type_updated, audio_diary_id))
                    self.conn.commit()
                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : update_lifstyle_analyze_state() - Execution Time : %s", stop - start)
                    logger.debug("DB : AFFECTED ROWS : %s rows", affected_rows)
                    return True

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At update_lifstyle_analyze_state()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_state_flags(self, audio_diary_id):
        """Creating new audio_diary to SD DB
        Usually, this method be called
        When User retrieving audio_diary


        :param audio_diary_info:
        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_retrieve_audio_diary = "SELECT pickle, lifestyle_analyze FROM audio_diary WHERE audio_diary_id = %s  "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_retrieve_audio_diary, audio_diary_id)
                result = cur.fetchone()
                if result:
                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : retrieve_pickle_state() - Execution Time : %s", stop - start)
                    logger.debug('DB RESULT : %s', result)
                    return result
                else:
                    return None

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_pickle_state()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False


class TextDiaryManager(DBManager):
    def __init__(self):
        """DB Model Class for smaryaudio_diary.text_diary table

        """
        DBManager.__init__(self)

    def create_text_diary(self, audio_diary_id, content, created_date):
        """Adding new audio_diary to SD DB
        Usually, this method be called
        When User creating new audio_diary

        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected
        query_for_create_c_text = "INSERT INTO text_diary " \
                             "(audio_diary_id, content, created_date) " \
                             "VALUES (%s, %s, %s) "
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_create_c_text, (audio_diary_id, content, created_date))
                cur.execute("SELECT LAST_INSERT_ID()")
                self.conn.commit()

                text_diary_id = cur.fetchone()['LAST_INSERT_ID()']

                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : create_text_diary() - Execution Time : %s", stop - start)
            return text_diary_id
        except pymysql.MySQLError as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At create_text_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_text_diary_by_audio_diary_id(self, audio_diary_id):
        """retrieving converted text from SD DB
        Usually, this method be called
        When ...

        :param text_diary_id:
        :rtype: dict contains user's inforamtion
        """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected  # Connection Check Flag
        query_for_c_text = "SELECT * FROM text_diary WHERE audio_diary_id = %s"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_c_text, audio_diary_id)
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : get_text_diary() - Execution Time : %s", stop - start)
                result = cur.fetchone()
                if result:
                    logger.debug('DB RESULT : %s', result) 
                    return result
                else:
                    return None
        except pymysql.MySQLError as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At get_text_diary()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_text_diary_list(self, audio_diary_info):
        """Creating new audio_diary to SD DB
        Usually, this method be called
        When User creating new audio_diary



        :type audio_diary_info: dict contains user_id, timestamp_from, timestamp_to
        :rtype query result[list] or false:
        """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected
        query_for_get_c_text_list = "SELECT d.content, d.audio_diary_id, m.created_date, m.title FROM text_diary AS d " \
                                    "INNER JOIN audio_diary as m " \
                                    "ON m.audio_diary_id = d.audio_diary_id " \
                                    "WHERE m.user_id = %s"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                if 'timestamp_from' in audio_diary_info and 'timestamp_to' in audio_diary_info:
                    if 'limit' in audio_diary_info:
                        if 'offset' in audio_diary_info:
                            query_for_get_c_text_list += " AND m.created_date>=%s AND m.created_date<=%s ORDER BY m.created_date DESC LIMIT %s OFFSET %s"
                            cur.execute(query_for_get_c_text_list, (audio_diary_info["user_id"], audio_diary_info["timestamp_from"],
                                        audio_diary_info["timestamp_to"], int(audio_diary_info['limit']),
                                        int(audio_diary_info['offset'])))
                        else:
                            query_for_get_c_text_list += " AND m.created_date>=%s AND m.created_date<=%s ORDER BY m.created_date DESC LIMIT %s"
                            cur.execute(query_for_get_c_text_list, (audio_diary_info["user_id"], audio_diary_info["timestamp_from"],
                                                                    audio_diary_info["timestamp_to"], int(audio_diary_info['limit'])))
                    else:
                        query_for_get_c_text_list += " AND m.created_date>=%s AND m.created_date<=%s ORDER BY m.created_date DESC"
                        cur.execute(query_for_get_c_text_list, (audio_diary_info["user_id"], audio_diary_info["timestamp_from"],
                                                                audio_diary_info["timestamp_to"]))
                elif 'timestamp_from' in audio_diary_info:
                    if 'limit' in audio_diary_info:
                        if 'offset' in audio_diary_info:
                            query_for_get_c_text_list += " AND m.created_date>=%s ORDER BY m.created_date DESC LIMIT %s OFFSET %s"
                            cur.execute(query_for_get_c_text_list, (audio_diary_info["user_id"], audio_diary_info["timestamp_from"],
                                        int(audio_diary_info['limit']), int(audio_diary_info['offset'])))
                        else:
                            query_for_get_c_text_list += " AND m.created_date>=%s ORDER BY m.created_date DESC LIMIT %s"
                            cur.execute(query_for_get_c_text_list,
                                        (audio_diary_info["user_id"], audio_diary_info["timestamp_from"],
                                        int(audio_diary_info['limit'])))
                    else:
                        query_for_get_c_text_list += " AND m.created_date>=%s ORDER BY m.created_date DESC"
                        cur.execute(query_for_get_c_text_list, (audio_diary_info["user_id"], audio_diary_info["timestamp_from"]))

                elif 'timestamp_to' in audio_diary_info:
                    if 'limit' in audio_diary_info:
                        if 'offset' in audio_diary_info:
                            query_for_get_c_text_list += " AND m.created_date<=%s ORDER BY m.created_date DESC LIMIT %s OFFSET %s"
                            cur.execute(query_for_get_c_text_list, (audio_diary_info["user_id"], audio_diary_info["timestamp_to"],
                                        int(audio_diary_info['limit']), int(audio_diary_info['offset'])))
                        else:
                            query_for_get_c_text_list += " AND m.created_date<=%s ORDER BY m.created_date DESC LIMIT %s"
                            cur.execute(query_for_get_c_text_list, (audio_diary_info["user_id"], audio_diary_info["timestamp_to"],
                                        int(audio_diary_info['limit'])))
                    else:
                        query_for_get_c_text_list += " AND m.created_date<=%s ORDER BY m.created_date DESC"
                        cur.execute(query_for_get_c_text_list, (audio_diary_info["user_id"], audio_diary_info["timestamp_to"]))
                else:
                    if 'limit' in audio_diary_info:
                        if 'offset' in audio_diary_info:
                            query_for_get_c_text_list += " ORDER BY m.created_date DESC LIMIT %s OFFSET %s"
                            cur.execute(query_for_get_c_text_list, (audio_diary_info["user_id"], int(audio_diary_info['limit']),
                                                                    int(audio_diary_info['offset'])))
                        else:
                            query_for_get_c_text_list += " ORDER BY m.created_date DESC LIMIT %s"
                            cur.execute(query_for_get_c_text_list, (audio_diary_info["user_id"], int(audio_diary_info['limit'])))
                    else:
                        query_for_get_c_text_list += " ORDER BY m.created_date DESC"
                        cur.execute(query_for_get_c_text_list, audio_diary_info["user_id"])
                result = cur.fetchall()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : get_text_diary_list() - Execution Time : %s", stop - start)
                if result:
                    logger.debug('DB RESULT : %s', result) 
                    return result
                else:
                    logger.debug('DB RESULT : %s', result)
                    return []
        except pymysql.MySQLError as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At get_text_diary_list()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def delete_text_diary_by_audio_diary_id(self, audio_diary_id):
        """retrieving new user from H.I.S DB
        Usually, this method be called
        When User Information need to be display

        :param user_id: id of user of Retrieving Target
        :rtype: dict contains user's inforamtion
        """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected  # Connection Check Flag
        query_for_delete_text_diary = "DELETE FROM text_diary WHERE audio_diary_id = %s "
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_delete_text_diary, audio_diary_id)
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : delete_text_diary_by_audio_diary_id() - Execution Time : %s", stop - start)
                return True

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At delete_text_diary_by_audio_diary_id()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False


class DiaryContextManager(DBManager):
    def __init__(self):
        """DB Model Class for smartaudio_diary.sentence table

        """
        DBManager.__init__(self)

    def create_diary_context(self, audio_diary_id, diary_context_info):
        """Adding new audio_diary to SD DB
            Usually, this method be called
            converted text has been parsed

            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        if type(diary_context_info) is list:
            query_for_create_diary_context = "INSERT INTO diary_context " \
                                        "(audio_diary_id, type, subtype, value, date_added) " \
                                        "VALUES"
            for d_item in diary_context_info:
                query_for_create_diary_context += " (%s, '%s', '%s', '%s', %s)," % (
                    audio_diary_id, d_item['type'], d_item['subtype'], d_item['value'], int(d_item['date_added']))
            query_for_create_diary_context = query_for_create_diary_context[:-1]
            try:
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(query_for_create_diary_context)
                    cur.execute("SELECT LAST_INSERT_ID()")
                    self.conn.commit()

                    diary_context_id = cur.fetchone()['LAST_INSERT_ID()']

                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : create_diary_context() - Execution Time : %s", stop - start)

                return diary_context_id
            except pymysql.MySQLError as exp:
                logger.error(">>>MYSQL ERROR<<<")
                logger.error("At create_diary_context()")
                num, error_msg = exp.args
                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
                return False
        else:
            query_for_create_analytics = "INSERT INTO diary_context " \
                                     "(audio_diary_id, type, subtype, value, date_added) " \
                                     "VALUES (%s, %s, %s, %s, %s)"
            try:
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(query_for_create_analytics, (diary_context_info['audio_diary_id'],
                                                             diary_context_info['type'],
                                                             diary_context_info['subtype'],
                                                             diary_context_info['value'],
                                                             diary_context_info['date_added']))
                    self.conn.commit()

                    diary_context_id = cur.fetchone()['LAST_INSERT_ID()']

                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : create_analytics() - Execution Time : %s", stop - start)
                return diary_context_id
            except pymysql.MySQLError as exp:
                logger.error(">>>MYSQL ERROR<<<")
                logger.error("At create_analytics()")
                num, error_msg = exp.args
                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
                return False

    def retrieve_diary_context_by_audio_diary_id(self, audio_diary_id):
        """Creating new audio_diary to SD DB
            Usually, this method be called
            When User retrieving audio_diary


            :param audio_diary_info:
            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_retrieve_audio_diary = "SELECT diary_context_id, type, subtype, value, date_added FROM diary_context WHERE audio_diary_id = %s  "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_retrieve_audio_diary, audio_diary_id)
                result = cur.fetchall()
                if result:
                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : retrieve_diary_context_by_audio_diary_id() - Execution Time : %s", stop - start)
                    logger.debug('DB RESULT : %s', result)
                    return result
                else:
                    return []

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_diary_context_by_audio_diary_id()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def delete_diary_context_by_audio_diary_id(self, audio_diary_id):
        """Creating new audio_diary to SD DB
            Usually, this method be called
            When User retrieving audio_diary


            :param audio_diary_info:
            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        try:
            query_for_delete = "DELETE FROM diary_context WHERE audio_diary_id  = %s "
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                affected_rows = cur.execute(query_for_delete, int(audio_diary_id))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : delete_diary_context_by_audio_diary_id() - Execution Time : %s", stop - start)
                logger.debug("DB : AFFECTED ROWS : %s rows", affected_rows)
                return True

        except Exception as exp:
            logger.exception(exp)
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At delete_diary_context_by_audio_diary_id()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False


class SentenceManager(DBManager):
    def __init__(self):
        """DB Model Class for smartaudio_diary.sentence table

        """
        DBManager.__init__(self)

    def create_sentence(self, sentence_info):
        """Adding new audio_diary to SD DB
        Usually, this method be called
        converted text has been parsed

        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected

        if type(sentence_info['text']) is list:
            query_for_create_sentence = "INSERT INTO sentence " \
                                   "(text_diary_id, text) " \
                                   "VALUES"
            for sentence_line in sentence_info['text']:
                query_for_create_sentence += " ('%s', '%s')," % (sentence_info['text_diary_id'], sentence_line.replace('\'','\\\''))
            query_for_create_sentence = query_for_create_sentence[:-1]
            try:
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(query_for_create_sentence)
                    cur.execute("SELECT LAST_INSERT_ID()")
                    self.conn.commit()

                    sentence_id = cur.fetchone()['LAST_INSERT_ID()']

                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : create_sentence() - Execution Time : %s", stop - start)

                return sentence_id
            except pymysql.MySQLError as exp:
                logger.error(">>>MYSQL ERROR<<<")
                logger.error("At create_sentence()")
                num, error_msg = exp.args
                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
                return False

        else:
            query_for_create_sentence = "INSERT INTO sentence " \
                                   "(text_diary_id, text) " \
                                   "VALUES (%s, %s)"
            try:
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(query_for_create_sentence, (sentence_info['text_diary_id'], sentence_info['text'].replace('\'','\\\'')))
                    cur.execute("SELECT LAST_INSERT_ID()")
                    self.conn.commit()

                    sentence_id = cur.fetchone()['LAST_INSERT_ID()']

                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : create_sentence() - Execution Time : %s", stop - start)
                return sentence_id
            except pymysql.MySQLError as exp:
                logger.error(">>>MYSQL ERROR<<<")
                logger.error("At create_sentence()")
                num, error_msg = exp.args
                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
                return False


class SentenceElementManager(DBManager):
    def __init__(self):
        """DB Model Class for smartaudio_diary.sentence table

        """
        DBManager.__init__(self)

    def create_s_element(self, s_element_info):
        """Adding new audio_diary to SD DB
        Usually, this method be called
        converted text has been parsed

        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected

        if type(s_element_info) is list:
            query_for_create_s_element = "INSERT INTO s_element " \
                                     "(sentence_id, text, pos, role, element_no, ne) " \
                                     "VALUES"
            for s_element_line in s_element_info:
                query_for_create_s_element += " (%s, '%s', '%s', '%s', %s, '%s')," \
                                          % (s_element_line['sentence_id'],
                                             s_element_line['text'].replace('\'', '\\\''),
                                             s_element_line['pos'], s_element_line['role'],
                                             s_element_line['element_no'], s_element_line['ne'])
            query_for_create_s_element = query_for_create_s_element[:-1]
            try:
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(query_for_create_s_element)
                    cur.execute("SELECT LAST_INSERT_ID()")
                    self.conn.commit()

                    s_element_id = cur.fetchone()['LAST_INSERT_ID()']

                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : create_s_element(list) - Execution Time : %s", stop - start)
                return s_element_id
            except pymysql.MySQLError as exp:
                logger.error(">>>MYSQL ERROR<<<")
                logger.error("At create_s_element()")
                num, error_msg = exp.args
                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
                return False


class AnalyticsManager(DBManager):
    def __init__(self):
        """DB Model Class for smartaudio_diary.sentence table

        """
        DBManager.__init__(self)

    def create_analytics(self, user_id, audio_diary_id, t_type, value, created_date):
        """Adding new audio_diary to SD DB
            Usually, this method be called
            converted text has been parsed

            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        query_for_create_analytics = "INSERT INTO analytics " \
                                 "(user_id, audio_diary_id, type, value, created_date) " \
                                 "VALUES (%s, %s, %s, %s, %s)"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_create_analytics,
                            (user_id, audio_diary_id, t_type, value, created_date))
                self.conn.commit()

                sentence_id = cur.fetchone()['LAST_INSERT_ID()']

                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : create_analytics() - Execution Time : %s", stop - start)
            return sentence_id
        except pymysql.MySQLError as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At create_analytics()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def retrieve_analytics(self, user_id, audio_diary_id, t_type=None):
        pass


class LifeStyleManager(DBManager):
    def __init__(self):
        """DB Model Class for smartaudio_diary.sentence table

        """
        DBManager.__init__(self)

    def create_lifestyle(self, audio_diary_id, thing_type, thing, score):
        """Adding new audio_diary to SD DB
            Usually, this method be called
            converted text has been parsed

            :rtype None:
            """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected
        query_for_create_lifestyle = "INSERT INTO lifestyle " \
                                     "(audio_diary_id, thing_type, thing, score) " \
                                     "VALUES (%s, %s, %s, %s)"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_create_lifestyle,
                            (audio_diary_id, thing_type, thing, score))
                self.conn.commit()

                sentence_id = cur.fetchone()['LAST_INSERT_ID()']

                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : create_lifestyle() - Execution Time : %s", stop - start)
            return sentence_id
        except pymysql.MySQLError as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At create_lifestyle()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False

    def create_lifestyle_by_list(self, ls_list):
        """Adding new audio_diary to SD DB
        Usually, this method be called
        converted text has been parsed

        :rtype None:
        """
        # START : for calculating execution time
        start = timeit.default_timer()

        assert self.connected

        if type(ls_list) is list:
            query_for_ls = "INSERT INTO lifestyle " \
                                         "(audio_diary_id, thing_type, thing, score) " \
                                         "VALUES"
            for ls in ls_list:
                query_for_ls += " (%s, '%s', '%s', %s)," \
                                              % (ls['audio_diary_id'],
                                                 ls['thing_type'],
                                                 ls['thing'], ls['score'])
            query_for_ls = query_for_ls[:-1]
            try:
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(query_for_ls)
                    cur.execute("SELECT LAST_INSERT_ID()")
                    self.conn.commit()

                    s_element_id = cur.fetchone()['LAST_INSERT_ID()']

                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : create_lifestyle_by_list(list) - Execution Time : %s", stop - start)
                return s_element_id
            except pymysql.MySQLError as exp:
                logger.error(">>>MYSQL ERROR<<<")
                logger.error("At create_lifestyle_by_list()")
                num, error_msg = exp.args
                logger.error("ERROR NO : %s", num)
                logger.error("ERROR MSG : %s", error_msg)
                return False

    def retrieve_lifestyle_with_period(self, thing_type, timestamp_from, timestamp_to):
        pass

    def retrieve_lifestyle(self, audio_diary_id, thing_type):
        """retrieving converted text from SD DB
        Usually, this method be called
        When ...

        :param text_diary_id:
        :rtype: dict contains user's inforamtion
        """
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected  # Connection Check Flag
        query_for_lifestyle = "SELECT * FROM lifestyle WHERE audio_diary_id "
        try:
            if type(audio_diary_id) is list:
                query_for_lifestyle = query_for_lifestyle + 'IN('
                for a_d_id in audio_diary_id:
                    query_for_lifestyle = query_for_lifestyle + str(a_d_id) + ','
                query_for_lifestyle = query_for_lifestyle[:-1]
                query_for_lifestyle = query_for_lifestyle + ')'
                query_for_lifestyle = query_for_lifestyle + ' AND thing_type = %s'
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(query_for_lifestyle, thing_type)
                    stop = timeit.default_timer()
                    logger.debug("DB : retrieve_lifestyle() - Execution Time : %s", stop - start)
                    result = cur.fetchall()
                    if result:
                        logger.debug('DB RESULT : %s', result)
                        return result
                    else:
                        return None
            else:
                with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                    query_for_lifestyle = query_for_lifestyle + ' = %s AND thing_type = %s'
                    cur.execute(query_for_lifestyle, (audio_diary_id, thing_type))
                    # END : for calculating execution time
                    stop = timeit.default_timer()
                    logger.debug("DB : retrieve_lifestyle() - Execution Time : %s", stop - start)
                    result = cur.fetchall()
                    if result:
                        logger.debug('DB RESULT : %s', result)
                        return result
                    else:
                        return None
        except pymysql.MySQLError as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_lifestyle()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return False



