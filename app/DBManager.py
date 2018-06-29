import pymysql
import datetime
import os

# class for connection to DataBase
class DBConnection:
    conn = None
    cursor = None

    def __init__(self):
        # Connect to database
        self.conn = pymysql.connect(host='localhost', user='root', password='2',
                                    db='eyes', charset='utf8')

        # From conn, make Dictionary Cursor
        self.curs = self.conn.cursor(pymysql.cursors.DictCursor)
        self.create_table()

    def create_table(self):
        try:
            with self.conn.cursor() as cursor:
                sql = 'CREATE TABLE Patient(userID VARCHAR(32), userName VARCHAR(10), userInfo VARCHAR(100), userImage VARCHAR(50), PRIMARY KEY(userID))'
                cursor.execute(sql)
        except Exception as e:
            print(e)

        try:
            with self.conn.cursor() as cursor:
                sql = 'CREATE TABLE History(callDate VARCHAR(30), latitude FLOAT, longitude FLOAT, startTime VARCHAR)'
                #startTIme, endTime, taker
                cursor.execute(sql)
        except Exception as e:
            print(e)
        self.conn.commit()

    def insert_patient(self, user_id, user_name, user_info, user_image):
        with self.conn.cursor() as cursor:
            try:
                sql = 'INSERT INTO Patient(userID, userName, userInfo, userImage) VALUES (%s, %s, %s, %s)'
                cursor.execute(sql, (user_id, user_name, user_info, user_image))
                self.conn.commit()
            except Exception as e:
                print(e)

    def get_image_path(self, user_id):
        with self.conn.cursor() as cursor:
            try:
                sql = 'SELECT userImage FROM Patient WHERE userID = %s'
                cursor.execute(sql, user_id)
                return cursor.fetchone()[0]
            except Exception as e:
                print(e)

    def update_image_path(self, user_image, user_id):
        with self.conn.cursor() as cursor:
            try:
                sql = 'UPDATE Patient SET userImage = %s WHERE userID = %s'
                cursor.execute(sql, (user_image, user_id))
                self.conn.commit()
            except Exception as e:
                print(e)

    def get_histories(self, user_id):
        with self.conn.cursor() as cursor:
            try:
                sql = 'SELECT * FROM History WHERE userID = %s'
                cursor.execute(sql, user_id)
                return cursor.fetchall()
            except Exception as e:
                print(e)
