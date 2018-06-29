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

    def create_table(self):#, user_id, patient_id):
        try:
            with self.conn.cursor() as cursor:
                sql = 'CREATE TABLE User(userID VARCHAR(32), patientID VARCHAR(32), PRIMARY KEY(userID))'
                cursor.execute(sql)#, (user_id, patient_id))
        except Exception as e:
            print(e)

        try:
            with self.conn.cursor() as cursor:
                sql = 'CREATE TABLE Patient(patientID VARCHAR(32), patientName VARCHAR(10), patientInfo VARCHAR(100), patientImage VARCHAR(50), PRIMARY KEY(patientID))'
                cursor.execute(sql)
        except Exception as e:
            print(e)
        '''
        try:
            with self.conn.cursor() as cursor:
                sql = 'CREATE TABLE History(callDate VARCHAR(30), latitude FLOAT, longitude FLOAT, startTime VARCHAR)'
                cursor.execute(sql)
        except Exception as e:
            print(e)
        '''
        self.conn.commit()

    def get_patient_id(self, user_id):
        try:
            with self.conn.cursor() as cursor:
                sql = 'SELECT * FROM Patient WHERE userID = %s'
                cursor.execute(sql, user_id)
                return cursor.fetchone()
        except Exception as e:
            print(e)


    def get_patient_info(self, patient_id):
        try:
            with self.conn.cursor() as cursor:
                sql = 'SELECT * FROM Patient WHERE userID = %s'
                cursor.execute(sql, patient_id)
                return cursor.fetchone()
        except Exception as e:
            print(e)

    def insert_user(self, user_id, patient_id):
        try:
            with self.conn.cursor() as cursor:
                sql = 'INSERT INTO User(userID, patientID) VALUES (%s, %s)'
                cursor.execute(sql, (user_id, patient_id))
        except Exception as e:
            print(e)

    def insert_patient(self, patient_id, patient_name, patient_info, patient_image):
        with self.conn.cursor() as cursor:
            try:
                sql = 'INSERT INTO Patient(patientID, patientName, patientInfo, patientImage) VALUES (%s, %s, %s, %s)'
                cursor.execute(sql, (patient_id, patient_name, patient_info, patient_image))
                self.conn.commit()
            except Exception as e:
                print(e)

    def get_image_path(self, patient_id):
        with self.conn.cursor() as cursor:
            try:
                sql = 'SELECT patientImage FROM Patient WHERE patientID = %s'
                cursor.execute(sql, patient_id)
                return cursor.fetchone()[0]
            except Exception as e:
                print(e)

    def update_image_path(self, patient_image, patient_id):
        with self.conn.cursor() as cursor:
            try:
                sql = 'UPDATE Patient SET patientImage = %s WHERE patientID = %s'
                cursor.execute(sql, (patient_image, patient_id))
                self.conn.commit()
            except Exception as e:
                print(e)

    def get_histories(self, patient_id):
        with self.conn.cursor() as cursor:
            try:
                sql = 'SELECT * FROM History WHERE patientID = %s'
                cursor.execute(sql, patient_id)
                return cursor.fetchall()
            except Exception as e:
                print(e)
