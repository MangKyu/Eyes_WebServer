from app import app, db
from app.Model import PatientVO
if __name__ == '__main__':
    host = "192.168.43.197"
    #host = "172.16.16.115"
    app.run(host=host, debug=True, use_reloader=False, port=5000)

    '''start_x = 126.9850380932383
    start_y = 37.566567545861645
    db.insert_history('aaaaaaaaaaaaaaaa', 'b', 'c', 'd', start_x, start_y)
    db.insert_history('aaaaaaaaaaaaaaaa', 'e', 'f', 'g', start_x, start_y)
    db.insert_history('aaaaaaaaaaaaaaaa', 'h', 'i', 'j', start_x, start_y)
    histories = db.get_histories('aaaaaaaaaaaaaaaa')
    print(history_list)
    print(history_list)
    '''
    '''
    user_id = 'c5aff6fe207207a4'

    patient_id = db.get_patient_id(user_id)
    patient_info = db.get_patient_info(patient_id)
    patient_vo = PatientVO.PatientVO(patient_info)
'''
    #print(image_path)
    #user_histories = db.get_histories('a')
    #print(user_histories)