from app import app, db
from app.Model import PatientVO
if __name__ == '__main__':
    host = "192.168.43.197"
    #host = "172.16.16.115"
    app.run(host=host, debug=True, use_reloader=False, port=5000)
