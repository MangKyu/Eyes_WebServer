from app import app, db

if __name__ == '__main__':
    host = "192.168.43.197"
    #host = "172.16.16.115"
    app.run(host=host, debug=True, use_reloader=False, port=5000)
    db.insert_patient('a', 'b', 'c', 'd')
    image_path = db.get_image_path('a')
    print(image_path)
    #user_histories = db.get_histories('a')
    #print(user_histories)