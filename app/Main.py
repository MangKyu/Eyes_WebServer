from app import app

if __name__ == '__main__':
    app.run(host='localhost', debug=True, use_reloader=False, port=6000)