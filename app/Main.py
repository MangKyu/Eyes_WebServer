from app import app

if __name__ == '__main__':
    host = "172.16.16.115"
    app.run(host=host, debug=True, use_reloader=False, port=5000)