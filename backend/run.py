import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    # threaded=True: MJPEG stream (/api/stream) bağlantıyı açık tutacağı
    # için, aynı process başka istekleri de paralel işleyebilmeli.
    # use_reloader=False: reloader ikinci bir process açıp video kaynağını
    # (kamera) iki kez açmaya çalışmasın diye kapalı.
    #
    # debug varsayılan olarak KAPALI: Werkzeug'un debug modu, hata sayfası
    # üzerinden sunucuda kod çalıştırılabilen interaktif bir konsol açar
    # (host 0.0.0.0 olduğu için bu konsol ağdaki diğer makinelere de
    # görünür olur). Geliştirirken açmak için: FLASK_DEBUG=true python run.py
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    app.run(host="0.0.0.0", port=5000, debug=debug, use_reloader=False, threaded=True)
