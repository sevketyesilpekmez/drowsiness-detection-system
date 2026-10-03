"""
Testleri ortam değişkenlerinden (özellikle .env dosyasından) izole eder.

README'nin "Yerel geliştirme" bölümü `cp ../.env.example .env` diyor ve
.env.example'ın ENABLE_DETECTION_ENGINE varsayılanı GERÇEK çalıştırma için
bilerek "true". Bir geliştirici bu adımı izleyip aynı backend/ dizininde
`pytest` çalıştırırsa, app/config.py'deki load_dotenv() bu .env dosyasını
okur ve ENABLE_DETECTION_ENGINE=true os.environ'a sızar — bu da testlerin
gerçek kamerayı açmaya çalışmasına yol açar (kamera yoksa CAMERA_ERROR,
ilgili testler bunu beklemediği için patlar). Bu proje boyunca "testler
asla donanıma dokunmaz" en temel invariant olduğu için bu bir regresyon.

python-dotenv'in load_dotenv() varsayılanı override=False'tur: os.environ
içinde bir anahtar ZATEN varsa .env'deki değeri yok sayar. Bu yüzden bu
dosya (pytest tarafından her test modülünden ÖNCE, en tepede import
edilir) ENABLE_DETECTION_ENGINE'i açıkça "false" yaparak .env'in içeriği
ne olursa olsun testleri izole eder.
"""

import os

os.environ["ENABLE_DETECTION_ENGINE"] = "false"
