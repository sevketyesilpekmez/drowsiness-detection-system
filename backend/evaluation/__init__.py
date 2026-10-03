"""
Uçtan uca sistem değerlendirmesi.

Bu paket, TEK KARELİK sınıflandırıcı metriklerinden (bkz.
scripts/train_eye_classifier.py) farklı bir soruyu cevaplar: etiketli bir
videoda denek 10. saniyede gözünü kapattığında sistem alarm verdi mi,
kaç saniye sonra verdi, dakikada kaç yanlış alarm üretti?

Bilinçli olarak torch/OpenCV/Flask import ETMEZ — metrik hesabı saf
Python'dur, video olmadan test edilebilir ve CI'da koşabilir. Videoyu
okuyup alarm zaman damgalarını üreten replay katmanı ayrıdır.

Neden `app/` ALTINDA DEĞİL: `app/__init__.py` Flask uygulama fabrikası
ve modül seviyesinde mediapipe, cv2, flask import ediyor. Bu paket
`app.evaluation` olsaydı, `from app.evaluation.metrics import ...` tek
satırı bile o ağır bağımlılıkların tamamını yüklerdi ve "saf Python,
her yerde koşar" iddiası pratikte geçersiz olurdu. Ayrıca burası bir
analiz aracı, çalışma zamanı uygulamasının parçası değil — `tests/` ve
`scripts/` gibi Docker imajına da girmiyor (bkz. .dockerignore).
"""
