# Kamera/video kaynağı process içinde tekil bir background thread tarafından
# açılacağı için uygulama TEK worker ile çalışmalı. Birden fazla worker aynı
# kamerayı açmaya çalışır ve çakışma/CAMERA_ERROR üretir. HTTP eşzamanlılığı
# worker sayısı yerine thread sayısıyla sağlanıyor.
workers = 1
threads = 4
worker_class = "gthread"
bind = "0.0.0.0:5000"

# MJPEG stream endpoint'i (/api/stream) bağlantıyı süresiz açık tutar.
# Varsayılan 30s timeout bu bağlantıyı "asılı kaldı" sanıp worker'ı öldürür.
# 0 = timeout devre dışı.
timeout = 0
