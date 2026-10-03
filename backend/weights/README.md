# Model ağırlıkları

Bu dosya eski (proje henüz planlama aşamasındayken yazılmış) bir notu
taşıyordu — artık gerçek mimariyle uyuşmuyordu, güncellendi.

- `yolov8n.pt` — kişi tespiti için resmi Ultralytics YOLOv8n ağırlığı.
  `backend/` kökünde duruyor, `.gitignore`'daki `*.pt` kuralı yüzünden
  git'e commitlenmez; yerelde bulunamazsa ultralytics ilk çalıştırmada
  kendi sunucusundan otomatik indirir.
- `eye_classifier.pth` — göz açık/kapalı sınıflandırıcısı, bu depoda
  `scripts/train_eye_classifier.py` ile MRL Eye Dataset üzerinde
  eğitildi. Doğrulama seti kişi bazlı ayrıldı (train 25 denek, val 12
  denek, ortak denek yok): val_accuracy = 0.9522, closed_recall =
  0.9780, false-awake oranı = 0.0220. Ayrıntı için
  `eye_classifier_metrics.json` ve kök `README.md`'nin "Veri Seti ve
  Model Değerlendirmesi" bölümü. `.pth` uzantısı `.gitignore`'da yok — yani bu
  dosya BİLEREK git'e commitlenir, çünkü `ALLOW_DETECTION_FALLBACK=false`
  ile teslim edildiğinde sistemin çalışması bu ağırlığın depoda mevcut
  olmasına bağlı.

Not: eğitim script'ini denerken üretilen `eye_classifier_test.pth` /
`eye_classifier_test_metrics.json` ara çıktıları teslim öncesi silindi;
`.gitignore` ve `backend/.dockerignore` artık `weights/*_test.*`
desenini hariç tutuyor, yani yeni denemeler kazara depoya/imaja girmez.
