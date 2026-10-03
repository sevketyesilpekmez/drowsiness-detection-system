# CHANGELOG

Projeye Flask tabanlı backend ve React tabanlı frontend mimarisi
oluşturularak başlandı.
Bu aşamada;
-   Flask uygulaması bir **app factory** deseniyle (`create_app()`)
    kuruldu — bu sayede test ortamı ile gerçek çalıştırma ortamı farklı
    konfigürasyonlarla (`ENABLE_DETECTION_ENGINE=false`/`true`) aynı
    kod üzerinden ayrıştırılabildi. `GET /api/health` endpoint'i ile
    servisin ayakta olup olmadığı doğrulanabilir hale getirildi.
-   React + Vite arayüzü hazırlandı; geliştirme sırasında `/api/*`
    isteklerinin backend'e proxy'lenmesi vite dev server üzerinden
    sağlandı, böylece CORS yapılandırmasıyla uğraşmaya gerek kalmadı.
-   Docker Compose yapısı oluşturuldu: `backend` ve `frontend` olmak
    üzere iki servis, frontend tarafında nginx üzerinden statik dosya
    sunumu ve `/api` reverse proxy planlandı.
-   MJPEG video akışı (`multipart/x-mixed-replace; boundary=frame`)
    geliştirildi — tarayıcıda ek bir kütüphane gerekmeden doğrudan
    `<img src="/api/stream">` ile tüketilebilecek şekilde tasarlandı.
-   Kamera görüntüsünün frontend'e aktarılması, backend'de kamerayı
    açıp kapatan bağımsız bir bileşen (ileride `DetectionEngine`
    olacak arka plan thread'i) üzerinden sağlandı.
-   `SharedState` yapısı oluşturuldu: `threading.Lock` ile korunan, en
    son karenin JPEG halini ve en son tespit sonucunu tutan, birden
    fazla HTTP isteğinin aynı anda güvenle okuyabildiği paylaşılan bir
    durum deposu.

Bu sürüm, projenin temelini oluşturmaktadır.

# v1.1.0 - İlk Uyku Tespit Sistemi
MediaPipe Face Mesh kullanılarak yüz landmarkları çıkarıldı.

Yapılan çalışmalar;

-   EAR (Eye Aspect Ratio) hesaplaması eklendi: her göz için 6 landmark
    noktası (`LEFT_EYE_INDICES`/`RIGHT_EYE_INDICES`) kullanılarak
    `EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)` formülüyle hesaplanan
    oran, `EAR_THRESHOLD` (varsayılan `0.22`) değerinin altına
    düştüğünde göz "kapalı" sayılacak şekilde kuruldu.
-   Süre tabanlı `DrowsinessService` geliştirildi: gözler kapalı
    olarak işaretlendiği ilk an kaydedilip, bu süre `DROWSINESS_SECONDS`
    eşiğini (başlangıçta `2.0`, sonraki bir sürümde gerçek kamera
    testiyle `2.5`'e yükseltildi) aştığında durum `DROWSY`'ye
    geçirilecek şekilde tasarlandı — tek bir kare değil, zaman
    penceresi esas alındı.
-   `AlarmService` oluşturuldu: `is_drowsy=true` sinyali geldiğinde
    cooldown'lu (belirli bir süre geçmeden tekrar tetiklenmeyen) bir
    sayaç (`alarm_trigger_count`) artıran bağımsız bir servis olarak
    kuruldu.
-   `DetectionEngine` altyapısı tamamlandı: kamerayı/video kaynağını
    açan, her karede bir `detect_fn` çağıran, sonucu `SharedState`'e
    yazan arka plan thread'i. `detect_fn` enjeksiyonu sayesinde
    algılama mantığı (MediaPipe/YOLO) motorun kendisinden tamamen
    ayrıştırıldı.

Bu sürümde sistem ilk kez gerçek zamanlı olarak çalışabilir hale geldi.

# v1.2.0 - React Dashboard

Frontend tamamen yenilendi.

Eklenen özellikler;
-   Canlı kamera görüntüsü (`CameraFeed` bileşeni, `<img>` üzerinden
    MJPEG stream)
-   FPS bilgisi (backend'in hesapladığı anlık değerin ekranda
    gösterilmesi)
-   Sistem durumu (`StatusCard`: `STARTING`/`AWAKE`/`DROWSY`/`NO_FACE`/
    `CAMERA_ERROR`/`MODEL_ERROR` durumlarının okunabilir etiketlerle
    gösterimi)
-   Alarm durumu (aktif/pasif göstergesi)
-   Göz durumu kartı (açık/kapalı, güven değeri)

Backend ile REST API üzerinden haberleşme sağlandı: `useDetectionStatus`
hook'u `/api/detection/status` endpoint'ini belirli aralıklarla poll ederek arayüzü güncel tutacak şekilde yazıldı.

# v1.3.0 - Docker Desteği

Projenin farklı işletim sistemlerinde kolay çalıştırılabilmesi amacıyla
Docker desteği geliştirildi.

Yapılan çalışmalar;

-   Backend `Dockerfile`'ı hazırlandı: `python:3.11-slim` tabanlı,
    üretimde `gunicorn --config gunicorn.conf.py run:app` ile
    başlayacak şekilde kuruldu. `gunicorn.conf.py` bilinçli olarak
    `workers=1` (kamera tek bir process/thread tarafından yönetildiği
    için birden fazla worker çakışma üretir), `threads=4` (HTTP
    eşzamanlılığı için) ve `timeout=0` (MJPEG stream süresiz açık
    kalan bir bağlantı olduğu için varsayılan 30s timeout devre dışı)
    olarak ayarlandı.
-   Frontend `Dockerfile`'ı çok aşamalı (multi-stage) build olarak
    hazırlandı: `npm run build` ile üretilen statik dosyalar
    `nginx:alpine` imajına kopyalandı.
-   Docker Compose güncellendi: `backend` ve `frontend` servisleri,
    `restart: unless-stopped` politikası, backend için host portu
    `5001:5000` olarak ayarlandı (macOS'ta AirPlay Receiver'ın
    varsayılan olarak 5000 portunu dinlemesiyle çakışmayı önlemek
    için — container içi port hâlâ 5000).
-   Nginx yapılandırıldı: statik dosya sunumu ve `/api/*` isteklerinin
    `backend:5000`'e (Docker'ın iç ağı üzerinden) reverse proxy'lenmesi
    sağlandı — frontend hiçbir zaman backend'in mutlak adresini bilmek
    zorunda kalmadı, CORS yapılandırmasına gerek kalmadı.
-   Ortam değişkenleri `.env.example` üzerinden düzenlendi;
    `docker-compose.yml`'in `environment:` bölümü bu değişkenleri
    `${VAR:-default}` söz dizimiyle okuyacak şekilde kuruldu.


# v1.4.0 - Test Altyapısı

Kod kalitesini artırmak amacıyla test altyapısı oluşturuldu.

Backend tarafında;

-   Pytest kuruldu, `backend/pytest.ini` ile test taraması yalnızca
    `tests/` klasörüyle sınırlandırıldı.
-   Unit testler: `SharedState`, `VideoSource`, `DetectionEngine` gibi
    altyapı sınıfları için, gerçek kamera veya ağır bağımlılıklar
    olmadan sahte (fake) nesnelerle çalışan testler yazıldı.
-   API testleri: `/api/health`, `/api/detection/status`, `/api/stream`
    route'larının doğru veri döndürdüğü doğrulandı.
-   Detection Engine testleri: thread yaşam döngüsü (`start()`/`stop()`),
    hata sınıflandırması (`CAMERA_ERROR`/`MODEL_ERROR`/`SYSTEM_ERROR`),
    kaynakların (`video_source.release()`, `detect_fn.close()`) doğru
    şekilde serbest bırakılması test edildi.
-   Alarm testleri: cooldown mekanizması, `alarm_trigger_count`'un
    monoton artışı, callback davranışı test edildi.

Frontend tarafında;

-   Vitest ve `@testing-library/react` kuruldu.
-   Component testleri: `CameraFeed`, `StatusCard`, `AlarmController`,
    `useDetectionStatus` hook'u için ilk testler yazıldı.

eklendi.


# v1.5.0 - YOLOv8 + PyTorch Geçişi

MediaPipe tabanlı ilk yapı geliştirildikten sonra görev gereksinimleri
doğrultusunda YOLOv8 ve PyTorch tabanlı yeni algılama hattı oluşturuldu.

Yapılan çalışmalar;

-   YOLOv8 ile kişi tespiti: `UltralyticsPersonDetector`, resmi
    `yolov8n.pt` ağırlığıyla (COCO veri setinde sınıf 0 = "person")
    karede bir kişi kutusu arıyor. Özel/kaynağı belirsiz bir yüz-tespit
    ağırlığı indirmek yerine bilinçli olarak bu yol seçildi.
-   Haar Cascade ile göz bölgesi: `EyeRegionLocator`, OpenCV'nin kendi
    bünyesinde gelen (`cv2.data.haarcascades/haarcascade_eye.xml`,
    ek indirme gerektirmeyen) Haar cascade'i, YOLO'nun bulduğu kişi
    kutusu içinde çalıştırıyor.
-   PyTorch göz sınıflandırıcısı: `EyeStateCNN` mimarisi ve
    `TorchEyeClassifier` adaptörü ile her göz kırpımı `OPEN_EYES`/
    `CLOSED_EYES` olarak sınıflandırılıyor; her iki göz de kapalı
    çıkarsa gözler kapalı sayılıyor (tek göz kapalı görünmesi yanlış
    pozitif üretmesin diye kasıtlı bir "AND" kararı).
-   Yeni Detection Pipeline: `YoloDrowsinessDetector`, bu üç bileşeni
    (`UltralyticsPersonDetector` → `EyeRegionLocator` →
    `EyeClassifierService`) sırayla çağırıp `DrowsinessService`'in
    beklediği ortak sözleşmeye (`REQUIRED_DETECT_FN_KEYS`) uyan bir
    sonuç sözlüğü üretecek şekilde yazıldı.

Sistem, `ALLOW_DETECTION_FALLBACK` bayrağı sayesinde gerektiğinde
MediaPipe hattına geri dönebilecek şekilde tasarlandı: `yolo_pytorch`
kurulamazsa (bağımlılık ya da eğitilmiş ağırlık eksikse) `true`
değerinde otomatik ve sessizce `mediapipe`'a düşülüyor ; `false` değerinde düşülmüyor, durum kalıcı
olarak `MODEL_ERROR` gösteriyor — böylece istenen teknoloji setinin
gerçekten çalıştığı gizlenmiyor.

# v1.6.0 - Model Eğitimi

MRL Eye Dataset kullanılarak göz sınıflandırma modeli eğitildi.
Yapılan çalışmalar;
-   Veri seti hazırlandı: `scripts/prepare_eye_dataset.py` ile MRL Eye
    Dataset, `torchvision.datasets.ImageFolder` uyumlu bir klasör
    yapısına (`train/closed`, `train/open`, `val/closed`, `val/open`)
    dönüştürüldü — toplam 67.919 eğitim ve 16.979 doğrulama görüntüsü.
-   Eğitim scripti geliştirildi: `scripts/train_eye_classifier.py`,
    `ImageFolder`'ın ürettiği sınıf sırasının `["closed", "open"]`
    olduğunu çalışma zamanında doğrulayarak eğitim/çıkarım arasında
    sessiz bir etiket-indeksi karışıklığı riskini ortadan kaldırdı.
-   Doğrulama testleri yapıldı: her epoch sonunda train/val loss ve
    val accuracy hesaplanıp raporlandı.
-   En başarılı model kaydedildi: 20 epoch'luk eğitimde en iyi sonuç
    epoch 12'de elde edildi, **val_accuracy = 0.9891**.
-   Confusion Matrix oluşturuldu: `[[8275, 114], [71, 8519]]` —
    satır/sütun toplamları `data/eyes/val/` klasöründeki gerçek dosya
    sayılarıyla birebir örtüşüyor. Eğitim geçmişi ve confusion matrix
    `weights/eye_classifier_metrics.json` dosyasına kaydedildi.

> **Bu bölümdeki eğitim sonuçları (0.9891 doğruluk, yukarıdaki
> confusion matrix ve veri seti sayıları) geçerli değildir.** Teslim öncesi
> kontrolde, veri setinin görüntü bazında bölündüğü ve 37 deneğin
> 37'sinin de hem eğitim hem doğrulama tarafında yer aldığı tespit
> edildi; bu skor modelin gördüğü kişileri hatırlama başarımını
> ölçüyordu. Bölme kişi bazlı hale getirilip model yeniden eğitildi.
> Geçerli sonuçlar için aşağıdaki "Teslim Öncesi Son Kontrol"
> bölümüne bakınız.
-   Ek doğrulama: `scripts/test_eye_classifier.py` ile tek göz
    görüntüsü üzerinde çalıştırılan gerçek bir çıkarım testinde
    "closed" örneği %100.00, "open" örneği %99.91 güvenle doğru
    sınıflandırıldı.


# v1.7.0 - Gerçek Kamera Testleri

Gerçek kamera testleri sırasında çeşitli problemler tespit edildi.

Çözülen başlıca problemler;

## Tek karelik hatalar

Göz kapalı serisi sürerken Haar cascade'in gözü kaçırdığı ya da sınır
çizgisinde yanlış sınıflandırma yaptığı tek bir kare, `closed_duration`
sayacını DROWSY eşiğine çok yakınken sıfırlıyordu — gerçek logda bu
durum ~1.9 saniyede üç kez tekrarlandığı için rastgele bir göz kırpması
olmadığı netleşti.

Çözüm;

-   `CLOSED_EYES_GRACE_PERIOD_SECONDS` (varsayılan `0.3`) eklendi:
    göz kapalı serisi sürerken gelen kısa bir hatalı "açık" okuması
    (ya da YOLO'nun kişiyi tek karede kaçırdığı kısa bir
    `face_detected=False` durumu) artık birikimi sıfırlamıyor.

## Haar Cascade kararsızlığı

Kullanıcı başını öne eğdiğinde (tam da bir sürücünün uyuklarken yaptığı
hareket) Haar cascade'in göz bulma başarısı ciddi şekilde düşüyordu —
cascade frontal/dik açılara göre eğitilmiş olduğu için poz değiştikçe
kırılganlaşıyordu.

Çözüm;

-   MediaPipe Face Mesh tabanlı Eye Locator (`MediaPipeEyeLocator`)
    geliştirildi: göz konumu artık 3D yüz landmark tahminiyle
    bulunuyor, bu da baş açısına Haar'dan çok daha dayanıklı.
    YOLO (kişi tespiti) ve PyTorch (göz sınıflandırma) hiç değişmedi —
    yalnızca aradaki "göz nerede" adımı değiştirildi.
-   Haar Cascade tamamen kaldırılmadı, yedek yöntem olarak bırakıldı:
    `FallbackEyeLocator`, MediaPipe'ı birincil, Haar'ı yedek olarak
    sarmalıyor — MediaPipe o karede kutu bulamazsa otomatik olarak
    Haar'a düşülüyor (`EYE_LOCATOR_BACKEND=mediapipe`, yeni varsayılan).

## Kare kırpma problemi

MediaPipe'a geçildikten hemen sonra, kapalı gözlerin "açık" olarak
sınıflandırıldığı ikinci bir problem ortaya çıktı. Kök sebep: landmark
noktalarından çıkarılan göz kutusu doğası gereği yatay bir dikdörtgendi
(genişlik >> yükseklik, göz kapandıkça daha da yassılaşıyor); PyTorch
sınıflandırıcı ise bu kutuyu sabit bir kareye (64x64) resize ediyordu —
yassı bir dikdörtgeni kareye zorlamak görüntüyü dikeyde ciddi şekilde
geriyor, özellikle göz kapalıyken.

Çözüm;

-   Kare crop yöntemi geliştirildi: crop artık iki boyuttan büyük
    olana (genelde genişlik) göre kare olarak kesiliyor, göz
    açık/kapalı fark etmeksizin kırpma oranı sabit kalıyor.
-   Ön işleme güncellendi ve ilgili testler eklendi/güncellendi
    (özellikle "göz kapanınca kare hâlâ genişliğe göre mi
    belirleniyor" senaryosunu doğrulayan bir test).

Bu üç düzeltmenin ardından backend test sayısı 199'a ulaştı; her ikisi
(baş eğikliği ve kapalı göz tespiti) gerçek kamerada tekrar test
edilerek doğrulandı.


# v1.8.0 - Alarm Sistemi

Eklenen özellikler;

-   Kesintisiz alarm: sesli alarm artık `alarm_trigger_count`'un
    (cooldown'lu sayaç) tetiklediği kısa kısa bir sinyale değil,
    `alarm_active`'e bağlı — gözler kapalı kaldığı sürece kesintisiz
    çalıyor, gözler açılır açılmaz otomatik duruyor.
-   Alarm geçmişi: son 5 alarmın metadata'sı (`id`, `timestamp`,
    `state`, `confidence`, `closed_duration`) `SharedState` içinde
    `collections.deque(maxlen=5)` ile tutuluyor.
-   Alarm anı görüntüsü: `alarm_triggered=True` döndüğü TEK karede
    ilgili JPEG frame otomatik yakalanıp saklanıyor, bir sonraki
    alarma kadar sabit kalıyor.
-   Alarm Snapshot API: `GET /api/detection/alarm-snapshot` (en son
    alarm) ve `GET /api/detection/alarm-snapshot/<id>` (geçmişteki
    belirli bir alarm) eklendi.
-   Alarm geçmişi API: `GET /api/detection/alarm-history` eklendi.

Frontend tarafında;

-   Alarm ekranı: `AlarmBanner` (üstte büyük uyarı bandı) ve
    `AlarmFlashOverlay` (ekranı saran kırmızı nabız efekti).
-   Alarm geçmişi: `AlarmHistory` bileşeni, son 3-5 alarmı listeliyor.
-   Snapshot görüntüleme: `AlarmSnapshot` bileşeni, alarm anındaki
    kareyi kapalı kalma süresi/güven/durum bilgisiyle birlikte
    gösteriyor.
-   Alarm susturma: mevcut alarmı susturan ama sistemi kapatmayan bir
    "Sustur" butonu — bir sonraki alarm yine sesli başlıyor.
-   Geliştirilmiş animasyonlar: alarm modunda sayfa temasının
    (radial tint, kamera glow, kırmızı status card) değişmesi.

eklendi.


# v1.9.0 - Performans Çalışmaları

Yapılan geliştirmeler;

-   Ortalama FPS hesabı güncellendi: anlık `1/elapsed` değeri yerine
    son 30 karenin süresi biriktirilip `1 / ortalama_süre` şeklinde
    hesaplanıyor (`DetectionEngine._record_frame_and_get_average_fps`)
    — bu, tek bir yavaş karenin ekrandaki FPS sayısını aşırı
    dalgalandırmasını önlüyor.
-   Kamera çözünürlüğü yapılandırılabilir hale getirildi:
    `CAMERA_WIDTH`/`CAMERA_HEIGHT` daha önce tanımlı olmasına rağmen
    hiçbir zaman gerçek kameraya uygulanmıyordu — kamera her zaman
    sürücünün varsayılan (genelde çok daha yüksek) çözünürlüğünde
    açılıyordu. `VideoSource.open()` artık bu değerleri
    `cv2.CAP_PROP_FRAME_WIDTH/HEIGHT` ile gerçekten uyguluyor (yalnızca
    kamera kaynakları için, video dosyaları için değil).
-   YOLO görüntü boyutu (.env) üzerinden yönetilebilir hale getirildi:
    `YOLO_IMGSZ` (varsayılan `640`) eklendi, `UltralyticsPersonDetector`
    artık bunu açıkça `imgsz` parametresi olarak modele geçiriyor.
-   API üzerinden performans bilgileri raporlanmaya başlandı:
    `/api/detection/status` yanıtına `camera_read_ms`, `jpeg_encode_ms`,
    `person_detect_ms`, `eye_pipeline_ms` alanları eklendi — darboğazın
    hangi adımda olduğu artık her poll'da görülebiliyor.
-   `TORCH_DEVICE` (varsayılan `cpu`, Apple Silicon'da `mps` seçilebilir)
    eklendi; `UltralyticsPersonDetector` ve `TorchEyeClassifier` artık
    bu değeri kullanıyor.
-   Docker ortamındaki eksik ortam değişkenleri tamamlandı:
    `CAMERA_WIDTH`/`CAMERA_HEIGHT`/`TORCH_DEVICE`/`YOLO_IMGSZ` gibi
    değişkenler `Config`'e eklenmişti ama `docker-compose.yml`'in
    `environment:` listesine hiç eklenmemişti — Docker ile çalıştıran
    biri için bu değişkenler `.env`'de tanımlı olsa bile container'a
    hiç ulaşmıyordu, bu eksiklik giderildi.


# v2.0.0 - Dokümantasyon

Yapılan çalışmalar;

-   README tamamen yeniden yazıldı: doğal, sade bir dille; gereksiz
    tekrarlar ve geliştirme sürecine dair notlar (bunlar bu
    CHANGELOG'a taşındı) README'den ayrıştırıldı.
-   Proje yapısı güncellendi: gerçek klasör ağacını (`backend/app/
    {api,core,models,services}`, `scripts/`, `weights/`, `tests/`)
    birebir yansıtan bir bölüm eklendi.
-   API dokümantasyonu düzenlendi: tüm endpoint'ler (health, stream,
    detection/status, alarm-history, alarm-snapshot) tablo halinde
    listelendi.
-   Kurulum adımları sadeleştirildi: Docker ile ve Docker'sız
    çalıştırma tek bir "Kurulum" bölümü altında toplandı.
-   Ortam değişkenleri açıklandı: `.env`'deki öne çıkan değişkenler ve
    anlamları README'ye eklendi.
-   Docker ve Docker'sız kullanım ayrıldı: her ikisi için ayrı, adım
    adım komut blokları verildi.
-   Test bölümü güncellendi: gerçek test sayıları (backend 280,
    frontend 55) ve nelerin test edildiği eklendi.
-   CHANGELOG yeniden düzenlendi: sürüm numaralarına göre
    (v1.0.0 → v2.0.0) kronolojik bir yapıya kavuşturuldu, her sürümde
    karşılaşılan problem/çözüm/sonuç ayrımı netleştirildi.


## Genel Özet

Proje geliştirme süreci boyunca;

-   Flask tabanlı backend, app factory deseniyle ve tek worker/çoklu
    thread (`gunicorn`, `workers=1, threads=4`) mimarisiyle geliştirildi.
-   React tabanlı kullanıcı arayüzü hazırlandı, backend'e relative
    `/api` yolu ve nginx/vite proxy üzerinden bağlandı.
-   YOLOv8 (resmi `yolov8n.pt`) ile kişi tespiti gerçekleştirildi.
-   MediaPipe Face Mesh ile göz bölgeleri belirlendi (Haar Cascade
    yedek yöntem olarak korundu).
-   PyTorch modeli (MRL Eye Dataset üzerinde, kişi bazlı ayrılmış
    train/val setiyle eğitildi; val_accuracy = 0.9522,
    closed_recall = 0.9780) ile göz sınıflandırması yapıldı.
-   Süre tabanlı (`DROWSINESS_SECONDS`, grace period toleranslı) uyku
    tespit algoritması geliştirildi.
-   Alarm sistemi, gözler açılana kadar kesintisiz çalacak ve son 5
    alarmın geçmişini/görüntüsünü sunacak şekilde gerçek zamanlı
    tasarlandı.
-   Docker desteği eklendi (backend + frontend + nginx, tek komutla
    `docker compose up --build`).
-   REST API geliştirildi (health, stream, detection/status,
    alarm-history, alarm-snapshot).
-   Backend için 303, frontend için 57 testten oluşan kapsamlı bir
    test altyapısı oluşturuldu.
-   Gerçek kamera testleri sonucunda tespit edilen üç ayrı problem
    (tek karelik hatalar, Haar cascade'in baş pozuna kırılganlığı,
    kapalı gözde kare kırpma bozulması) giderildi.
-   Dokümantasyon (README + bu CHANGELOG) teslim öncesinde tamamen
    güncellendi.

## Son Kontrol

Teslimden önce projeyi baştan sona bir kez daha gözden geçirdim. Bulduğum
şeyleri ve nasıl çözdüğümü buraya yazdım — bir kısmı sonuçları doğrudan
etkilediği için sadece "düzelttim" demek yeterli olmazdı.

**Ölçüm ve model tarafı**

-   **Eğitim ve doğrulama setleri arasında veri sızıntısı vardı.** Veri
    setini görüntü bazında rastgele bölüyordum. MRL veri setinde bir
    deneğin aynı oturumdan çekilmiş yüzlerce neredeyse aynı karesi olduğu
    için, kontrol ettiğimde 37 deneğin 37'sinin de hem eğitim hem
    doğrulama tarafında yer aldığını gördüm; yalnızca doğrulamada bulunan
    tek bir kişi bile yoktu. Yani model yeni bir sürücüye genelleme
    yapmıyor, gördüğü kişiyi hatırlıyordu. Bölmeyi kişi bazlı hale
    getirdim: bir deneğin tüm kareleri tek bir tarafta kalıyor, iki set
    ortak kişi içermiyor. Eski davranışı `--split-by image` altında
    bıraktım ama kullanılınca uyarı basıyor. Aynı hatanın tekrar
    oluşmaması için 9 test yazdım.
-   **Modeli yeni bölmeyle sıfırdan eğittim.** 37 deneğin 25'i eğitim
    (66.449 görüntü), 12'si doğrulama (18.449 görüntü). En iyi sonuç yine
    epoch 12'de çıktı: **doğruluk 0.9522**, `closed` sınıfı için
    precision/recall/F1 = 0.8875 / 0.9780 / 0.9306, false-awake oranı
    0.0220, false-alarm oranı 0.0604, confusion matrix
    `[[5911, 133], [749, 11656]]`. Önceki 0.9891'e göre yaklaşık 3.7
    puanlık düşüş, modelin ezberlediği kısmın ölçümden çıkması demek —
    yani düşen sayı aslında doğru olan sayı. Ayrıntılı tablo
    `README.md`'nin "Veri Seti ve Model Değerlendirmesi" bölümünde.
-   **Model seçimini toplam doğruluktan `closed_f1`e çevirdim.** Bu
    sistemde kapalı gözü "açık" sanmak uyuklayan sürücünün uyarılmaması
    demek; tersi ise sadece gereksiz alarm. Toplam doğruluk bu ikisini
    tek sayıda eritiyordu. Eğitim script'i artık sınıf bazlı
    precision/recall/F1 ile `false_awake_rate` ve `false_alarm_rate`
    hesaplayıp hem ekrana basıyor hem metrik dosyasına yazıyor; en iyi
    epoch varsayılan olarak kapalı göz F1'ine göre seçiliyor
    (`--select-by`). Metrik hesaplarını torch'tan bağımsız tutup 10
    testle doğruladım.
-   **Bu tercihin ne kadar yerinde olduğunu aynı eğitimde gördüm.**
    Epoch 5'te `closed_recall` 0.9957'ye çıktı (false-awake yalnızca
    0.0043) ama doğruluk 0.8606'ya, `closed_f1` 0.8240'a düştü: model
    açık gözleri de "kapalı" saymaya başlamış, sürekli yanlış alarm veren
    bir şeye dönüşmüştü. Sadece recall'e göre seçim yapsaydım teslim
    ettiğim model bu olacaktı.

**Davranış ve güvenlik**

-   **`face_detected` yalnızca kişi tespitine bakıyordu.** Sürücü yüzünü
    çevirdiğinde gövdesi kadrajda kaldığı için sistem, hiç göz ölçümü
    alamadığı halde `NO_FACE` yerine `AWAKE` diyebiliyordu. Bir uyku
    tespit sisteminde "ölçemiyorum" ile "uyanık" aynı şey olmadığı için
    bunu düzelttim: göz bölgesinin bulunamadığı ardışık kareleri
    `EYE_LOCATOR_MISS_TOLERANCE` (varsayılan 2) ile sınırladım, tolerans
    aşılınca `face_detected=False` olup sistem `NO_FACE`'e geçiyor. Kısa
    kaçırmaların birikmiş süreyi sıfırlamaması davranışını korudum.
    Durum çıktısına ayrıca `person_detected`, `eye_region_found` ve
    `face_detected` alanlarını ekledim.
-   **`run.py` içinde `debug=True` kalmıştı.** Werkzeug'un debug modu,
    hata sayfası üzerinden sunucuda kod çalıştırılabilen bir konsol
    açıyor; uygulama `0.0.0.0` dinlediği için bu konsol ağdaki diğer
    makinelere de görünür oluyordu. Varsayılanı kapattım, geliştirirken
    `FLASK_DEBUG=true` ile açılabiliyor.
-   **Backend portu Docker'da dışarı açılıyordu.** API'nin tamamı kimlik
    doğrulamasız ve içinde canlı kamera görüntüsüyle alarm fotoğrafları
    var; `5001:5000` yayını bunları host ağına açıyordu. Yayını
    kaldırdım — nginx zaten compose ağı üzerinden backend'e ulaşıyor,
    sistem `http://localhost:3000` üzerinden çalışıyor.
-   **Alarm görüntüsü kartı tek bir hatada kalıcı olarak bozuluyordu.**
    `AlarmSnapshot.jsx` içindeki `imageError` yeni alarm geldiğinde
    sıfırlanmıyordu. Hata durumunda `<img>` DOM'dan kaldırıldığı için
    `key={triggerCount}` de remount tetikleyemiyor, dolayısıyla sonraki
    alarmların görüntüleri hiç denenmiyordu. Yeni alarmda hata durumunu
    sıfırlayacak şekilde düzeltip regresyon testi ekledim.

**Depo ve yapılandırma**

-   **Eğitim veri setini sürüm kontrolünün ve Docker imajının dışına
    aldım.** `backend/data/` altındaki ~170.000 dosya / ~900 MB veri ne
    `.gitignore`'da ne `.dockerignore`'da vardı; her
    `docker compose build` çağrısında bu verinin tamamı build
    context'ine, `COPY . .` ile de imaja giriyordu. Eğitim script'lerini
    ve ara deneme checkpoint'lerini de runtime imajından çıkardım.
-   **ML bağımlılıklarının sürümlerini sabitledim.** `torch>=2.2`,
    `torchvision>=0.17` ve `ultralytics>=8.2` açık uçlu aralıklardı.
    Teslim edilen `eye_classifier.pth` belirli sürümlerle eğitildiği
    için hepsini `==` ile pinledim.
-   **Ara deneme çıktılarını sildim.** `weights/eye_classifier_test.pth`
    ve metrik dosyası kaldırıldı; teslim ağırlığı yalnızca
    `weights/eye_classifier.pth`.
-   **Projeyi git deposu altına aldım.**

**Belgelenen sınırlamalar**

Çözmediğim ama bilinmesi gereken konuları README'nin "Bilinen
Sınırlamalar" bölümüne yazdım: kadrajda birden fazla kişi olduğunda
YOLO'nun seçtiği kişiyle gözleri sınıflandırılan kişinin farklı
olabilmesi, alarm kayıtlarının yalnızca bellekte tutulması, API'de kimlik
doğrulaması bulunmaması, `/api/health`'in yalnızca process'in ayakta
olduğunu bildirmesi ve `DetectionEngine.stop()`'un thread'in gerçekten
kapandığını doğrulamaması. Veri seti bölme yöntemi ile metrik seçimini
ise "Veri Seti ve Model Değerlendirmesi" bölümünde anlattım.