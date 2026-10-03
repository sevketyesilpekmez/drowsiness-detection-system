# Kamera Tabanlı Uyku Tespit Sistemi

## Proje Hakkında
Bu proje, sürücülerin dikkat seviyesini gerçek zamanlı olarak takip ederek olası uyku durumlarını tespit etmeyi amaçlayan kamera tabanlı bir görüntü işleme sistemidir. Sistem, bilgisayar kamerasından alınan görüntüleri analiz ederek sürücünün göz durumunu sürekli izlemekte ve gözlerin belirlenen süre boyunca kapalı kalması durumunda kullanıcıyı sesli alarm ile uyarmaktadır. Proje, React tabanlı bir web arayüzü ve Flask tabanlı bir backend mimarisi kullanılarak geliştirilmiştir. Görüntü işleme sürecinde YOLOv8, MediaPipe Face Mesh ve PyTorch teknolojileri birlikte kullanılmıştır. YOLOv8 insan tespiti için, MediaPipe Face Mesh yüz ve göz noktalarının belirlenmesi için, PyTorch ise göz görüntülerinin açık veya kapalı olarak sınıflandırılması için kullanılmıştır. Sistem modüler bir yapıda tasarlanmış olup kamera yönetimi, görüntü işleme, alarm sistemi ve API servisleri birbirinden bağımsız olarak geliştirilmiştir. Ayrıca Docker desteği sayesinde farklı işletim sistemlerinde kolayca kurulup çalıştırılabilmektedir. Geliştirilen yapı gerçek zamanlı çalışabilecek şekilde optimize edilmiş ve kullanıcıya web arayüzü üzerinden anlık durum bilgileri sunacak şekilde tasarlanmıştır.


## Kullanılan Teknolojiler

### Backend
- Python 
- Flask
- Gunicorn
- OpenCV
- MediaPipe Face Mesh
- YOLOv8
- PyTorch
- NumPy

### Frontend
- React
- Vite
- JavaScript
- HTML
- CSS

### DevOps
- Docker
- Docker Compose
- Nginx

### Test
- Pytest
- Vitest

## Sistem Mimarisi
```text
               Kamera
                  │
                  ▼
          Video Capture
                  │
                  ▼
          YOLOv8 Person Detection
                  │
                  ▼
       MediaPipe Face Mesh
                  │
                  ▼
         Eye Region Extraction
                  │
                  ▼
       PyTorch Eye Classifier
                  │
                  ▼
       Drowsiness Detection
                  │
                  ▼
          Alarm Service
                  │
                  ▼
            Flask REST API
                  │
                  ▼
           React Dashboard
```

## Proje Yapısı

```text
project/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   └── services/
│   ├── evaluation/        uçtan uca ölçüm (app'ten bağımsız, saf)
│   ├── scripts/
│   ├── tests/
│   ├── weights/
│   └── run.py
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── Dockerfile
│
├── benchmark/             etiketli değerlendirme seti
│   ├── clips/             videolar (git'e girmez)
│   ├── labels/            JSON etiketler
│   └── results/           koşu çıktıları (git'e girmez)
│
├── sample_data/
├── docker-compose.yml
├── .env.example
└── README.md
```


## Ön Gereksinimler

- Python 3.10+
- Node.js 20+
- Docker & Docker Compose (isteğe bağlı)
- Git

## Kurulum

### Docker ile
```bash
cp .env.example .env
docker compose up --build
```
Ardından sistem `http://localhost:3000` üzerinden kullanılır. Backend portu dışarı yayınlanmaz; nginx `/api/*` isteklerini compose ağı üzerinden backend'e iletir. Backend'e doğrudan erişmek gerekirse `docker-compose.yml` içindeki `ports` satırları geçici olarak açılabilir (bkz. "Bilinen Sınırlamalar" — API kimlik doğrulaması içermez).

> **Not**
>
> - Docker ile çalıştırıldığında kamera erişimi işletim sistemine bağlıdır.
> - Linux ortamında fiziksel kamera (`/dev/video0`) doğrudan Docker container'ına aktarılabilir.
> - macOS ve Windows'ta Docker Desktop, yerel kameraya doğrudan erişemeyebilir. Bu durumda backend'in yerel ortamda (`python run.py`) çalıştırılması veya IP Camera (DroidCam, Iriun vb.) kullanılması önerilir.

### Docker Olmadan

Backend bağımlılıkları sistem geneline kurulmamalı, proje için bir sanal ortam (virtual environment) oluşturulmalıdır.

Backend;
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp ../.env.example .env
python run.py
```

Windows (PowerShell) kullanılıyorsa sanal ortam şu şekilde etkinleştirilir:
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

Frontend;
```bash
cd frontend
npm install
npm run dev
```
Bu şekilde frontend `http://localhost:5173`, backend `http://localhost:5000` üzerinden çalışır.

> **Not**
>
> - Sanal ortam etkinleştirilmeden `python` ve `pip` komutları bazı sistemlerde bulunmayabilir; bu durumda `python3` ve `pip3` kullanılmalıdır.
> - macOS'ta `5000` portu sistemin AirPlay Receiver servisi tarafından kullanılabilir. 


## Ortam Değişkenleri (.env)
- `VIDEO_SOURCE` — kamera index'i veya video dosyası yolu (varsayılan `0`)
- `DETECTION_BACKEND` — `yolo_pytorch` (varsayılan) veya `mediapipe`
- `EYE_LOCATOR_BACKEND` — göz bölgesi MediaPipe landmark ile mi (`mediapipe`, varsayılan) yoksa Haar Cascade ile mi (`haar`) bulunur
- `DROWSINESS_SECONDS` — gözler kaç saniye kesintisiz kapalı kalırsa alarm tetiklenir (varsayılan `2.5`)
- `TORCH_DEVICE` — modelin çalışacağı cihaz (`cpu`, `mps`, `cuda`)
- `ALARM_COOLDOWN_SECONDS` — yeni bir alarm döngüsünün en az kaç saniyede bir sayılacağı
- `EYE_MEASUREMENT_GRACE_SECONDS` — kişi görünürken göz durumunun ölçülemediği serinin kaç saniye tolere edileceği; aşılırsa sistem `UNVERIFIABLE` durumuna geçer (varsayılan `0.35`)
- `CORS_ORIGINS` — izin verilen origin adresleri

Yukarıdakiler en sık değiştirilen ayarlardır; kullanılabilir değişkenlerin tamamı açıklamalarıyla birlikte `.env.example` içindedir.

Ayrıca `FLASK_DEBUG` (varsayılan `false`) yalnızca `python run.py` ile yerel çalıştırmayı etkiler. Flask'in debug modu, hata sayfası üzerinden sunucuda kod çalıştırılabilen interaktif bir konsol açar ve uygulama `0.0.0.0` dinlediği için bu konsol aynı ağdaki diğer makinelere de görünür olur; bu nedenle varsayılan olarak kapalıdır. Geliştirme sırasında `FLASK_DEBUG=true python run.py` ile açılabilir.


## Çalışma Mantığı
Sistem çalıştırıldıktan sonra kamera açılır.
1. Kameradan görüntü alınır.
2. YOLO modeli ile insan tespiti yapılır.
3. İnsan tespit edilirse yüz bölgesi analiz edilir.
4. MediaPipe Face Mesh kullanılarak göz noktaları bulunur.
5. Göz görüntüsü kırpılır.
6. PyTorch modeli gözün açık veya kapalı olduğunu tahmin eder.
7. Sonuçlar zaman bazlı değerlendirilir.
8. Gözler belirlenen süreden fazla kapalı kalırsa alarm çalıştırılır.
9. Sonuçlar REST API üzerinden React arayüzüne gönderilir.


## YOLOv8 Kullanımı
YOLO modeli sistem içerisinde kişi tespiti amacıyla kullanılmaktadır.
- Kamerada insan bulunup bulunmadığı anlaşılır.
- Gereksiz görüntüler işlenmez.

## MediaPipe Face Mesh
YOLO tarafından kişi tespit edildikten sonra yüz analizi MediaPipe Face Mesh ile yapılmaktadır.
Noktalar kullanılarak;
- Sağ göz
- Sol göz
konumları belirlenmektedir.
MediaPipe kullanılmasının nedeni;
- Daha yüksek doğruluk
- Daha kararlı göz tespiti
- Baş hareketlerinden daha az etkilenmesidir.

## Göz Kırpma Analizi
MediaPipe tarafından bulunan göz noktaları kullanılarak göz bölgesi kırpılır.
Kırpılan görüntü normalize edilir.
Daha sonra PyTorch modeline gönderilir.
Model iki sınıf üretmektedir: `Open` ve `Closed`.

## PyTorch Modeli
PyTorch modeli göz görüntülerini sınıflandırmaktadır.
Model `Open` ve `Closed` etiketleri ile eğitilmiştir.
Model yalnızca göz görüntüsünü kullanmaktadır.
Çıkış olarak `Open` veya `Closed` sonucu üretmektedir.

## Uyku Tespit Algoritması
Sistem zaman tabanlı çalışmaktadır.
Algoritma;
```text
Göz Açık
↓
Sayaç Sıfırlanır
↓
Göz Kapalı
↓
Sayaç Artar
↓
Belirlenen Süre Aşıldı mı?
↓
Evet
↓
Alarm Çalıştır
```

## Alarm Sistemi
Alarm sistemi bağımsız bir servis olarak geliştirilmiştir.
Süre eşiği aşılıp göz kapalılığı devam ettiği sürece alarm sesi kesintisiz çalar.
Kullanıcı gözlerini açtığı anda alarm otomatik olarak durur ve sistem normal moda döner.

## REST API
Backend ile frontend arasında REST API kullanılmaktadır.

| Endpoint | Açıklama |
| --- | --- |
| `GET /api/health` | Sistem durumu |
| `GET /api/stream` | Canlı görüntü (MJPEG) |
| `GET /api/detection/status` | Anlık tespit sonucu |
| `GET /api/detection/alarm-history` | Alarm geçmişi |
| `GET /api/detection/alarm-snapshot` | En son alarmın anındaki görüntü |
| `GET /api/detection/alarm-snapshot/<id>` | Geçmişteki belirli bir alarmın görüntüsü |


## Frontend
Frontend React ve Vite kullanılarak geliştirilmiştir. Backend ile REST API üzerinden haberleşmektedir. Kullanıcı arayüzünde canlı kamera görüntüsü, FPS bilgisi, sistem durumu, göz durumu ve alarm geçmişi gösterilmektedir. React belirli aralıklarla backend API'sini sorgulayarak güncel bilgileri kullanıcıya yansıtmaktadır.

## Testler
Backend tarafında Pytest kullanılarak birim ve entegrasyon testleri gerçekleştirilmiştir. Kamera yönetimi, göz tespiti hattı, uyku tespit algoritması, alarm servisi ve API endpointleri test edilmiştir. Ayrıca veri seti bölme (kişi bazlı ayrımın sızıntı üretmemesi) ve model değerlendirme metrikleri de test kapsamındadır. Frontend tarafında ise Vitest ile testler uygulanmıştır.

Testler şu komutlarla çalıştırılır:
```bash
cd backend && python -m pytest      # 458 test
cd frontend && npm test             # 61 test
```

## Veri Seti ve Model Değerlendirmesi

Göz sınıflandırıcısı MRL Eye Dataset üzerinde eğitilmiştir. Veri seti
`scripts/prepare_eye_dataset.py` ile hazırlanır.

**Bölme kişi bazlıdır.** MRL veri setinde aynı deneğin aynı oturumdan
alınmış yüzlerce neredeyse aynı karesi bulunur. Görüntü bazında rastgele
bölme yapılırsa bir kişinin komşu kareleri hem eğitim hem doğrulama
setine düşer; model o kişiyi ezberler ve doğrulama skoru "yeni bir
sürücüye genelleme" yerine "gördüğü kişiyi hatırlama" başarımını ölçer.
Bu nedenle bölme deneğe göre yapılır: bir kişinin tüm kareleri tek bir
tarafta kalır, train ve val setleri ortak denek içermez.

```bash
cd backend
python scripts/prepare_eye_dataset.py --source data/raw/mrlEyes_2018_01 --output data/eyes
python scripts/train_eye_classifier.py --data-dir data/eyes --epochs 20
```

`--split-by image` seçeneği eski (sızıntılı) davranışı yeniden üretir ve
yalnızca iki yöntemin farkını göstermek için tutulmaktadır; raporlanan
sonuçlarda kullanılmamalıdır.

### Sonuçlar

Model, 37 deneğin 25'i eğitim (66.449 görüntü), 12'si doğrulama (18.449
görüntü) olacak şekilde ayrılmış veri seti üzerinde 20 epoch eğitilmiş,
en iyi sonuç epoch 12'de elde edilmiştir. Doğrulama setindeki denekler
eğitimde hiç görülmemiştir.

| Metrik | Değer |
| --- | --- |
| Doğruluk (accuracy) | 0.9522 |
| `closed` precision / recall / F1 | 0.8875 / 0.9780 / 0.9306 |
| `open` precision / recall / F1 | 0.9887 / 0.9396 / 0.9635 |
| False-awake oranı (kapalı gözü "açık" sanma) | 0.0220 |
| False-alarm oranı (açık gözü "kapalı" sanma) | 0.0604 |

Confusion matrix (satır = gerçek, sütun = tahmin):

```text
                 tahmin=closed  tahmin=open
  gerçek=closed          5911          133
  gerçek=open             749        11656
```

Bu değerler tek bir kareye ilişkindir. Sistem kararı tek kareye değil,
`EyeStateSmoother` penceresine (7 geçerli karar, %60 çoğunluk) ve
`DROWSINESS_SECONDS` süre eşiğine dayandığı için, tek karelik
false-awake hatalarının alarm davranışına etkisi bu orandan daha
düşüktür.

Sistemin bir bütün olarak nasıl davrandığı ayrıca ölçülmüştür; bkz.
"Uçtan Uca Sistem Değerlendirmesi".

**Model seçiminin gerekçesi.** Eğitim boyunca en yüksek `closed_recall`
epoch 5'te görülmüştür (0.9957, false-awake yalnızca 0.0043) — fakat
aynı epoch'ta doğruluk 0.8606'ya, `closed_f1` 0.8240'a düşmüştür: model
kapalı gözlerin neredeyse tamamını yakalarken açık gözleri de "kapalı"
saymaya başlamış, yani sürekli yanlış alarm üreten bir davranışa
kaymıştır. Bu, model seçiminin neden tek bir metriğe değil `closed_f1`e
bağlandığını somut olarak göstermektedir.

**Değerlendirme yalnızca accuracy'ye dayanmaz.** Bu sistemde iki hata
türünün maliyeti eşit değildir: kapalı gözü "açık" sanmak (false-awake)
uyuklayan sürücünün uyarılmaması demektir, tersi ise yalnızca gereksiz
bir alarma yol açar. Bu yüzden eğitim script'i her epoch'ta sınıf bazlı
precision/recall/F1 ile `false_awake_rate` ve `false_alarm_rate`
değerlerini raporlar ve bunları `weights/eye_classifier_metrics.json`
dosyasına kaydeder. Kaydedilecek "en iyi" epoch varsayılan olarak kapalı
göz sınıfının F1'ine göre seçilir (`--select-by closed_f1`); eski
davranış `--select-by accuracy` ile korunmuştur.

**Sürücü yüzünü çevirdiğinde.** YOLO kişi tespiti yapıyor, yüz tespiti
değil. Sürücü başını çevirdiğinde gövdesi kadrajda kaldığı için kişi
kutusu bulunmaya devam ediyor. İlk halinde sistem bu durumda hiç göz
ölçümü alamadığı halde "uyanık" diyordu — uyku tespiti yapan bir sistem
için yanlış tarafta bir hata.

Bunu iki adımda düzelttim. Önce göz bölgesinin bulunamadığı ardışık
kareleri sınırladım, ama bu yeterli değildi: göz KUTUSU bulunup içindeki
crop'ların hepsi geçersiz çıkabiliyor (çok küçük kutu, sınıflandırma
hatası, düşük güvenli tek göz) ve o durumda sayaç sıfırlandığı için aynı
hata bir katman derinde duruyordu. Karar artık kutunun varlığına değil,
geçerli bir sınıflandırma alınıp alınmadığına (`measurement_valid`)
bağlı.

Ayrıca "kadrajda kimse yok" ile "kişi var ama gözü göremiyorum" artık
ayrı durumlar: `NO_FACE` ve `UNVERIFIABLE`. İkisinin sebebi ve kullanıcıdan
beklenen tepki farklı — birinde kadraja girmek, diğerinde yüzü kameraya
dönmek gerekiyor.

**Toleranslar saniye cinsinden.** Ölçümsüz karelerin toleransı önce kare
sayısıydı (`EYE_LOCATOR_MISS_TOLERANCE=2`). Bir karenin süresi FPS'e
bağlı olduğu için bu, 30 FPS'te 67 ms, 5 FPS'te 400 ms demekti — aynı
ayar makineden makineye 6 kat farklı bir zaman toleransı üretiyordu.
`EYE_MEASUREMENT_GRACE_SECONDS` (saniye) ile CPU/MPS/CUDA arasında aynı
davranış sağlanıyor.

Aynı sorun `EyeStateSmoother`'ın penceresinde de vardı ve orada daha
büyük bir etki yaratıyordu. Uçtan uca FPS taraması bunu ölçtükten sonra
smoother'ın üç ayarı da saniye cinsine çevrildi — bkz. "FPS
dayanıklılığı".

**API'de kimlik doğrulaması yok.** `/api/stream`, alarm görüntüsü ve
alarm geçmişi endpoint'leri canlı kamera görüntüsü sunuyor ve hiçbiri
kimlik sormuyor. Bu yüzden backend portunu Docker'da dışarı açmıyorum;
her şey nginx arkasından, `http://localhost:3000` üzerinden çalışıyor.

**Alarm kayıtları kalıcı değil.** Son 5 alarmın bilgisi ve görüntüsü
sadece bellekte (`SharedState`) duruyor, backend yeniden başlayınca
siliniyor. Amaç canlı izleme ve demo olduğu için böyle bıraktım. 

**Health endpoint sadece "ayakta mıyım" diyor.** `GET /api/health`
Flask process'inin çalıştığını doğruluyor, kameranın açık olup olmadığına
veya modelin yüklenip yüklenmediğine bakmıyor. O bilgiler
`GET /api/detection/status` içinde `CAMERA_ERROR` / `MODEL_ERROR` olarak
zaten var. 

**Engine kapanırken thread'in bittiği doğrulanmıyor.**
`DetectionEngine.stop()` arka plan thread'ini `join(timeout=5)` ile
bekliyor ama süre dolduğunda thread gerçekten kapandı mı diye tekrar
kontrol etmiyor. 

**Model kontrollü koşullarda ölçüldü.** Başarım değerleri MRL Eye
Dataset üzerinde elde edildi. Gözlük, düşük ışık, gece sürüşü ve farklı
kamera açıları bu veri setinde sınırlı ölçüde temsil ediliyor.

**Tek kişi varsayımı.** Kadrajda birden fazla kişi olduğunda YOLO en
yüksek güvenli kutuyu seçiyor, MediaPipe ise kendisine verilen kişi
kutusunu kullanmayıp tüm karede yüz arıyor. Bunu bilerek böyle bıraktım.

## Uçtan Uca Sistem Değerlendirmesi

Yukarıdaki %95.22 doğruluk, göz sınıflandırıcısının **tek bir kare**
üzerindeki başarımıdır. Sistemin gerçekte yapması gereken iş farklıdır:
sürücü gözünü kapattığında alarm çalmak. Bu bölüm o soruyu ölçer —
denek gözünü kapattığında alarm çaldı mı, kaç saniye sonra çaldı,
dakikada kaç yanlış alarm üretti?

### Yöntem

Etiketli videolar `scripts/evaluate_system.py` ile gerçek dedektörden
geçirilir. Üç tasarım kararı ölçümün güvenilirliğini belirler:

**Ölçüm deterministiktir.** Zaman gerçek saatten değil kare indeksinden
türetilir (`evaluation/clock.py`). `DrowsinessService` ve `AlarmService`
zaten enjekte edilebilir bir `clock` parametresi alıyordu; aynı seam
kullanıldı. Aynı girdi her zaman aynı sonucu verir ve sonuç makinenin
hızından etkilenmez.

**İşleme hızı simüle edilir.** Gerçekte sistem CPU'da 30 FPS'lik
görüntüyü ~8 FPS işler, yani kare atlar. Kare atlama, zaman
eşiklerinin kaç ölçümle dolduğunu değiştirir. `--target-fps`
hedeflenen işleme hızını taklit eder.

**Etiketler dedektörden bağımsızdır.** Olay sınırları videoya bakılarak
konur, sistemin çıktısına bakılarak değil. Sistemin "kapalı" dediği
yerleri doğru etiket kabul etmek, onu kendi cevap kâğıdıyla sınamak
olurdu: kaçırdığı her olay etikette de eksik kalır ve tespit oranı
yapay olarak şişerdi.

### Metrik tanımları

Bir alarmın **isabet** sayılması için şu pencereye düşmesi gerekir:

```text
[olay_başlangıcı + DROWSINESS_SECONDS,  olay_bitişi + tolerans]
```

Alt sınırın eşik kadar kaydırılması kritiktir: bu sınırdan **önce**
gelen alarm "erken isabet" değil **yanlış alarm**'dır. Aksi halde her
karede alarm çalan bozuk bir sistem %100 tespit oranı alırdı.

| Metrik | Tanım |
| --- | --- |
| Tespit oranı | İsabet alan olay / alarm üretmesi mümkün olan olay |
| Alarm gecikmesi | Eşik dolduktan sonra alarmın gelmesi için geçen süre |
| Yanlış alarm/dk | Paydası klibin tamamı değil, alarmın **meşru olmadığı** süre |
| Kör kare oranı | `NO_FACE` + `UNVERIFIABLE` + hata karelerinin oranı |

Eşikten **kısa** olaylar tespit oranına dahil edilmez: 2.5 saniye eşikte
2.3 saniyelik bir kapanmanın alarm üretmemesi doğru davranıştır. Bu
olaylar ayrıca sayılır, çünkü genelde etiketleme hatasına işaret ederler.

Kör kare oranı ayrı raporlanır, çünkü sistem karelerin yarısında
sürücüyü hiç göremediyse kalan yarıdaki yüksek tespit oranı güven
vermez.

### Sonuçlar

`backend=yolo_pytorch · drowsiness_seconds=2.5 · target_fps=8 · match_tolerance=0.5`

| Koşul | Klip | Tespit | Gecikme ort/med/p95 (sn) | Yanlış alarm/dk | Kör kare |
| --- | ---: | ---: | ---: | ---: | ---: |
| normal | 1 | 100.0% (4/4) | 0.50 / 0.58 / 0.63 | 1.62 | %0.0 |
| headturn | 1 | 100.0% (2/2) | 0.50 / 0.50 / 0.51 | 1.81 | %0.6 |
| **TOPLAM** | 2 | **100.0% (6/6)** | **0.50 / 0.55 / 0.62** | **1.71** | **%0.3** |

Altı uyku olayının tamamı yakalanmıştır. Gecikmeler 0.20–0.64 saniye
aralığında sıkı bir kümededir; bağımsız olaylarda bu tutarlılık,
etiketlerin sistemin ölçtüğüyle hizalandığının bir göstergesidir.

Aynı klipler MediaPipe/EAR yedek backend'iyle de koşulmuştur.
Karşılaştırma, eğitilen PyTorch sınıflandırıcısının katkısını uçtan uca
gösterir:

| | mediapipe/EAR | yolo_pytorch |
| --- | ---: | ---: |
| Tespit | 57.1% (4/7) | **100.0% (6/6)** |
| Yanlış alarm/dk | 4.63 | **1.71** |
| Kör kare | %1.8 | **%0.3** |

### İki yanlış alarmın kök nedeni

Gözlenen iki yanlış alarm rastgele değildir; ikisi de gözlerin doğrudan
görünmediği anlarda ortaya çıkmıştır:

| Zaman | Durum | Ne oluyor |
| --- | --- | --- |
| 53.0 sn | Baş öne eğik, gözler kadrajda yok | Sistem `UNVERIFIABLE`'a geçmiyor, "gördüm ve kapalı" diyor |
| 20.8 sn | Esnerken gözler kısılıyor | Kısık göz "kapalı" sayılıyor, süre eşiği doluyor |

Kör kare oranının %0.3 gibi düşük olması bu tabloyu tamamlar: sorun
"ölçemiyorum" değil, **"emin bir şekilde yanlış ölçüyorum"**. Eklenen
`measurement_valid` mekanizması burada devreye girmez, çünkü ölçüm
alınmaktadır — yalnızca yanlıştır. MediaPipe göz kutusunu bulur,
sınıflandırıcı yüksek güvenle cevap verir.

Bu, aşağı bakan, kısık ve kısmen örtülü gözlerin MRL eğitim setinde
yeterince temsil edilmediğine işaret eder. Eğitim transformu şu an
yalnızca gri tonlama, yeniden boyutlandırma ve normalizasyondan
oluşmaktadır; parlaklık/kontrast değişimi, hafif rotasyon ve kısmi örtme
gibi veri artırma teknikleri bu bulgunun doğrudan işaret ettiği ilk
iyileştirmedir.

### FPS dayanıklılığı

Aynı klipler 5, 10, 15, 20 ve 30 FPS'te koşuldu. Amaç şu soruyu
cevaplamaktı: sistemin davranışı çalıştığı makinenin hızına bağlı mı?

Kontrollü karşılaştırma için üç koşu yapıldı — tek fark
`EyeStateSmoother`: **A** kare tabanlı varsayılan ayarlarla, **B**
smoother etkisizleştirilerek, **C** smoother zaman tabanlı ayarlara
çevrildikten sonra.

| FPS | A gecikme (sn) | B gecikme (sn) | C gecikme (sn) |
| ---: | ---: | ---: | ---: |
| 5 | 0.88 | 0.10 | 0.46 |
| 10 | 0.37 | 0.06 | 0.47 |
| 15 | 0.28 | 0.04 | 0.42 |
| 20 | 0.09 | 0.04 | 0.40 |
| 30 | 0.09 | 0.04 | 0.40 |
| **yayılım** | **0.79** | **0.06** | **0.07** |

(Değerler ortalama gecikmedir. A ve B, birim düzeltmesinden önceki
kodla koşulmuştur.)

Smoother açıkken alarm gecikmesi 5→30 FPS aralığında **0.79 saniye**
kayıyordu; kapalıyken aynı kayma yalnızca 0.06 saniyeydi. Bu fark
smoother'ın kendisinden geliyordu: pencere 7 **kare** genişliğindeydi
ve bir karenin süresi FPS'e bağlı olduğu için 5 FPS'te 1.4 saniyeyi,
30 FPS'te 0.23 saniyeyi kapsıyordu.

Sonuç olarak sistem 5–10 FPS'te en iyi çalışıyordu, ama bu bir tasarım
tercihi değil tesadüftü — 7 karenin o hızlarda sensible bir süreye denk
gelmesi. Daha hızlı bir makinede yumuşatma sessizce zayıflıyor ve
kimse sebebini göremiyordu.

Bu bulgu üzerine `EyeStateSmoother`'ın üç ayarı da saniye cinsine
çevrildi (`EYE_SMOOTHER_WINDOW_SECONDS=0.9` vb.). Varsayılanlar eski
kare değerlerinin ~8 FPS'teki karşılıkları; yani mevcut davranış
korundu, benchmark skoruna göre ayarlanmadı. 6 olaylık bir sette
parametre optimizasyonu yapmak gürültüye fit etmek olurdu — yapılan
şey ayarın DEĞERİNİ değil BİRİMİNİ düzeltmekti.

Aynı tarama düzeltmeden sonra tekrarlandı (**C**): yayılım 0.79
saniyeden **0.07 saniyeye** indi, yani smoother'ın hiç olmadığı
B koşusunun seviyesine (0.06). Gecikme artık FPS'e değil yalnızca
smoother'ın kendi penceresine bağlı — beklenen davranış bu.

Ölçülen tek şey yayılımdır. Aynı koşuda tespit oranı da A'da
15 FPS'ten sonra 1.00'den 0.83'e düşerken C'de beş hızda da 1.00
çıkmıştır, ancak bu 6 olaylık sette tek bir olayın oynaması demektir
ve aşağıdaki sınırlamalar gereği kanıt sayılmamıştır.

Not: gecikme *yayılımı* etiket sınırlarındaki belirsizlikten
etkilenmez (sabit bir kayma tüm gecikmeleri aynı miktarda öteler), bu
yüzden 6 olaylık örneklemde bile güvenilir bir ölçüttür. Aynı koşudaki
tespit oranı ve yanlış alarm farkları ise güvenilir değildir: B'nin
"yanlış alarmları" gerçek olayların içinde ama etiket eşiğinden 0.1–0.5
saniye önce gelmiştir ve bu, etiket hassasiyetinin (±0.3 sn) altındadır.

### Sınırlamalar

Bu sonuçların kapsamı dardır ve rakamlar bu sınırlar içinde okunmalıdır:

- **Tek denek, iki klip, altı olay.** Bu örneklemde tespit oranındaki
  birkaç puanlık farklar istatistiksel olarak anlamsızdır. Gözlüklü ve
  düşük ışıklı koşullar ile ek denekler ölçülmemiştir.
- **Kapanmalar oynanmıştır.** Gerçek uykulu sürüş kaydetmek etik olarak
  mümkün değildir; bu literatürdeki standart uzlaşmadır ama taklit
  kapanmalar gerçek mikro-uykulardan genellikle daha keskin ve uzundur.
- **Etiketler koşudan sonra düzeltilmiştir.** İlk koşuda görünen bir
  kaçırma ve bir gecikme aykırısının ikisi de etiketleme hatası çıktı
  (bir göz kırpması olay başlangıcı sanılmış, bir kapanma eşiğin
  altındayken 3.8 sn diye işaretlenmişti). Düzeltmeler kare
  görüntülerine dayandırılmıştır, sistemin kararına değil; yine de
  doğru sıra etiketlemeyi koşudan önce bitirmektir.

### Tekrar üretme

```bash
cd backend
python scripts/evaluate_system.py \
    --clips ../benchmark/clips \
    --labels ../benchmark/labels \
    --target-fps 8 \
    --output ../benchmark/results/kosu.json
```

Koşu, hangi backend'in gerçekte kurulduğunu ekrana ve rapora yazar.
Bu kontrol bilinçlidir: `ALLOW_DETECTION_FALLBACK` varsayılanı `true`
olduğu için torch/ultralytics kurulu değilse ya da ağırlık bulunamazsa
sistem sessizce MediaPipe'a düşer ve koşu sorunsuz görünür. Ölçüm
aracının yapabileceği en kötü hata, yanlış sistemin sayılarını doğru
sistemin sayısı gibi sunmaktır.

Çekim protokolü, etiket şeması ve gizlilik notları için bkz.
`benchmark/README.md`.

## Sonuç
Bu çalışmada, gerçek zamanlı çalışan ve Docker ile tek komutta ayağa kalkan bir sürücü uyku tespit sistemi geliştirdim. YOLOv8 ile kişi tespiti, MediaPipe Face Mesh ile göz bölgesinin bulunması ve kendi eğittiğim PyTorch modeliyle gözün açık/kapalı sınıflandırılması bir araya getirildi; gözler belirlenen süre boyunca kapalı kalınca sistem sesli alarm veriyor. Göz sınıflandırıcısı, eğitimde hiç görmediği 12 denek üzerinde %95 doğruluk ve 0.9780 kapalı-göz recall değeri verdi.

Geliştirme sürecinin bana en çok şey öğreten kısmı ölçüm tarafı oldu. Veri setini önce görüntü bazında bölmüştüm ve model %98.9 gibi bir doğruluk veriyordu; aynı kişinin karelerinin hem eğitimde hem doğrulamada yer aldığını fark edip bölmeyi kişi bazlı yapınca rakam %95.2'ye indi. Düşük görünen bu sayı, modelin daha önce hiç görmediği insanlar üzerindeki gerçek başarımı. Benzer şekilde model seçimini toplam doğruluk yerine kapalı göz sınıfının F1 skoruna bağladım, çünkü bu sistemde kapalı gözü "açık" sanmakla açık gözü "kapalı" sanmanın sonucu aynı değil.

Aynı ayrım bir kez daha karşıma çıktı: sınıflandırıcının tek kare
başarımı, sistemin başarımı demek değil. Bunu ölçmek için etiketli
videolar üzerinde uçtan uca bir değerlendirme hattı kurdum. İlk ölçümde
altı uyku olayının tamamı ortalama 0.50 saniye gecikmeyle yakalandı,
ancak dakikada 1.71 yanlış alarm gözlendi. Bu yanlış alarmların ikisi de
gözlerin doğrudan görünmediği anlarda (baş öne eğik, esneme sırasında
kısılma) ortaya çıktı — yani sorun "ölçemiyorum" değil, "emin bir
şekilde yanlış ölçüyorum". Bu bulgu, bir sonraki adımın yeni bir mimari
değil, eğitim verisinin çeşitlendirilmesi olduğunu söylüyor.

Ölçüm hattının kendisi de iki hatayı ortaya çıkardı ve ikisi de benim
etiketlerimdeydi: bir göz kırpmasını olay başlangıcı sanmışım, eşiğin
altında kalan bir kapanmayı da olduğundan uzun işaretlemişim.
Düzeltilince sistemin her iki durumda da doğru davrandığı görüldü. Bir
ölçüm aracının ilk işi, ölçtüğü sistemi değil kendisini
sorgulatabilmekmiş.

Sistem modüler bir yapıda kurulduğu için ilerleyen aşamalarda tespit modelleri değiştirilebilir.


## Public repository contents

Personal camera recordings, benchmark video clips, training images, local `.env` settings and delivery archives are excluded from this public copy. The trained eye classifier and evaluation labels are included. To reproduce video evaluations, provide your own clips following `benchmark/README.md`; local recording paths shown in examples are not bundled.
