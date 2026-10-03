# Uçtan Uca Değerlendirme Seti

Bu klasör, sistemin **tek kare sınıflandırma başarımını değil**, gerçek
davranışını ölçer: denek gözünü kapattığında alarm çaldı mı, kaç saniye
sonra çaldı, dakikada kaç yanlış alarm üretti?

```
benchmark/
├── clips/     videolar        (git'e girmez — bkz. Gizlilik)
├── labels/    JSON etiketler  (git'e girer)
└── results/   koşu çıktıları  (git'e girmez)
```

## Çalıştırma

```bash
cd backend
python scripts/evaluate_system.py \
    --clips ../benchmark/clips \
    --labels ../benchmark/labels \
    --target-fps 8 \
    --output ../benchmark/results/kosu.json
```

`--target-fps` **kozmetik değil.** Gerçekte sistem CPU'da 30 FPS'lik
görüntüyü ~8 FPS işler, yani kare atlar. Kare atlama, zaman
eşiklerinin kaç ölçümle dolduğunu ve dolayısıyla yumuşatmanın ne kadar
veriye dayandığını değiştirir. Hedeflediğin işleme hızını ver. Kendi
makinendeki gerçek hızı dashboard'daki FPS göstergesinden
okuyabilirsin.

Ölçüm **deterministiktir**: zaman gerçek saatten değil kare indeksinden
türetilir, dolayısıyla aynı girdi her zaman aynı sonucu verir.

## Klip çekim protokolü

Hedef: 4 koşul × 3-5 denek, klip başına ~60 saniye.

| Koşul | İçerik |
| --- | --- |
| `normal` | İyi ışık, gözlüksüz, baş sabit |
| `glasses` | Gözlüklü — ışığı yandan ver ki parlama olsun |
| `lowlight` | Karartılmış oda / gece sürüşü benzeri |
| `headturn` | Yana bakma, öne eğilme, kadraj dışına çıkma |

Her klipte **hem gerçek olay hem çeldirici** olmalı:

- **Gerçek olay:** 3-5 saniye kesintisiz göz kapatma (eşikten uzun).
- **Çeldirici:** normal göz kırpma, kısa aşağı bakış, esneme.

Çeldiriciler olmadan yanlış alarm oranı ölçülemez ve benchmark'ın
yarısı işe yaramaz.

## Etiketleme

Elle JSON yazma — `scripts/label_clips.py` videoyu kare kare gezdirir:

```bash
cd backend
python scripts/label_clips.py ../benchmark/clips/glasses_01.mp4 \
    --subject S03 --conditions glasses daylight
```

| Tuş | İşlev |
| --- | --- |
| `boşluk` | oynat / duraklat |
| `←` `→` | bir kare geri / ileri |
| `,` `.` | bir saniye geri / ileri |
| `a` `d` | beş saniye geri / ileri |
| `c` | olay **başlangıcı** işaretle |
| `o` | olay **bitişi** işaretle |
| `b` | son olayı çeldirici yap / geri al |
| `u` | son işlemi geri al |
| `s` | kaydet |
| `q` | çık |

Üst şeritte işaretlenen aralıklar görünür: yeşil gerçek olay, mavi
çeldirici, açık mavi çerçeve kaydedilmekte olan aralık.

Araç **kaydetmeden önce** dosyayı değerlendirme koşusunun kullandığı
doğrulamadan geçirir. Çakışan olay ya da klip sonunu aşan aralık diske
hiç yazılmaz — "etiketledim ama koşu patlıyor" durumu oluşmaz.
Kaydederken ayrıca eşikten kısa olaylar ve çeldirici eksikliği uyarı
olarak bildirilir.

Yarım kalan bir klibe `--resume` ile devam edilebilir.

## Etiket biçimi

`labels/` altında klip başına bir JSON (araç bunu üretir; elle de
yazılabilir):

```json
{
  "clip": "glasses_01.mp4",
  "subject": "S03",
  "conditions": ["glasses", "daylight"],
  "fps": 30.0,
  "duration_s": 62.0,
  "events": [
    {"start_s": 10.2, "end_s": 14.0},
    {"start_s": 31.5, "end_s": 32.1, "expect_alarm": false, "label": "blink"}
  ]
}
```

- `clip` — `clips/` altındaki dosya adı.
- `conditions` — bir klip birden fazla koşul taşıyabilir; her birinin
  rapor satırına katkı verir. Bu yüzden **koşul satırlarının toplamı
  TOPLAM satırına eşit olmak zorunda değildir.**
- `expect_alarm` varsayılan `true`. `false` verilenler çeldiricidir:
  tespit oranına girmezler ama içlerine düşen alarm yanlış alarm sayılır.
- `fps` — dosyadan okunana göre önceliklidir. Bazı konteynerlerde
  OpenCV yanlış kare hızı bildirir ve tüm zaman hesabı buna dayanır.

Doğrulama katıdır: alarm beklenen iki olay çakışamaz, olaylar klip
süresini aşamaz. Etiket hatası ölçümü sessizce bozacağı için erken
yakalanır.

## Metriklerin tanımı

Bir alarmın **isabet** sayılması için şu pencereye düşmesi gerekir:

```
[start_s + DROWSINESS_SECONDS,  end_s + match_tolerance]
```

Alt sınırın eşik kadar kaydırılması kritik: bu sınırdan **önce** gelen
alarm "erken isabet" değil **yanlış alarm**'dır. Aksi halde her karede
alarm çalan bozuk bir sistem %100 tespit oranı alırdı.

| Metrik | Anlamı |
| --- | --- |
| `detection_rate` | İsabet alan olay / alarm üretmesi mümkün olan olay |
| `alarm_latency` | Eşik dolduktan sonra alarmın gelmesi için geçen süre |
| `false_alarms_per_minute` | Paydası klibin tamamı değil, alarmın **meşru olmadığı** süre |
| `blind_frame_ratio` | `NO_FACE` + `UNVERIFIABLE` + hata karelerinin oranı |

`blind_frame_ratio` ayrı raporlanır çünkü sistem karelerin yarısında
sürücüyü hiç göremediyse, kalan yarıdaki %100 tespit güven vermez.

### Neden tek bir genel ortalama yetmez

Rapor hem **koşul** hem **denek** bazında kırılım üretir.

Koşul kırılımı, gözlüklü kullanımın zayıf kalmasının toplam ortalamada
gizlenmesini engeller. Ama tek başına yeterli değil: sistem belirli bir
KİŞİDE kötü çalışıyorsa (göz şekli, gözlük çerçevesi, ten tonu), bu
sorun o kişinin bulunduğu tüm koşullara dağılarak seyrelir ve hiçbir
koşul satırında görünmez. Denek kırılımı bunu açar.

Bir fark: bir klip birden fazla koşul taşıyabildiği için koşul
satırlarının toplamı TOPLAM'a eşit **değildir**. Her klip tek bir
deneğe ait olduğu için denek satırlarının toplamı TOPLAM'a eşittir.

Oranlar her zaman ham sayılarla birlikte gösterilir (`%80.0 (4/5)`).
Koşul başına 4-5 olayla çalışıldığı için çıplak yüzde yanıltıcı olur:
4/5 ile 40/50 aynı %80'dir ama güven düzeyleri çok farklıdır. Toplam
olay sayısı 20'nin altındaysa rapor bunu ayrıca uyarır.

Eşikten **kısa** olaylar tespit oranına dahil edilmez: 2.5 saniye eşikte
1.5 saniyelik bir kapanmanın alarm üretmemesi doğru davranıştır. Rapor
bunları ayrıca sayar, çünkü genelde etiketleme hatasına işaret ederler.

## Parametre taraması

Benchmark hazır olduğunda config'deki her sihirli sayı veriyle
savunulabilir hale gelir:

```bash
python scripts/evaluate_system.py ... --sweep DROWSINESS_SECONDS=1.5,2.0,2.5,3.0
python scripts/evaluate_system.py ... --sweep EYE_SMOOTHER_WINDOW_SECONDS=0.5,0.9,1.4
python scripts/evaluate_system.py ... --sweep target_fps=5,10,15,20,30
```

`target_fps` dışındaki adlar Config ayarı olarak ele alınır ve gerçekten
var olup olmadıkları doğrulanır — `DROWSINESS_SECOND` gibi bir yazım
hatası sessizce hiçbir şey yapmaz, N kez aynı sonucu üretir ve
"parametrenin etkisi yok" diye yanlış bir sonuca götürürdü.

"Neden 2.5 saniye?" sorusunun cevabı artık "canlı testte ayarladım"
değil, tespit oranı ↔ yanlış alarm eğrisi olur.

Aynı araç FPS dayanıklılığını da ölçer: `--sweep target_fps=5,10,15,20,30`
ile aynı klipleri koşturmak, zaman eşiklerinin gerçekten makine
hızından bağımsız olup olmadığını gösterir. Bu tarama bir kez
`EyeStateSmoother`'ın kare bazlı penceresini yakalayıp saniye bazına
çevirmemizi sağladı (bkz. ana README, "FPS dayanıklılığı").

## Gizlilik

**Klipler git'e girmez.** İnsan yüzü içeren videolar; `clips/` ve
`results/` `.gitignore` kapsamındadır.

- Deneklerden kaydın amacı ve saklanma süresi konusunda açık onay alın.
- Klipleri repoya, bulut deposuna veya paylaşımlı bir sürücüye
  koymayın; yerel tutun.
- Yayımlanan raporda yüz görüntüsü değil, yalnızca sayısal sonuçlar yer
  almalı.

## Bir sınırlama: taklit uyku

Gerçek uykulu sürüş kaydetmek etik olarak mümkün değil, bu yüzden göz
kapanmaları **oynanmıştır**. Bu, literatürdeki standart uzlaşma ama
sonuçların yorumunu etkiler: taklit kapanmalar genellikle gerçek
mikro-uykulardan daha keskin ve daha uzundur. Raporda bu açıkça
belirtilmelidir.

Dış geçerlilik için NTHU-DDD gibi hem gerçek hem oynanmış uyuklama
içeren halka açık bir veri seti ikinci bir doğrulama seti olarak
eklenebilir.
