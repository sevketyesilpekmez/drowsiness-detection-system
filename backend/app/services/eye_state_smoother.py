from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

Clock = Callable[[], float]


class EyeStateSmoother:
    """
    Ardışık karelerdeki göz-kapalı kararlarını zamansal olarak yumuşatır.

    Gerçek bir webcam testinde tek bir hatalı/eksik karenin (Haar'ın gözü
    kaçırması, tek karelik yanlış sınıflandırma) birikmiş kapalı-göz
    süresini anında sıfırladığı görüldü (bkz. README "Zaman toleransı").
    Bu sınıf "kapalı" kararını tek bir kareye değil, son
    `window_seconds` içindeki geçerli kararların çoğunluğuna dayandırır
    (PERCLOS tarzı bir yaklaşım).

    Pencere neden KARE değil ZAMAN cinsinden: ilk sürümde pencere 7
    KARE genişliğindeydi. Bir karenin süresi FPS'e bağlı olduğu için bu,
    5 FPS'te 1.4 saniyeyi, 30 FPS'te 0.23 saniyeyi kapsıyordu — aynı
    kod, makinenin hızına göre 6 kat farklı bir zamansal davranış.
    Uçtan uca ölçüm bunu doğruladı: 5→30 FPS aralığında alarm gecikmesi
    0.79 saniye kayıyordu, smoother devre dışıyken aynı kayma yalnızca
    0.06 saniyeydi (bkz. README "FPS dayanıklılığı"). Yani sistem hızlı
    bir makinede sessizce daha az yumuşatma yapıyordu ve bunu hiçbir yer
    raporlamıyordu.

    Bir ölçümün "güvenilir" olup olmadığına karar vermez — bunu çağıran
    taraf (YoloDrowsinessDetector) `valid` parametresiyle bildirir. Bu
    sınıf yalnızca: (1) penceredeki geçerli kararların çoğunluğuna göre
    karar üretir, (2) art arda gelen geçersiz ölçümlerde kısa bir
    tolerans süresi boyunca önceki kararı korur, aşılırsa sıfırlar.

    Thread-safety: DrowsinessService ile aynı varsayım — tek arka plan
    thread'inden, her karede bir kez `update()` çağrılacak şekilde
    tasarlandı.
    """

    def __init__(
        self,
        *,
        window_seconds: float = 0.9,
        closed_ratio_threshold: float = 0.6,
        missing_tolerance_seconds: float = 0.25,
        minimum_valid_seconds: float = 0.35,
        clock: Clock = time.monotonic,
    ) -> None:
        """
        Args:
            window_seconds: Kararın dayandığı zaman penceresi; bu süreden
                eski oylar düşer. Varsayılan 0.9 sn, eski 7 karelik
                pencerenin ~8 FPS'teki (gerçekçi işleme hızı) karşılığı.
                Yani MEVCUT DAVRANIŞI KORUMAK için seçildi, benchmark
                skorunu optimize etmek için değil — 6 olaylık bir sette
                parametre optimizasyonu gürültüye fit etmek olurdu.

            closed_ratio_threshold: Penceredeki geçerli oyların bu oranı
                (veya üzeri) "kapalı" ise, yumuşatılmış karar "kapalı"
                olur. (0.0, 1.0] aralığında olmalı.

            missing_tolerance_seconds: Art arda kaç saniyelik geçersiz
                (valid=False) ölçümden sonra ÖNCEKİ kararın korunacağı.
                Bu süre aşılırsa geçmiş tamamen sıfırlanır ve "açık"
                (False) döner.

            minimum_valid_seconds: Sıfırlamadan sonra en az bu kadar
                süre geçerli ölçüm alınmadan "kapalı" kararı ASLA
                verilmez (sistem başlarken ya da resetten hemen sonra
                yanlışlıkla erken "kapalı" dememesi için güvenlik payı).
                window_seconds'tan büyük olamaz. 0.0 = koruma kapalı,
                ilk ölçüm de karara katılabilir (test double'ları için).

                Pencere İÇİNDEKİ yayılıma değil, ilk geçerli ölçümden
                bu yana geçen süreye bakar: düşük kare hızlarında
                pencerede tek oy kalabiliyor ve yayılım hep 0 çıkıyordu,
                yani koruma hiç açılmıyordu.

            clock: Süre ölçümü fonksiyonu — gerçek çalışmada
                time.monotonic, testlerde ve offline değerlendirmede
                enjekte edilen sahte/sanal saat.
        """
        if window_seconds <= 0:
            raise ValueError("window_seconds sıfırdan büyük olmalıdır.")
        if not 0.0 < closed_ratio_threshold <= 1.0:
            raise ValueError("closed_ratio_threshold 0 ile 1 arasında olmalıdır (0 hariç, 1 dahil).")
        if missing_tolerance_seconds < 0:
            raise ValueError("missing_tolerance_seconds negatif olamaz.")
        if minimum_valid_seconds < 0:
            raise ValueError("minimum_valid_seconds negatif olamaz.")
        if minimum_valid_seconds > window_seconds:
            raise ValueError("minimum_valid_seconds, window_seconds'tan büyük olamaz.")
        if not callable(clock):
            raise TypeError("clock çağrılabilir bir fonksiyon olmalıdır.")

        self._window_seconds = float(window_seconds)
        self._closed_ratio_threshold = closed_ratio_threshold
        self._missing_tolerance_seconds = float(missing_tolerance_seconds)
        self._minimum_valid_seconds = float(minimum_valid_seconds)
        self._clock = clock

        # (zaman, oy) çiftleri — süresi geçenler _evict_old ile düşer.
        self._votes: deque[tuple[float, bool]] = deque()
        # İlk geçerli ölçümün zamanı — minimum_valid_seconds bunu baz alır.
        self._first_vote_at: float | None = None
        self._missing_since: float | None = None
        self._last_smoothed = False
        self._last_confidence = 0.0

    @property
    def window_seconds(self) -> float:
        return self._window_seconds

    @property
    def last_confidence(self) -> float:
        """Pencereye eklenen son GEÇERLİ oyun confidence değeri. Hiç
        geçerli oy yoksa 0.0. Yalnızca gözlem/debug amaçlı — karar
        mantığını etkilemez."""
        return self._last_confidence

    @property
    def valid_vote_count(self) -> int:
        """Pencerede şu an biriken geçerli oy sayısı."""
        return len(self._votes)

    @property
    def valid_span_seconds(self) -> float:
        """İlk geçerli ölçümden son ölçüme kadar geçen süre.
        `minimum_valid_seconds` kontrolü buna bakar — pencere içindeki
        yayılıma değil, çünkü düşük kare hızlarında pencerede tek oy
        kalabilir ve yayılım hep 0 çıkardı."""
        if self._first_vote_at is None or not self._votes:
            return 0.0
        return self._votes[-1][0] - self._first_vote_at

    def reset(self) -> None:
        """Pencereyi, tolerans sayacını ve son kararı temizler."""
        self._votes.clear()
        self._first_vote_at = None
        self._missing_since = None
        self._last_smoothed = False
        self._last_confidence = 0.0

    def update(self, *, valid: bool, eyes_closed: bool, confidence: float) -> bool:
        """
        Yeni bir ölçümle yumuşatılmış "gözler kapalı mı" kararını günceller.

        Args:
            valid: Bu karenin ölçümünün güvenilir olup olmadığı (çağıran
                taraf belirler — örn. eyes_examined==0 ya da düşük
                confidence'lı tek göz `False` göndermeli).
            eyes_closed: Bu karenin HAM (yumuşatılmamış) kapalı/açık
                kararı. `valid=False` iken bu değer yok sayılır.
            confidence: Bu ölçümün güven değeri (yalnızca gözlem/debug
                amaçlı saklanır, karar mantığını etkilemez).

        Returns:
            Yumuşatılmış "gözler kapalı mı" kararı (bool).
        """
        now = self._clock()

        if not valid:
            if self._missing_since is None:
                self._missing_since = now

            if now - self._missing_since > self._missing_tolerance_seconds:
                self.reset()
                return False

            # Tolerans içinde: seri kesilmemiş gibi davran, önceki kararı koru.
            return self._last_smoothed

        self._missing_since = None
        if self._first_vote_at is None:
            self._first_vote_at = now
        self._votes.append((now, bool(eyes_closed)))
        self._last_confidence = float(confidence)
        self._evict_old(now)

        if self.valid_span_seconds < self._minimum_valid_seconds:
            # Henüz yeterince uzun bir geçerli ölçüm birikmedi — erken
            # "kapalı" kararı vermek yerine güvenli tarafta kal.
            self._last_smoothed = False
            return False

        closed_ratio = sum(1 for _, vote in self._votes if vote) / len(self._votes)
        self._last_smoothed = closed_ratio >= self._closed_ratio_threshold
        return self._last_smoothed

    def _evict_old(self, now: float) -> None:
        """
        Pencere süresinden eski oyları düşürür.

        En yeni oy HER ZAMAN korunur: kare hızı pencereden düşükse
        (ör. 1 FPS, 0.9 sn pencere) her yeni oy tek başına pencereyi
        aşardı ve boş pencereyle sıfıra bölme oluşurdu.
        """
        cutoff = now - self._window_seconds
        while len(self._votes) > 1 and self._votes[0][0] < cutoff:
            self._votes.popleft()
