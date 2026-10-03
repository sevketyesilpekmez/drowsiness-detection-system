"""
Kare indeksinden türetilen sanal saat.

Neden gerçek saat kullanılmıyor: `DetectionEngine` wall-clock ve thread
tabanlı çalışır, dolayısıyla aynı videoyu iki kez değerlendirmek iki
farklı sonuç verir ve sonuç makinenin hızına bağlı olur. Bir ölçüm
aracının bunu yapması kabul edilemez.

`DrowsinessService` ve `AlarmService` zaten `clock` parametresi alıyor
(testlerdeki `FakeClock` için tasarlanmıştı). Bu sınıf aynı seam'i
kullanır: zaman, videonun kendi kare hızından türetilir. Sonuç
deterministiktir, gerçek zamandan hızlı koşar ve CI'da tekrarlanabilir.
"""

from __future__ import annotations


class FrameClock:
    """
    `Callable[[], float]` sözleşmesine uyan sanal saat.

    Zaman, işlenen karenin videodaki gerçek zaman damgasıdır:

        t = frame_index / fps

    Kare ATLAMA durumunda bile doğru kalır — 30 FPS'lik bir videoda 4
    karede bir işlem yapılıyorsa ardışık iki ölçüm arasında 4/30 = 133 ms
    geçmiş sayılır, ki gerçekte de öyledir.
    """

    def __init__(self, *, fps: float, start_time_s: float = 0.0) -> None:
        if fps <= 0:
            raise ValueError("fps sıfırdan büyük olmalıdır.")
        if start_time_s < 0:
            raise ValueError("start_time_s negatif olamaz.")

        self._fps = float(fps)
        self._start_time_s = float(start_time_s)
        self._frame_index = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def __call__(self) -> float:
        """Şu anki sanal zamanı saniye cinsinden döndürür."""
        return self._start_time_s + self._frame_index / self._fps

    def reset(self, *, fps: float | None = None, start_time_s: float = 0.0) -> None:
        """
        Saati başa sarar; isteğe bağlı olarak kare hızını da değiştirir.

        Çok klipli bir değerlendirme koşusunda her klip kendi kare
        hızıyla ve sıfırdan başlamalı — aksi halde bir önceki klibin
        zamanı sonrakine taşınır ve `seek_to_frame`'in geri gitme yasağı
        yüzünden koşu patlar.
        """
        if fps is not None:
            if fps <= 0:
                raise ValueError("fps sıfırdan büyük olmalıdır.")
            self._fps = float(fps)

        if start_time_s < 0:
            raise ValueError("start_time_s negatif olamaz.")

        self._start_time_s = float(start_time_s)
        self._frame_index = 0

    def seek_to_frame(self, frame_index: int) -> None:
        """
        Saati verilen kareye ilerletir.

        Geriye sarma YASAK: `DrowsinessService` süreleri `now - since`
        ile hesaplıyor, saatin geri gitmesi negatif süreler ve sessizce
        yanlış kararlar üretirdi. Bir replay döngüsünde geriye gitmek
        zaten bir programlama hatasıdır.
        """
        if frame_index < 0:
            raise ValueError("frame_index negatif olamaz.")
        if frame_index < self._frame_index:
            raise ValueError(
                f"Sanal saat geri alınamaz (şu an {self._frame_index}, istenen {frame_index})."
            )

        self._frame_index = frame_index


def frame_step_for_target_fps(source_fps: float, target_fps: float | None) -> int:
    """
    Kaynak videonun kaç karesinde bir işlem yapılacağını hesaplar.

    Bu parametre kozmetik değil, sonucu DOĞRUDAN etkiler. Gerçekte sistem
    CPU'da 30 FPS'lik bir görüntüyü ~8 FPS işler, yani kare atlar.
    Offline değerlendirmede her kare işlenirse `EyeStateSmoother`'ın 7
    KARELİK penceresi 7/30 = 0.23 saniyeyi kapsar; gerçekte ise
    7/8 = 0.88 saniyeyi. Aynı kod, tamamen farklı zamansal davranış —
    bu yüzden değerlendirme, hedeflenen işleme hızını simüle etmelidir.

    Args:
        source_fps: Video dosyasının kare hızı.
        target_fps: Simüle edilecek işleme hızı. None ise her kare
            işlenir (adım = 1).

    Returns:
        En az 1 olan kare adımı.
    """
    if source_fps <= 0:
        raise ValueError("source_fps sıfırdan büyük olmalıdır.")

    if target_fps is None:
        return 1

    if target_fps <= 0:
        raise ValueError("target_fps sıfırdan büyük olmalıdır.")

    if target_fps >= source_fps:
        # Kaynaktan hızlı işlemek mümkün değil; her kare işlenir.
        return 1

    return max(1, round(source_fps / target_fps))
