class DetectionState:
    """
    Sistemin alabileceği tüm durumlar tek yerde — raw string yerine bu
    sabitler kullanılır, yazım hatası sessizce kırılmasın diye.

    eyes_closed/closed_duration state DEĞİL, ayrı alanlar: state "UI ne
    tepki versin" sorusuna, bunlar "şu an ne ölçülüyor" sorusuna cevap
    verir. AWAKE, "gözler açık" demek değil — "henüz alarm eşiği
    aşılmadı" demek; göz kırpma gibi kısa kapanmalarda state AWAKE kalır.
    """

    STARTING = "STARTING"
    AWAKE = "AWAKE"
    DROWSY = "DROWSY"
    NO_FACE = "NO_FACE"
    # Kadrajda bir kişi var ama göz durumu güvenilir şekilde ÖLÇÜLEMİYOR
    # (sürücü yüzünü çevirmiş, göz bölgesi bulunamıyor, ya da bulunan
    # crop'lar sınıflandırılamıyor). NO_FACE'ten ayrı tutuluyor çünkü iki
    # durumun sebebi ve çözümü farklı: NO_FACE "kadrajda kimse yok",
    # UNVERIFIABLE "kişi var, ben göremiyorum". İkisini aynı etiket altında
    # toplamak, bir uyku tespit sisteminde "ölçemiyorum"u "sorun yok" gibi
    # gösterme riski taşır.
    UNVERIFIABLE = "UNVERIFIABLE"
    CAMERA_ERROR = "CAMERA_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    # set_error()'ın varsayılanı: kameraya/modele özgü olmayan genel hatalar.
    SYSTEM_ERROR = "SYSTEM_ERROR"
