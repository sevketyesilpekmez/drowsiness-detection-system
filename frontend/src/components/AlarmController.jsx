import { useEffect, useRef, useState } from "react";

/**
 * Alarm sesini tarayıcıda çalar. Backend ses çalmaz (bkz.
 * backend/app/services/alarm_service.py) — yalnızca alarm_active/
 * alarm_trigger_count sinyallerini üretir.
 *
 * status.alarm_active kullanılıyor (alarm_trigger_count değil): istenen
 * davranış gözler açılana kadar alarmın kesintisiz çalmaya devam etmesi.
 * alarm_trigger_count cooldown'lu bir sayaç (varsayılan 5 saniyede bir
 * artar) — ona dayanan bir tasarım sesi 5 saniyede bir kısa kısa çalardı.
 * alarm_active ise is_drowsy=false olduğu anda (cooldown beklemeden)
 * false'a düşen bir seviye sinyali.
 *
 * <audio loop> ile play(), yalnızca "çalıyor olması gereken durum"
 * (enabled && alarm_active && !muted) false->true kenarında çağrılıyor —
 * her poll'da tekrar play() çağırmak sesi sürekli baştan sarardı.
 *
 * "Sustur": mevcut alarm episodunu susturur, sistemi kapatmaz — bir
 * sonraki alarmda otomatik sıfırlanır, her yeni alarm sesli başlar.
 */
export default function AlarmController({ status }) {
  const audioRef = useRef(null);
  const wasPlayingRef = useRef(false);

  const [enabled, setEnabled] = useState(false);
  const [audioError, setAudioError] = useState(null);
  const [muted, setMuted] = useState(false);

  const isActive = Boolean(status.alarm_active);

  async function enableAlarm() {
    const audio = audioRef.current;
    if (!audio) return;

    try {
      audio.volume = 1;
      audio.currentTime = 0;

      // Tarayıcılar kullanıcı etkileşimi olmadan otomatik ses çalmayı
      // engeller. Kısa bir play/pause, bu düğmeye tıklama etkileşimi
      // içinde tarayıcının ses iznini kalıcı olarak açmasını sağlar.
      await audio.play();
      audio.pause();
      audio.currentTime = 0;

      setEnabled(true);
      setAudioError(null);
    } catch {
      setEnabled(false);
      setAudioError("Tarayıcı alarm sesine izin vermedi.");
    }
  }

  // Alarm bittiğinde (gözler açıldı) muted'i sıfırla — bir sonraki alarm
  // her zaman sesli başlamalı, önceki susturma miras kalmamalı.
  useEffect(() => {
    if (!isActive && muted) {
      setMuted(false);
    }
  }, [isActive, muted]);

  useEffect(() => {
    const shouldPlay = enabled && isActive && !muted;
    const wasPlaying = wasPlayingRef.current;
    wasPlayingRef.current = shouldPlay;

    if (!audioRef.current) {
      return;
    }

    const audio = audioRef.current;

    if (shouldPlay && !wasPlaying) {
      // Alarm YENİ başladı (gözler yeni kapandı, ses az önce etkinleştirildi
      // ya da az önce susturmadan çıkıldı) — baştan çal, <audio loop> alarm
      // bitene kadar kendiliğinden tekrar tekrar çalmaya devam eder.
      audio.currentTime = 0;
      audio.play().catch(() => setAudioError("Alarm sesi oynatılamadı."));
    } else if (!shouldPlay && wasPlaying) {
      // Alarm bitti YA DA kullanıcı sustur dedi — hemen durdur.
      audio.pause();
      audio.currentTime = 0;
    }
  }, [enabled, isActive, muted]);

  const isRinging = enabled && isActive;

  return (
    <div className="alarm-panel">
      <audio ref={audioRef} src="/alarm.wav" preload="auto" loop />

      <div>
        <h3>Alarm Sistemi</h3>
        <p className="alarm-description">
          Tarayıcı güvenlik kısıtlaması nedeniyle sesli alarmı bir kez elle etkinleştirmen gerekiyor.
        </p>
      </div>

      <div className="alarm-actions">
        <span className={isActive ? "alarm-indicator alarm-indicator-active" : "alarm-indicator"}>
          {isActive ? "Dikkat: alarm aktif" : "Alarm beklemede"}
        </span>

        {!enabled && (
          <button type="button" onClick={enableAlarm}>
            Sesli alarmı etkinleştir
          </button>
        )}

        {enabled && !isRinging && (
          <span className="alarm-toggle-state alarm-toggle-on">Sesli alarm: Açık</span>
        )}

        {isRinging && !muted && (
          <button type="button" className="alarm-mute-button" onClick={() => setMuted(true)}>
            Alarm çalıyor — Sustur
          </button>
        )}

        {isRinging && muted && <span className="alarm-toggle-state alarm-toggle-muted">Susturuldu</span>}
      </div>

      {audioError && <p className="alert-error">{audioError}</p>}
    </div>
  );
}
