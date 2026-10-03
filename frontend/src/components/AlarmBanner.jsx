/**
 * Alarm aktifken kamera panelinin üstünde beliren büyük uyarı bandı.
 * Sesli alarm ve kırmızı çerçeveyle aynı anda görünür/kaybolur (hepsi
 * status.alarm_active'e bağlı).
 *
 * closed_duration canlı değeri kullanılır (dondurulmuş
 * alarm_snapshot_closed_duration değil) — banner alarm sürerken
 * gösterilir, süre artmaya devam etmeli.
 */
export default function AlarmBanner({ status }) {
  if (!status?.alarm_active) {
    return null;
  }

  const closedDuration = typeof status.closed_duration === "number" ? status.closed_duration : 0;
  const time = new Date().toLocaleTimeString("tr-TR");

  return (
    <div className="alarm-banner" role="alert">
      <span className="alarm-icon" aria-hidden="true">
        !
      </span>

      <div>
        <strong>Uyku hali algılandı</strong>
        <span>Gözler {closedDuration.toFixed(1)} saniyedir kapalı</span>
      </div>

      <span className="alarm-time">{time}</span>
    </div>
  );
}
