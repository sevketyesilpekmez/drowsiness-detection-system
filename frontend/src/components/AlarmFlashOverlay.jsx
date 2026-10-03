/**
 * Alarm aktifken tüm ekranı saran, yanıp sönen kırmızı bir çerçeve
 * gösterir. Ses (AlarmController) ile aynı sinyali (status.alarm_active)
 * kullanır, böylece ikisi tam senkron başlar/durur.
 *
 * position: fixed kasıtlı — tüm viewport'u kaplaması gereken bir uyarı
 * katmanı. pointer-events: none, altındaki hiçbir etkileşimi engellemez.
 */
export default function AlarmFlashOverlay({ active }) {
  if (!active) {
    return null;
  }

  return <div className="alarm-flash-overlay" data-testid="alarm-flash-overlay" aria-hidden="true" />;
}
