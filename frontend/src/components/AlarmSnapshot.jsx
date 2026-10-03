import { useEffect, useState } from "react";
import { getAlarmSnapshotUrl } from "../services/api";
import { STATE_LABELS } from "../constants/detectionStates";

function formatSeconds(value) {
  return typeof value === "number" ? `${value.toFixed(1)} sn` : "-";
}

function formatPercentage(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "-";
}

function formatTime(isoTimestamp) {
  if (typeof isoTimestamp !== "string") {
    return "-";
  }

  const date = new Date(isoTimestamp);
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleTimeString("tr-TR");
}

/**
 * Son tetiklenen alarmın kompakt "olay kaydı" kartı: solda küçük bir
 * snapshot, sağda saat/süre/güven/durum bağlamı — sadece alarmın çaldığını
 * değil, alarma hangi görüntüye bakılarak karar verildiğini de gösterir
 * (bkz. backend/app/core/shared_state.py get_alarm_snapshot).
 *
 * status.alarm_trigger_count img src'inin bir parçası: URL değişmezse
 * tarayıcı önceki (cache'lenmiş) görüntüyü gösterebilir, bu yüzden her
 * yeni alarmda URL'i değiştirip yeniden indirmeye zorluyoruz (bkz.
 * CameraFeed.jsx'teki retryToken ile aynı desen).
 *
 * imageError'ın her yeni alarmda sıfırlanması ŞART: hata durumunda
 * <img> DOM'dan tamamen kaldırıldığı için `key={triggerCount}` artık
 * remount tetikleyemez, dolayısıyla tek bir başarısız indirme kartı
 * kalıcı olarak "görüntü alınamadı" durumunda bırakırdı — sonraki
 * alarmların görüntüleri de hiç denenmezdi.
 */
export default function AlarmSnapshot({ status }) {
  const [imageError, setImageError] = useState(false);
  const available = Boolean(status?.alarm_snapshot_available);
  const triggerCount = status?.alarm_trigger_count ?? 0;
  const snapshotUrl = getAlarmSnapshotUrl(triggerCount);

  useEffect(() => {
    setImageError(false);
  }, [triggerCount]);
  const stateLabel = STATE_LABELS[status?.alarm_snapshot_state] ?? status?.alarm_snapshot_state ?? "-";

  return (
    <section className="panel alarm-snapshot-panel">
      <div className="panel-header">
        <h3>Alarm Kaydı</h3>
      </div>

      {available && !imageError ? (
        <div className="alarm-record">
          <img
            key={triggerCount}
            className="alarm-record-image"
            src={snapshotUrl}
            alt="Alarmın tetiklendiği andaki kamera görüntüsü"
            onError={() => setImageError(true)}
          />

          <dl className="alarm-record-details">
            <div>
              <dt>Saat</dt>
              <dd>{formatTime(status?.alarm_snapshot_timestamp)}</dd>
            </div>
            <div>
              <dt>Kapalı kalma süresi</dt>
              <dd>{formatSeconds(status?.alarm_snapshot_closed_duration)}</dd>
            </div>
            <div>
              <dt>Güven</dt>
              <dd>{formatPercentage(status?.alarm_snapshot_confidence)}</dd>
            </div>
            <div>
              <dt>Durum</dt>
              <dd>{stateLabel}</dd>
            </div>

            <a
              className="alarm-record-enlarge"
              href={snapshotUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              Görüntüyü büyüt
            </a>
          </dl>
        </div>
      ) : (
        <div className="alarm-snapshot-placeholder">
          <p>{available ? "Alarm görüntüsü alınamadı." : "Henüz tetiklenmiş bir alarm yok."}</p>
        </div>
      )}
    </section>
  );
}
