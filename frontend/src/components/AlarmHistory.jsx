import { useEffect, useState } from "react";
import { getAlarmHistory, getAlarmHistorySnapshotUrl } from "../services/api";

function formatTime(isoTimestamp) {
  if (typeof isoTimestamp !== "string") {
    return "-";
  }

  const date = new Date(isoTimestamp);
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleTimeString("tr-TR");
}

function formatSeconds(value) {
  return typeof value === "number" ? `${value.toFixed(1)} sn` : "-";
}

function formatPercentage(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "-";
}

/**
 * Son ALARM_HISTORY_MAX_LEN (5) alarmın kısa listesi.
 *
 * useDetectionStatus'un 500ms'lik pollingine bağlı değil — alarm geçmişi
 * o kadar sık değişmiyor. Bunun yerine alarmTriggerCount (status'tan)
 * her arttığında yeniden çekiyor; ayrıca mount'ta bir kere çeker.
 */
export default function AlarmHistory({ alarmTriggerCount }) {
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    getAlarmHistory(controller.signal)
      .then((data) => {
        if (!cancelled) {
          setHistory(Array.isArray(data?.history) ? data.history : []);
          setError(null);
        }
      })
      .catch((fetchError) => {
        if (!cancelled && fetchError.name !== "AbortError") {
          setError(fetchError.message ?? "Alarm geçmişi alınamadı.");
        }
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [alarmTriggerCount]);

  return (
    <section className="panel alarm-history-panel">
      <div className="panel-header">
        <h3>Son Alarm Kayıtları</h3>
      </div>

      {error && <p className="alert-error">{error}</p>}

      {history.length === 0 ? (
        <div className="alarm-snapshot-placeholder">
          <p>Henüz tetiklenmiş bir alarm yok.</p>
        </div>
      ) : (
        <ul className="alarm-history-list">
          {history.map((entry) => (
            <li key={entry.id} className="alarm-history-row">
              <span className="alarm-history-time">{formatTime(entry.timestamp)}</span>
              <span className="alarm-history-duration">{formatSeconds(entry.closed_duration)}</span>
              <span className="alarm-history-confidence">{formatPercentage(entry.confidence)}</span>
              <a
                className="alarm-history-view"
                href={getAlarmHistorySnapshotUrl(entry.id)}
                target="_blank"
                rel="noopener noreferrer"
              >
                Görüntüle
              </a>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
