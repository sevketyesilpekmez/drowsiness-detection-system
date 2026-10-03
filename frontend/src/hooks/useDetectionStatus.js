import { useEffect, useState } from "react";
import { getDetectionStatus } from "../services/api";

// Backend her zaman bu alanları döndürür, ama ilk render'da veya bir hata
// anında status henüz gelmediyse bileşenlerin undefined kontrolüyle
// uğraşmaması için güvenli varsayılanlar.
const INITIAL_STATUS = {
  state: "STARTING",
  is_drowsy: false,
  eyes_closed: false,
  confidence: 0,
  closed_duration: 0,
  fps: 0,
  ear: null,
  left_ear: null,
  right_ear: null,
  measurement_valid: true,
  alarm_triggered: false,
  alarm_active: false,
  alarm_trigger_count: 0,
  alarm_snapshot_available: false,
  alarm_snapshot_closed_duration: null,
  alarm_snapshot_confidence: null,
  alarm_snapshot_state: null,
  alarm_snapshot_timestamp: null,
  error: null,
};

/**
 * Backend'deki /api/detection/status'u belirli aralıklarla çeker.
 *
 * setInterval YERİNE zincirleme setTimeout kullanılır: bir istek bitmeden
 * (yavaş ağ/backend durumunda) bir sonraki asla başlamaz, üst üste binen
 * istekler oluşmaz.
 */
export function useDetectionStatus(pollIntervalMs = 500) {
  const [status, setStatus] = useState(INITIAL_STATUS);
  const [connectionError, setConnectionError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    let timeoutId = null;
    let controller = null;

    async function poll() {
      controller = new AbortController();

      try {
        const data = await getDetectionStatus(controller.signal);

        if (!cancelled) {
          setStatus({ ...INITIAL_STATUS, ...data });
          setConnectionError(null);
        }
      } catch (requestError) {
        if (!cancelled && requestError.name !== "AbortError") {
          setConnectionError(requestError.message ?? "Backend bağlantısı kurulamadı.");
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
          timeoutId = window.setTimeout(poll, pollIntervalMs);
        }
      }
    }

    poll();

    return () => {
      cancelled = true;

      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }

      controller?.abort();
    };
  }, [pollIntervalMs]);

  return { status, connectionError, isLoading };
}
