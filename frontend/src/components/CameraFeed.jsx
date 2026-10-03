import { useState } from "react";
import { getStreamUrl } from "../services/api";

// StatusCard.jsx'teki STATE_LABELS ile aynı sözlük, burada da kısa bir
// pill etiketi için gerekiyor (bkz. backend/app/core/states.py).
const STATE_LABELS = {
  STARTING: "Başlatılıyor",
  AWAKE: "Uyanık",
  DROWSY: "Uyku hali",
  NO_FACE: "Yüz yok",
  CAMERA_ERROR: "Kamera hatası",
  MODEL_ERROR: "Model hatası",
  SYSTEM_ERROR: "Sistem hatası",
};

const DANGER_STATES = new Set(["DROWSY", "CAMERA_ERROR", "MODEL_ERROR", "SYSTEM_ERROR"]);

function formatFps(value) {
  return typeof value === "number" ? value.toFixed(1) : "-";
}

/**
 * Kameranın üzerine bindirilen "State/FPS" bilgisi normal bir HTML/CSS
 * "pill" olarak görüntünün üzerine bindiriliyor — görüntünün piksellerini
 * değiştirmiyor, tamamen esnek.
 */
export default function CameraFeed({ status }) {
  const [streamError, setStreamError] = useState(false);
  // <img src> aynı kalırsa "Tekrar dene" tıklamasında tarayıcı yeni bir
  // istek başlatmayabilir; her denemede URL'e önemsiz bir query param
  // ekleyip tarayıcıyı gerçekten yeni bir bağlantı açmaya zorluyoruz.
  const [retryToken, setRetryToken] = useState(0);

  function retry() {
    setStreamError(false);
    setRetryToken((token) => token + 1);
  }

  const state = typeof status?.state === "string" ? status.state : null;
  const statePillClass = state && DANGER_STATES.has(state) ? "status-pill danger" : "status-pill safe";

  return (
    <section className="panel camera-panel">
      <div className="panel-header">
        <h3>Canlı Kamera</h3>
        <span
          className={streamError ? "dot dot-error" : "dot dot-ok"}
          aria-label={streamError ? "Kamera bağlantısı yok" : "Kamera bağlantısı aktif"}
        />
      </div>

      <div className="camera-frame">
        {!streamError ? (
          <>
            <img
              key={retryToken}
              src={`${getStreamUrl()}${retryToken ? `?retry=${retryToken}` : ""}`}
              alt="Canlı uyku tespit kamerası"
              onError={() => setStreamError(true)}
            />

            {state && (
              <div className="camera-overlay">
                <span className={statePillClass}>{STATE_LABELS[state] ?? state}</span>
                <span className="fps-pill">{formatFps(status?.fps)} FPS</span>
              </div>
            )}
          </>
        ) : (
          <div className="camera-placeholder">
            <p>Kamera yayını alınamadı.</p>
            <p>Backend'in ve ENABLE_DETECTION_ENGINE'in çalıştığını kontrol et.</p>
            <button type="button" onClick={retry}>
              Tekrar dene
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
