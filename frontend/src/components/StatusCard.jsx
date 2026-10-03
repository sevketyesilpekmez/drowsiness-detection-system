import { STATE_HINTS, STATE_LABELS, isMeasured } from "../constants/detectionStates";


function formatPercentage(value) {
  return `${Math.round((typeof value === "number" ? value : 0) * 100)}%`;
}

function formatNumber(value, digits) {
  return typeof value === "number" ? value.toFixed(digits) : "-";
}

export default function StatusCard({ status, connectionError, isLoading }) {
  const state = typeof status?.state === "string" ? status.state : "STARTING";
  const stateClass = state.toLowerCase().replaceAll("_", "-");

  // "Yaklaşıyor" göstergesi: backend henüz DROWSY'ye geçmedi ama gözler
  // şu an kapalı ve süre birikiyor — erken bir turuncu uyarı için backend'e
  // dokunmadan eyes_closed + closed_duration'dan türetilir.
  const isApproaching =
    state === "AWAKE" && Boolean(status?.eyes_closed) && (status?.closed_duration ?? 0) > 0;

  const measured = isMeasured(status);

  const driverStateClass = `driver-state driver-state-${stateClass}${isApproaching ? " driver-state-approaching" : ""}`;

  return (
    <div className="status-card">
      <div className="panel-header">
        <h3>Sürücü Durumu</h3>
      </div>

      {connectionError && <p className="alert-error">{connectionError}</p>}
      {status?.error && <p className="alert-error">{status.error}</p>}

      <div className={driverStateClass}>
        <span className="state-dot" aria-hidden="true" />
        <div>
          <small>Sürücü durumu</small>
          <strong>{isLoading ? "Yükleniyor" : (STATE_LABELS[state] ?? state)}</strong>
          {!isLoading && STATE_HINTS[state] && <small>{STATE_HINTS[state]}</small>}
        </div>
      </div>

      <dl className="metrics-grid">
        <div className="metric-card">
          <dt>Gözler</dt>
          {/* Ölçüm alınamadığında "Açık" yazmak yanıltıcı olurdu — gözler
              açık değil, bilinmiyor. */}
          <dd>{measured ? (status?.eyes_closed ? "Kapalı" : "Açık") : "Bilinmiyor"}</dd>
        </div>

        <div className="metric-card">
          <dt>Kapalı Kalma</dt>
          <dd>{formatNumber(status?.closed_duration, 2)} sn</dd>
        </div>

        <div className="metric-card">
          <dt>Güven</dt>
          <dd>{formatPercentage(status?.confidence)}</dd>
        </div>

        <div className="metric-card">
          <dt>FPS</dt>
          <dd>{formatNumber(status?.fps, 1)}</dd>
        </div>

        <div className="metric-card">
          <dt>Alarm</dt>
          <dd>{status?.alarm_active ? "Aktif" : "Pasif"}</dd>
        </div>

        <div className="metric-card metric-card-secondary">
          <dt>EAR</dt>
          <dd>{formatNumber(status?.ear, 3)}</dd>
        </div>
      </dl>
    </div>
  );
}
