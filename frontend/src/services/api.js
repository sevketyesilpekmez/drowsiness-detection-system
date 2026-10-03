// Prod'da (Docker/nginx) ve dev'de (vite proxy) relative "/api" her zaman
// çalışır — nginx.conf ve vite.config.js bunu backend'e proxy'ler.
// VITE_API_BASE_URL yalnızca build-time'da override etmek istersen kullanılır.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error("Health check başarısız.");
  }

  return response.json();
}

export async function getDetectionStatus(signal) {
  const response = await fetch(`${API_BASE_URL}/detection/status`, { signal });

  if (!response.ok) {
    throw new Error("Durum bilgisi alınamadı.");
  }

  return response.json();
}

// Relative "/api/stream" — nginx.conf (prod) ve vite dev proxy backend'e
// yönlendirir. img src'i olarak doğrudan kullanılır (tarayıcı MJPEG
// stream'i kendi çözer).
export function getStreamUrl() {
  return `${API_BASE_URL}/stream`;
}

// Alarmın tetiklendiği andaki kareyi döndüren endpoint (bkz. backend
// SharedState.get_alarm_snapshot). triggerCount URL'in bir parçası: aynı
// URL tarayıcının eski görüntüyü cache'ten göstermesine yol açabilir, bu
// yüzden her yeni alarmda URL değiştirilip yeni indirme zorlanır.
export function getAlarmSnapshotUrl(triggerCount) {
  return `${API_BASE_URL}/detection/alarm-snapshot?count=${triggerCount ?? 0}`;
}

// Son ALARM_HISTORY_MAX_LEN (5) alarmın metadata listesini döner —
// görüntü içermez, her kaydın görüntüsü ayrıca id bazlı çekilir.
export async function getAlarmHistory(signal) {
  const response = await fetch(`${API_BASE_URL}/detection/alarm-history`, { signal });

  if (!response.ok) {
    throw new Error("Alarm geçmişi alınamadı.");
  }

  return response.json();
}

// Geçmişteki belirli bir alarm kaydının görüntüsü — cache-bust'a gerek
// yok, her id'nin görüntüsü sabit/değişmez (geçmiş kayıtlar üzerine
// yazılmaz, yalnızca en eskisi deque'den düşer).
export function getAlarmHistorySnapshotUrl(alarmId) {
  return `${API_BASE_URL}/detection/alarm-snapshot/${alarmId}`;
}

export { API_BASE_URL };
