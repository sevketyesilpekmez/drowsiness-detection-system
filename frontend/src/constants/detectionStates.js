// Backend'in DetectionState sabitlerinin (backend/app/core/states.py)
// Türkçe karşılıkları — TEK kaynak.
//
// Bu dosya, aynı sözlüğün StatusCard.jsx / AlarmSnapshot.jsx /
// StatusCard.test.jsx içinde üç ayrı kopya olarak durmasından sonra
// çıkarıldı: UNVERIFIABLE eklenirken kopyalar birbirinden ayrıştı ve
// test, ürün kodunu değil kendi kopyasını doğruladığı için sessizce
// kırıldı.
export const STATE_LABELS = {
  STARTING: "Başlatılıyor",
  AWAKE: "Uyanık",
  DROWSY: "Uykulu",
  NO_FACE: "Kimse Yok",
  UNVERIFIABLE: "Göz Durumu Okunamıyor",
  CAMERA_ERROR: "Kamera Hatası",
  MODEL_ERROR: "Model Hatası",
  SYSTEM_ERROR: "Sistem Hatası",
};

// Kullanıcının bir şey yapması gereken durumlar için kısa yönlendirme.
// Hata durumlarına (CAMERA_ERROR vb.) ipucu yazılmıyor; onların mesajı
// zaten backend'den `error` alanıyla geliyor.
export const STATE_HINTS = {
  NO_FACE: "Kamerada kimse görünmüyor.",
  UNVERIFIABLE: "Kişi görünüyor ama gözler okunamıyor — yüzünüzü kameraya dönün.",
};

/**
 * Göz durumunun bu karede gerçekten ölçülüp ölçülemediği.
 *
 * `measurement_valid` alanını yalnızca yolo_pytorch backend'i gönderir;
 * alan yoksa (mediapipe/EAR backend'i) ölçüm alındığı varsayılır.
 */
export function isMeasured(status) {
  return status?.measurement_valid !== false && status?.state !== "UNVERIFIABLE";
}
