import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import AlarmSnapshot from "./AlarmSnapshot";

describe("AlarmSnapshot", () => {
  it("henüz alarm tetiklenmediyse placeholder gösterir", () => {
    render(<AlarmSnapshot status={{ alarm_snapshot_available: false, alarm_trigger_count: 0 }} />);

    expect(screen.getByText(/henüz tetiklenmiş bir alarm yok/i)).toBeInTheDocument();
    expect(screen.queryByAltText(/alarmın tetiklendiği/i)).not.toBeInTheDocument();
  });

  it("alarm_snapshot_available true ise görüntüyü gösterir", () => {
    render(<AlarmSnapshot status={{ alarm_snapshot_available: true, alarm_trigger_count: 2 }} />);

    const img = screen.getByAltText(/alarmın tetiklendiği/i);
    expect(img).toBeInTheDocument();
    expect(img.src).toContain("/api/detection/alarm-snapshot");
    expect(img.src).toContain("count=2");
  });

  it("yeni bir alarm tetiklendiğinde img src'i değişir (cache-bust)", () => {
    const { rerender } = render(
      <AlarmSnapshot status={{ alarm_snapshot_available: true, alarm_trigger_count: 1 }} />
    );
    const firstSrc = screen.getByAltText(/alarmın tetiklendiği/i).src;

    rerender(<AlarmSnapshot status={{ alarm_snapshot_available: true, alarm_trigger_count: 2 }} />);
    const secondSrc = screen.getByAltText(/alarmın tetiklendiği/i).src;

    expect(secondSrc).not.toBe(firstSrc);
  });

  it("görüntü yüklenemezse hata mesajı gösterir", () => {
    render(<AlarmSnapshot status={{ alarm_snapshot_available: true, alarm_trigger_count: 1 }} />);

    fireEvent.error(screen.getByAltText(/alarmın tetiklendiği/i));

    expect(screen.getByText(/alarm görüntüsü alınamadı/i)).toBeInTheDocument();
    expect(screen.queryByAltText(/alarmın tetiklendiği/i)).not.toBeInTheDocument();
  });

  it("bir görüntü yüklenemese bile SONRAKİ alarmın görüntüsünü gösterir", () => {
    // Regresyon testi: imageError yeni alarmda sıfırlanmadığında, hata
    // durumunda <img> DOM'dan kaldırıldığı için key={triggerCount} artık
    // remount tetikleyemiyordu ve kart kalıcı olarak "görüntü alınamadı"
    // durumunda kalıyordu — sonraki alarmların görüntüleri hiç denenmiyordu.
    const { rerender } = render(
      <AlarmSnapshot status={{ alarm_snapshot_available: true, alarm_trigger_count: 1 }} />
    );

    fireEvent.error(screen.getByAltText(/alarmın tetiklendiği/i));
    expect(screen.getByText(/alarm görüntüsü alınamadı/i)).toBeInTheDocument();

    rerender(<AlarmSnapshot status={{ alarm_snapshot_available: true, alarm_trigger_count: 2 }} />);

    const img = screen.getByAltText(/alarmın tetiklendiği/i);
    expect(img).toBeInTheDocument();
    expect(img.src).toContain("count=2");
    expect(screen.queryByText(/alarm görüntüsü alınamadı/i)).not.toBeInTheDocument();
  });

  it("aynı alarmda hata durumu korunur (her render'da yeniden denenmez)", () => {
    const { rerender } = render(
      <AlarmSnapshot status={{ alarm_snapshot_available: true, alarm_trigger_count: 4 }} />
    );

    fireEvent.error(screen.getByAltText(/alarmın tetiklendiği/i));

    rerender(
      <AlarmSnapshot
        status={{
          alarm_snapshot_available: true,
          alarm_trigger_count: 4,
          alarm_snapshot_confidence: 0.5,
        }}
      />
    );

    expect(screen.getByText(/alarm görüntüsü alınamadı/i)).toBeInTheDocument();
  });

  it("status verilmediğinde çökmeden placeholder gösterir", () => {
    render(<AlarmSnapshot status={undefined} />);

    expect(screen.getByText(/henüz tetiklenmiş bir alarm yok/i)).toBeInTheDocument();
  });

  it("kapalı kalma süresini ve güveni görüntüyle birlikte gösterir", () => {
    render(
      <AlarmSnapshot
        status={{
          alarm_snapshot_available: true,
          alarm_trigger_count: 1,
          alarm_snapshot_closed_duration: 2.567,
          alarm_snapshot_confidence: 0.914,
        }}
      />
    );

    expect(screen.getByText("2.6 sn")).toBeInTheDocument();
    expect(screen.getByText("91%")).toBeInTheDocument();
  });

  it("bağlam bilgisi henüz gelmediyse '-' gösterir (çökmez)", () => {
    render(
      <AlarmSnapshot
        status={{
          alarm_snapshot_available: true,
          alarm_trigger_count: 1,
          alarm_snapshot_closed_duration: null,
          alarm_snapshot_confidence: null,
          alarm_snapshot_timestamp: null,
          alarm_snapshot_state: null,
        }}
      />
    );

    // Saat, Süre, Güven, Durum — dördü de eksik veriyle "-" göstermeli.
    const dashes = screen.getAllByText("-");
    expect(dashes).toHaveLength(4);
  });

  it("saati ve durumu okunur biçimde gösterir", () => {
    render(
      <AlarmSnapshot
        status={{
          alarm_snapshot_available: true,
          alarm_trigger_count: 1,
          alarm_snapshot_timestamp: "2026-07-24T12:43:18.000Z",
          alarm_snapshot_state: "DROWSY",
          alarm_snapshot_closed_duration: 2.5,
          alarm_snapshot_confidence: 0.9,
        }}
      />
    );

    expect(screen.getByText("Uykulu")).toBeInTheDocument();
    // Saat tarayıcı yereline göre biçimlendiği için tam string'i değil,
    // formatTime'ın "-" DEĞİL gerçek bir saat ürettiğini doğruluyoruz.
    expect(screen.queryByText("-")).not.toBeInTheDocument();
  });

  it("'Görüntüyü büyüt' linki aynı görüntüyü yeni sekmede açar", () => {
    render(<AlarmSnapshot status={{ alarm_snapshot_available: true, alarm_trigger_count: 3 }} />);

    const link = screen.getByRole("link", { name: /görüntüyü büyüt/i });
    expect(link).toHaveAttribute("target", "_blank");
    expect(link.getAttribute("href")).toContain("/api/detection/alarm-snapshot");
    expect(link.getAttribute("href")).toContain("count=3");
  });
});
