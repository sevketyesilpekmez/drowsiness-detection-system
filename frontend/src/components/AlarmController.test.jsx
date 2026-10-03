import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AlarmController from "./AlarmController";

describe("AlarmController", () => {
  let playMock;
  let pauseMock;

  beforeEach(() => {
    // Önceki testin spy'ını (ve çağrı geçmişini) sıfırlamadan yeni bir
    // vi.spyOn() eklemek, aynı prototip metodunun çağrı sayısının
    // testler arasında sızmasına yol açar — bu yüzden önce restore.
    vi.restoreAllMocks();
    playMock = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
    pauseMock = vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  });

  async function renderEnabled(initialStatus) {
    const utils = render(<AlarmController status={initialStatus} />);

    await userEvent.click(screen.getByRole("button", { name: /sesli alarmı etkinleştir/i }));
    await waitFor(() => expect(screen.getByText(/sesli alarm: açık/i)).toBeInTheDocument());

    // enableAlarm() kendi içinde bir play/pause yapıyor (tarayıcı izni
    // almak için) — asıl alarm testlerini kirletmesin diye temizliyoruz.
    playMock.mockClear();
    pauseMock.mockClear();

    return utils;
  }

  it("alarm_active false'tan true'ya geçince <audio loop> çalmaya başlar", async () => {
    const { rerender } = await renderEnabled({ alarm_active: false, alarm_trigger_count: 0 });

    rerender(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 1 }} />);

    await waitFor(() => expect(playMock).toHaveBeenCalledTimes(1));
  });

  it("alarm_active true kaldığı sürece play() TEKRAR çağrılmaz (baştan sarmaz)", async () => {
    const { rerender } = await renderEnabled({ alarm_active: false, alarm_trigger_count: 0 });

    rerender(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 1 }} />);
    await waitFor(() => expect(playMock).toHaveBeenCalledTimes(1));

    // Polling devam ediyor, alarm_active hâlâ true — <audio loop> zaten
    // kendi kendine tekrar ediyor, play()'in tekrar çağrılmaması gerekir.
    rerender(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 1 }} />);
    rerender(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 1 }} />);

    expect(playMock).toHaveBeenCalledTimes(1);
    expect(pauseMock).not.toHaveBeenCalled();
  });

  it("gözler açılıp alarm_active false olunca sesi hemen durdurur", async () => {
    const { rerender } = await renderEnabled({ alarm_active: false, alarm_trigger_count: 0 });

    rerender(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 1 }} />);
    await waitFor(() => expect(playMock).toHaveBeenCalledTimes(1));

    rerender(<AlarmController status={{ alarm_active: false, alarm_trigger_count: 1 }} />);

    await waitFor(() => expect(pauseMock).toHaveBeenCalledTimes(1));
  });

  it("alarm tekrar tetiklenirse (açılıp tekrar kapanırsa) sesi yeniden baştan çalar", async () => {
    const { rerender } = await renderEnabled({ alarm_active: false, alarm_trigger_count: 0 });

    rerender(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 1 }} />);
    await waitFor(() => expect(playMock).toHaveBeenCalledTimes(1));

    rerender(<AlarmController status={{ alarm_active: false, alarm_trigger_count: 1 }} />);
    await waitFor(() => expect(pauseMock).toHaveBeenCalledTimes(1));

    rerender(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 2 }} />);
    await waitFor(() => expect(playMock).toHaveBeenCalledTimes(2));
  });

  it("alarm zaten aktifken ses etkinleştirilirse yine de hemen çalmaya başlar", async () => {
    // Sayfa açıldığında alarm zaten aktifse (nadir ama mümkün), kullanıcı
    // "Sesli alarmı etkinleştir"e bastığı anda ses başlamalı — bir sonraki
    // açık->kapalı döngüsünü beklememeli.
    render(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 1 }} />);

    await userEvent.click(screen.getByRole("button", { name: /sesli alarmı etkinleştir/i }));

    await waitFor(() => expect(playMock.mock.calls.length).toBeGreaterThanOrEqual(2));
  });

  it("alarm etkinleştirilmeden alarm_active true olsa bile ses çalmaz", async () => {
    const { rerender } = render(<AlarmController status={{ alarm_active: false, alarm_trigger_count: 0 }} />);

    rerender(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 1 }} />);

    expect(playMock).not.toHaveBeenCalled();
  });

  it("alarm_active durumunu gösterir", () => {
    render(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 1 }} />);

    expect(screen.getByText(/dikkat: alarm aktif/i)).toBeInTheDocument();
  });

  it("audio elementi loop attribute'una sahiptir", () => {
    const { container } = render(<AlarmController status={{ alarm_active: false, alarm_trigger_count: 0 }} />);

    expect(container.querySelector("audio")).toHaveAttribute("loop");
  });

  describe("Sustur (mute)", () => {
    it("alarm çalarken Sustur butonu görünür", async () => {
      const { rerender } = await renderEnabled({ alarm_active: false, alarm_trigger_count: 0 });

      rerender(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 1 }} />);

      expect(await screen.findByRole("button", { name: /sustur/i })).toBeInTheDocument();
    });

    it("Sustur'a basınca sesi durdurur ve 'Susturuldu' gösterir", async () => {
      const { rerender } = await renderEnabled({ alarm_active: false, alarm_trigger_count: 0 });

      rerender(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 1 }} />);
      await waitFor(() => expect(playMock).toHaveBeenCalledTimes(1));

      await userEvent.click(screen.getByRole("button", { name: /sustur/i }));

      await waitFor(() => expect(pauseMock).toHaveBeenCalledTimes(1));
      expect(screen.getByText(/susturuldu/i)).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /sustur/i })).not.toBeInTheDocument();
    });

    it("susturma alarm SİSTEMİNİ kapatmaz — bir sonraki alarm yine sesli başlar", async () => {
      const { rerender } = await renderEnabled({ alarm_active: false, alarm_trigger_count: 0 });

      // 1. alarm: başlat, sustur.
      rerender(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 1 }} />);
      await waitFor(() => expect(playMock).toHaveBeenCalledTimes(1));
      await userEvent.click(screen.getByRole("button", { name: /sustur/i }));
      await waitFor(() => expect(pauseMock).toHaveBeenCalledTimes(1));

      // Gözler açıldı, alarm bitti.
      rerender(<AlarmController status={{ alarm_active: false, alarm_trigger_count: 1 }} />);

      // 2. alarm: yeni tetiklenme — susturma miras kalmamalı, ses tekrar çalmalı.
      rerender(<AlarmController status={{ alarm_active: true, alarm_trigger_count: 2 }} />);

      await waitFor(() => expect(playMock).toHaveBeenCalledTimes(2));
      expect(await screen.findByRole("button", { name: /sustur/i })).toBeInTheDocument();
    });
  });
});
