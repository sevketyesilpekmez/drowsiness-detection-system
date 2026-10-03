import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AlarmHistory from "./AlarmHistory";

describe("AlarmHistory", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("boş geçmişte placeholder gösterir", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ history: [] }),
      })
    );

    render(<AlarmHistory alarmTriggerCount={0} />);

    expect(await screen.findByText(/henüz tetiklenmiş bir alarm yok/i)).toBeInTheDocument();
  });

  it("geçmişteki kayıtları saat/süre/güven ile listeler", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            history: [
              { id: 3, timestamp: "2026-07-24T12:43:18.000Z", closed_duration: 2.8, confidence: 0.96 },
              { id: 2, timestamp: "2026-07-24T12:39:04.000Z", closed_duration: 3.1, confidence: 0.93 },
            ],
          }),
      })
    );

    render(<AlarmHistory alarmTriggerCount={2} />);

    expect(await screen.findByText("2.8 sn")).toBeInTheDocument();
    expect(screen.getByText("96%")).toBeInTheDocument();
    expect(screen.getByText("3.1 sn")).toBeInTheDocument();

    const links = screen.getAllByRole("link", { name: /görüntüle/i });
    expect(links).toHaveLength(2);
    expect(links[0]).toHaveAttribute("href", expect.stringContaining("/api/detection/alarm-snapshot/3"));
  });

  it("alarmTriggerCount arttığında geçmişi yeniden çeker", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ history: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const { rerender } = render(<AlarmHistory alarmTriggerCount={1} />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    rerender(<AlarmHistory alarmTriggerCount={2} />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
  });

  it("istek başarısız olursa hata mesajı gösterir", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false }));

    render(<AlarmHistory alarmTriggerCount={0} />);

    expect(await screen.findByText(/alarm geçmişi alınamadı/i)).toBeInTheDocument();
  });
});
