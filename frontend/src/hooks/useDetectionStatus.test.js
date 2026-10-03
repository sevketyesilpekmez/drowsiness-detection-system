import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useDetectionStatus } from "./useDetectionStatus";

describe("useDetectionStatus", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("başarısız bir istekten sonra polling'e devam eder", async () => {
    // NOT: intervalMs çok küçük olursa (örn. 5ms), reddedilen ilk çağrı ile
    // ondan hemen sonraki başarılı çağrı arasındaki pencere o kadar dar
    // olur ki connectionError'ın "network down" olduğu an test'in kendi
    // waitFor'unun ilk kontrolünden bile ÖNCE geçip gidebilir (flaky).
    // Bunun yerine mock'u başta HER ZAMAN reddedecek şekilde kurup, hatayı
    // gördükten SONRA başarılı yanıta geçiriyoruz — zamanlamaya bağımlı
    // olmayan, deterministik bir sıralama.
    const fetchMock = vi.fn().mockRejectedValue(new Error("network down"));
    vi.stubGlobal("fetch", fetchMock);

    const { result, unmount } = renderHook(() => useDetectionStatus(20));

    await waitFor(() => expect(result.current.connectionError).toBe("network down"));

    const callsWhileFailing = fetchMock.mock.calls.length;
    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThan(callsWhileFailing));

    fetchMock.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ state: "AWAKE" }),
    });

    await waitFor(() => expect(result.current.status.state).toBe("AWAKE"));
    expect(result.current.connectionError).toBeNull();

    unmount();
  });

  it("unmount sırasında aktif isteği iptal eder", async () => {
    const abortSpy = vi.fn();

    vi.stubGlobal(
      "fetch",
      vi.fn((_url, options) => {
        options.signal.addEventListener("abort", abortSpy);
        return new Promise(() => {}); // hiç çözülmeyen bir istek
      }),
    );

    const { unmount } = renderHook(() => useDetectionStatus(5));

    await waitFor(() => expect(fetch).toHaveBeenCalled());

    act(() => {
      unmount();
    });

    await waitFor(() => expect(abortSpy).toHaveBeenCalledTimes(1));
  });

  it("başarılı yanıttan sonra connectionError'ı temizler", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ state: "DROWSY", alarm_trigger_count: 2 }),
        }),
      ),
    );

    const { result, unmount } = renderHook(() => useDetectionStatus(5));

    await waitFor(() => expect(result.current.status.state).toBe("DROWSY"));
    expect(result.current.connectionError).toBeNull();
    // Backend'in göndermediği alanlar (bkz. INITIAL_STATUS) varsayılan
    // değerleriyle doldurulmalı, undefined kalmamalı.
    expect(result.current.status.alarm_active).toBe(false);

    unmount();
  });
});
