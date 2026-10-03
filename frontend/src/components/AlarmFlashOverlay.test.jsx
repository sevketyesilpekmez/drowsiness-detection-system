import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import AlarmFlashOverlay from "./AlarmFlashOverlay";

describe("AlarmFlashOverlay", () => {
  it("active=false iken hiçbir şey render etmez", () => {
    render(<AlarmFlashOverlay active={false} />);

    expect(screen.queryByTestId("alarm-flash-overlay")).not.toBeInTheDocument();
  });

  it("active=true iken yanıp sönen kırmızı çerçeveyi render eder", () => {
    render(<AlarmFlashOverlay active={true} />);

    const overlay = screen.getByTestId("alarm-flash-overlay");
    expect(overlay).toBeInTheDocument();
    expect(overlay).toHaveClass("alarm-flash-overlay");
  });

  it("dekoratif katman ekran okuyuculardan gizlenir", () => {
    render(<AlarmFlashOverlay active={true} />);

    expect(screen.getByTestId("alarm-flash-overlay")).toHaveAttribute("aria-hidden", "true");
  });
});
