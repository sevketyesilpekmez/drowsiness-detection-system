import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import AlarmBanner from "./AlarmBanner";

describe("AlarmBanner", () => {
  it("alarm_active false iken hiçbir şey render etmez", () => {
    render(<AlarmBanner status={{ alarm_active: false }} />);

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("alarm_active true iken uyarı bandını gösterir", () => {
    render(<AlarmBanner status={{ alarm_active: true, closed_duration: 2.84 }} />);

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(/uyku hali algılandı/i)).toBeInTheDocument();
    expect(screen.getByText(/gözler 2\.8 saniyedir kapalı/i)).toBeInTheDocument();
  });

  it("closed_duration olmadan çökmez, 0.0 gösterir", () => {
    render(<AlarmBanner status={{ alarm_active: true }} />);

    expect(screen.getByText(/gözler 0\.0 saniyedir kapalı/i)).toBeInTheDocument();
  });

  it("status verilmediğinde çökmez", () => {
    render(<AlarmBanner status={undefined} />);

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
