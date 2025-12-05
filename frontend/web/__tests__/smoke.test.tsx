import { render, screen } from "@testing-library/react";
import React from "react";

function Hello() {
  return <div data-testid="hello">Hello</div>;
}

describe("frontend smoke", () => {
  it("renders a basic component", () => {
    render(<Hello />);
    expect(screen.getByTestId("hello")).toHaveTextContent("Hello");
  });
});
