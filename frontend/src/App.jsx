import { useState } from "react";
import PlaygroundView from "./views/PlaygroundView.jsx";
import ApiPlaygroundView from "./views/ApiPlaygroundView.jsx";
import LayoutView from "./views/LayoutView.jsx";
import yellowsenseMark from "./assets/yellowsense-mark.png";

// Dashboard, Batch queue, Citizen upload and Benchmark views still exist in
// views/ but are off the nav: the demo shows only these three working tabs.
const VIEWS = {
  ocr: {
    label: "OCR",
    subtitle: "Word-level and page-level OCR across 10 Indian scripts — try a dataset sample or drop an image",
    Component: PlaygroundView,
  },
  layout: {
    label: "Layout",
    subtitle: "Detect a page's structure and rebuild it — titles, tables, form fields, figures, seals — in reading order",
    Component: LayoutView,
  },
  api: {
    label: "API playground",
    subtitle: "The developer console — send a document, get ULCA-schema JSON back",
    Component: ApiPlaygroundView,
  },
};

export default function App() {
  const [view, setView] = useState("ocr");
  const { subtitle, Component } = VIEWS[view];

  return (
    <>
      <header>
        <div className="header-top">
          <div className="brand">
            <img src={yellowsenseMark} alt="Yellowsense Technologies" className="brand-logo" />
            <h1>AksharDrishti</h1>
          </div>
          <nav className="view-tabs">
            {Object.entries(VIEWS).map(([key, { label }]) => (
              <button key={key} className={view === key ? "active" : ""} onClick={() => setView(key)}>
                {label}
              </button>
            ))}
          </nav>
        </div>
        <p className="subtitle">{subtitle}</p>
      </header>
      <Component />
    </>
  );
}
