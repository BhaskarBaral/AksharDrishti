import { useState } from "react";
import ClassicView from "./views/ClassicView.jsx";
import PlaygroundView from "./views/PlaygroundView.jsx";
import yellowsenseMark from "./assets/yellowsense-mark.png";

export default function App() {
  const [view, setView] = useState("playground"); // "playground" | "classic"

  return (
    <>
      <header>
        <div className="header-top">
          <div className="brand">
            <img src={yellowsenseMark} alt="Yellowsense Technologies" className="brand-logo" />
            <h1>AksharDrishti</h1>
          </div>
          <nav className="view-tabs">
            <button className={view === "playground" ? "active" : ""} onClick={() => setView("playground")}>
              Playground
            </button>
            <button className={view === "classic" ? "active" : ""} onClick={() => setView("classic")}>
              Dashboard
            </button>
          </nav>
        </div>
        <p className="subtitle">
          {view === "playground"
            ? "Drop an image or try a dataset sample and compare OCR engines instantly"
            : "Stage 3 prototype — browse the Stage 2 dataset and run OCR baselines against it"}
        </p>
      </header>
      {view === "playground" ? <PlaygroundView /> : <ClassicView />}
    </>
  );
}
