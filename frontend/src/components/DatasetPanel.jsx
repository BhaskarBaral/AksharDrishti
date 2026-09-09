export default function DatasetPanel({ scripts, script, onScriptChange, samples, selected, onSelectSample }) {
  return (
    <section id="dataset-panel">
      <h2>1. Dataset</h2>
      <div className="row">
        <label htmlFor="script-select">Script</label>
        <select id="script-select" value={script ?? ""} onChange={(e) => onScriptChange(e.target.value)}>
          {scripts.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>
      <div id="sample-list" className="thumb-grid">
        {samples.map((s) => {
          const isSelected = selected && selected.split === s.split && selected.filename === s.filename;
          return (
            <div
              key={`${s.split}/${s.filename}`}
              className={`thumb${isSelected ? " selected" : ""}`}
              onClick={() => onSelectSample(s.split, s.filename)}
            >
              <span>
                {s.fieldCount} fields{" "}
                {s.isSynthetic && <span className="tag-synthetic">synthetic</span>}
              </span>
              <span className="fname">{s.filename}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
