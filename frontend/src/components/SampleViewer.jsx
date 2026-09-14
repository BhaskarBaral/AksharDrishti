import { useEffect, useRef } from "react";
import { imageUrl } from "../api.js";

function pointInBBox(x, y, vertices) {
  const xs = vertices.map((v) => v.x);
  const ys = vertices.map((v) => v.y);
  return x >= Math.min(...xs) && x <= Math.max(...xs) && y >= Math.min(...ys) && y <= Math.max(...ys);
}

export default function SampleViewer({ script, split, filename, records, selectedField, onRunField }) {
  const canvasRef = useRef(null);
  const scaleRef = useRef(1);

  useEffect(() => {
    if (!script || !split || !filename) return;
    const img = new Image();
    img.onload = () => draw(img);
    img.src = imageUrl(script, split, filename);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [script, split, filename, records, selectedField]);

  function draw(img) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const maxWidth = 480;
    const scale = Math.min(1, maxWidth / img.naturalWidth);
    scaleRef.current = scale;
    canvas.width = img.naturalWidth * scale;
    canvas.height = img.naturalHeight * scale;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    records.forEach((rec, i) => {
      const verts = rec.boundingBox.vertices;
      ctx.beginPath();
      verts.forEach((v, vi) => {
        const x = v.x * scale;
        const y = v.y * scale;
        if (vi === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.strokeStyle = i === selectedField ? "#273A6B" : "rgba(168,69,44,0.75)";
      ctx.lineWidth = i === selectedField ? 2.5 : 1.5;
      ctx.stroke();
    });
  }

  function handleClick(evt) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scale = scaleRef.current;
    const x = ((evt.clientX - rect.left) * (canvas.width / rect.width)) / scale;
    const y = ((evt.clientY - rect.top) * (canvas.height / rect.height)) / scale;
    const hit = records.findIndex((rec) => pointInBBox(x, y, rec.boundingBox.vertices));
    if (hit >= 0) onRunField(hit);
  }

  const hasSample = Boolean(script && split && filename);

  return (
    <section id="viewer-panel">
      <h2>2. Sample</h2>
      {!hasSample ? (
        <div className="empty">Pick a sample on the left to view its fields.</div>
      ) : (
        <div>
          <div className="canvas-wrap">
            <canvas ref={canvasRef} onClick={handleClick} />
          </div>
          <table id="field-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Ground truth</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {records.map((rec, i) => (
                <tr key={i} className="field-row" onClick={() => onRunField(i)}>
                  <td>{i}</td>
                  <td>{rec.groundTruth}</td>
                  <td>Run &rarr;</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
