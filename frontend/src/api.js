async function jsonFetch(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const getScripts = () => jsonFetch("/api/dataset/scripts");

export const getEngines = (script) => jsonFetch(`/api/ocr/engines/${script}`);

export const getSamples = (script) =>
  jsonFetch(`/api/dataset/scripts/${script}/samples`);

export const getRecords = (script, split, filename) =>
  jsonFetch(`/api/dataset/scripts/${script}/${split}/${filename}/records`);

export const imageUrl = (script, split, filename) =>
  `/api/dataset/scripts/${script}/${split}/${filename}/image`;

export function runOnSample(script, split, filename, fieldIndex, engines) {
  const form = new FormData();
  form.set("script", script);
  form.set("split", split);
  form.set("filename", filename);
  form.set("field_index", fieldIndex);
  engines.forEach((name) => form.append("engines", name));
  return jsonFetch("/api/ocr/run-on-sample", { method: "POST", body: form });
}

export function runOnUpload(script, file, engines) {
  const form = new FormData();
  form.set("script", script);
  form.set("file", file);
  engines.forEach((name) => form.append("engines", name));
  return jsonFetch("/api/ocr/run-on-upload", { method: "POST", body: form });
}

export function runBenchmark(script, engines, limit) {
  const form = new FormData();
  form.set("script", script);
  form.set("limit", limit);
  engines.forEach((name) => form.append("engines", name));
  return jsonFetch("/api/ocr/benchmark", { method: "POST", body: form });
}

export function runPageUpload(script, file, engines) {
  const form = new FormData();
  form.set("script", script);
  form.set("file", file);
  (engines || []).forEach((name) => form.append("engines", name));
  return jsonFetch("/api/ocr/run-page-upload", { method: "POST", body: form });
}
