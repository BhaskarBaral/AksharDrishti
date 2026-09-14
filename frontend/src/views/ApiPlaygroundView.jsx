import "./demoScreens.css";

export default function ApiPlaygroundView() {
  return (
    <main className="demo-root">
      <div className="s-head">
        <div>
          <h1 className="title">API playground</h1>
          <p className="desc">
            The developer console — send a document, get ULCA-schema JSON back. This is what makes the tool
            deployment-ready and Bhashini-compatible: the response shape follows the locked ULCA schema, so
            integrators can wire it straight into the national stack.
          </p>
        </div>
      </div>

      <div className="api-pg">
        <div className="api-card">
          <div className="api-chead"><span>Request</span><span className="api-method">POST</span></div>
          <div className="api-cbody">
            <div className="api-field">
              <label>Endpoint</label>
              <input className="api-ctl" value="/v1/ocr/recognize" readOnly />
            </div>
            <div className="api-field">
              <label>Source language</label>
              <select className="api-ctl" defaultValue="hi">
                <option value="hi">Hindi (hi)</option>
                <option value="bn">Bengali (bn)</option>
                <option value="ta">Tamil (ta)</option>
                <option value="ur">Urdu (ur)</option>
              </select>
            </div>
            <div className="api-field">
              <label>Script</label>
              <select className="api-ctl" defaultValue="Deva">
                <option value="Deva">Devanagari (Deva)</option>
                <option value="Beng">Bengali (Beng)</option>
                <option value="Taml">Tamil (Taml)</option>
              </select>
            </div>
            <div className="api-field">
              <label>Output format</label>
              <select className="api-ctl" defaultValue="ulca">
                <option value="ulca">ULCA JSON</option>
                <option value="text">Plain text</option>
                <option value="pdf">Searchable PDF</option>
              </select>
            </div>
            <div className="api-field">
              <label>Document</label>
              <div className="api-filepick">📄 ration-card-2841.jpg</div>
            </div>
            <button className="api-send">Send request</button>
          </div>
        </div>

        <div className="api-card">
          <div className="api-chead"><span>Response</span><span className="tag conf-high">200 OK · 1.2s</span></div>
          <div className="api-cbody">
            <pre className="api-code">
              <span className="c">{"// ULCA-compatible OCR record"}</span>
              {"\n{\n  "}
              <span className="k">"datasetType"</span>
              {": "}
              <span className="s">"ocr-corpus"</span>
              {",\n  "}
              <span className="k">"languages"</span>
              {": {\n    "}
              <span className="k">"sourceLanguage"</span>
              {": "}
              <span className="s">"hi"</span>
              {",\n    "}
              <span className="k">"sourceScriptCode"</span>
              {": "}
              <span className="s">"Deva"</span>
              {"\n  },\n  "}
              <span className="k">"imageTextType"</span>
              {": "}
              <span className="s">"computer-typed-text"</span>
              {",\n  "}
              <span className="k">"groundTruth"</span>
              {": "}
              <span className="s">"नाम: सुनीता देवी"</span>
              {",\n  "}
              <span className="k">"confidence"</span>
              {": "}
              <span className="n">0.99</span>
              {",\n  "}
              <span className="k">"boundingBox"</span>
              {": {\n    "}
              <span className="k">"vertices"</span>
              {": [\n      {"}
              <span className="k">"x"</span>
              {":"}
              <span className="n">120</span>
              {","}
              <span className="k">"y"</span>
              {":"}
              <span className="n">80</span>
              {"},\n      {"}
              <span className="k">"x"</span>
              {":"}
              <span className="n">640</span>
              {","}
              <span className="k">"y"</span>
              {":"}
              <span className="n">80</span>
              {"},\n      {"}
              <span className="k">"x"</span>
              {":"}
              <span className="n">640</span>
              {","}
              <span className="k">"y"</span>
              {":"}
              <span className="n">130</span>
              {"},\n      {"}
              <span className="k">"x"</span>
              {":"}
              <span className="n">120</span>
              {","}
              <span className="k">"y"</span>
              {":"}
              <span className="n">130</span>
              {"}\n    ]\n  }\n}"}
            </pre>
          </div>
        </div>
      </div>
      <p className="demo-note">Sample request/response — this endpoint isn't exposed publicly yet.</p>
    </main>
  );
}
