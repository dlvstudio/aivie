import { useCallback, useEffect, useState } from "react";
import { fetchHealth } from "./api";
import EmbeddingExplorer from "./EmbeddingExplorer";
import GraphExplorer from "./GraphExplorer";
import InferenceLab from "./InferenceLab";

type Tab = "graph" | "inference" | "embedding";

export default function App() {
  const [tab, setTab] = useState<Tab>("graph");
  const [checkpoint, setCheckpoint] = useState<string>("…");
  const [apiError, setApiError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((h) => {
        setCheckpoint(h.checkpoint);
        setApiError(null);
      })
      .catch((e) => setApiError(String(e)));
  }, []);

  const onTab = useCallback((t: Tab) => setTab(t), []);

  return (
    <div className="app">
      <header className="header">
        <h1>AIVIE Graph UI</h1>
        <span className="meta">TK_001 · {checkpoint}</span>
        <nav className="tabs">
          <button
            type="button"
            className={tab === "graph" ? "active" : ""}
            onClick={() => onTab("graph")}
          >
            Parent Explorer
          </button>
          <button
            type="button"
            className={tab === "inference" ? "active" : ""}
            onClick={() => onTab("inference")}
          >
            Inference Lab
          </button>
          <button
            type="button"
            className={tab === "embedding" ? "active" : ""}
            onClick={() => onTab("embedding")}
          >
            Embedding E
          </button>
        </nav>
      </header>
      {apiError && (
        <div style={{ padding: "0.5rem 1rem", color: "#e06c75" }}>
          API: {apiError} — chạy server:{" "}
          <code>python -m viz.server.main</code>
        </div>
      )}
      {tab === "graph" && <GraphExplorer />}
      {tab === "inference" && <InferenceLab />}
      {tab === "embedding" && <EmbeddingExplorer />}
    </div>
  );
}
