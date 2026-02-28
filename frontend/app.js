/**
 * MindGap Frontend — app.js
 * Fetches data from the Flask REST API and renders the UI.
 * API base: same origin (Flask serves both API and static frontend)
 */

const API = "";  // same origin — change to "http://localhost:5000" if running separately

/* ══════════════ Utility ══════════════ */
async function apiFetch(path, opts = {}) {
  try {
    const res = await fetch(API + path, opts);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.error("API error:", path, e);
    throw e;
  }
}

function setApiStatus(ok) {
  const dot = document.getElementById("api-status");
  const dotEl = document.querySelector(".api-dot");
  if (ok) {
    dot.textContent = "API connected";
    dotEl.className = "api-dot green";
  } else {
    dot.textContent = "API unavailable";
    dotEl.className = "api-dot red";
  }
}

/* ══════════════ Tab Switching ══════════════ */
function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
  document.getElementById("tab-" + name).classList.add("active");
  document.getElementById("content-" + name).classList.add("active");

  // Lazy-load graph when tab is first opened
  if (name === "graph" && !window._graphLoaded) {
    loadGraph();
    window._graphLoaded = true;
  }
}

/* ══════════════ Sidebar Toggle (mobile) ══════════════ */
function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
}

/* ══════════════ Load Metrics (sidebar + metrics tab) ══════════════ */
async function loadMetrics() {
  try {
    const data = await apiFetch("/api/metrics");
    setApiStatus(true);

    // Sidebar
    document.getElementById("auc-value").textContent = data.roc_auc_pct;
    document.getElementById("node-count").textContent = data.graph.nodes.toLocaleString();
    document.getElementById("edge-count").textContent = data.graph.edges.toLocaleString();
    document.getElementById("paper-count").textContent = data.dataset.total_papers.toLocaleString();

    // Sources sidebar
    const srcHtml = Object.entries(data.dataset.sources)
      .sort((a, b) => b[1] - a[1])
      .map(([src, cnt]) =>
        `<div class="source-row">
           <span class="source-name">${src}</span>
           <span class="source-count">${cnt}</span>
         </div>`)
      .join("");
    document.getElementById("sources-list").innerHTML = srcHtml;

    // Metrics tab — AUC
    document.getElementById("metrics-auc").textContent = data.roc_auc_pct;
    const barPct = Math.min(100, ((data.roc_auc - 0.5) / 0.5) * 100);
    setTimeout(() => {
      document.getElementById("auc-bar").style.width = barPct + "%";
    }, 200);

    // Graph stats
    const statsRows = [
      ["Nodes (Concepts)",   data.graph.nodes.toLocaleString()],
      ["Edges (Known Links)", data.graph.edges.toLocaleString()],
      ["Total Papers",        data.dataset.total_papers.toLocaleString()],
      ["Graph Density",       data.graph.density],
      ["Avg Degree",          data.graph.avg_degree],
    ];
    document.getElementById("graph-stats-list").innerHTML =
      statsRows.map(([k, v]) =>
        `<div class="stat-row"><span class="stat-key">${k}</span><span class="stat-val">${v}</span></div>`
      ).join("");

    // Architecture detail
    const arch = data.architecture;
    const archRows = [
      ["Encoder",   arch.encoder],
      ["Decoder",   arch.decoder],
      ["Input Dim", arch.input_dim],
      ["Hidden",    arch.hidden_dim],
      ["Embed Dim", arch.embedding_dim],
      ["Dropout",   arch.dropout],
      ["Optimizer", arch.optimizer + ` · lr=${arch.lr}`],
      ["Max Epochs", arch.max_epochs],
      ["Early Stop", `patience ${arch.early_stop_patience}`],
    ];
    document.getElementById("arch-detail-list").innerHTML =
      archRows.map(([k, v]) =>
        `<div class="stat-row">
           <span class="stat-key">${k}</span>
           <span class="stat-val" style="color:var(--purple)">${v}</span>
         </div>`
      ).join("");

    // Pie chart
    const sources = Object.keys(data.dataset.sources);
    const counts  = Object.values(data.dataset.sources);
    const palette = ["#60d0ff","#4ade80","#ff6b9d","#b794f6","#ffd700"];
    Plotly.newPlot("pie-chart", [{
      type: "pie",
      labels: sources,
      values: counts,
      hole: 0.55,
      marker: { colors: palette.slice(0, sources.length), line: { color: "#0f1117", width: 2 } },
      textinfo: "label+percent",
      textfont: { color: "#e0e0e0", size: 12 },
    }], {
      paper_bgcolor: "transparent",
      margin: { l: 10, r: 10, t: 10, b: 10 },
      height: 320,
      showlegend: false,
      font: { color: "#e0e0e0", family: "Inter" },
    }, { responsive: true, displayModeBar: false });

  } catch {
    setApiStatus(false);
  }
}

/* ══════════════ Load Global Predictions (graph tab) ══════════════ */
async function loadGlobalPredictions() {
  const el = document.getElementById("global-predictions");
  try {
    const preds = await apiFetch("/api/predictions?top_k=20&n_samples=10000");
    if (!preds.length) { el.innerHTML = "<p style='color:var(--text-muted)'>No predictions available.</p>"; return; }
    el.innerHTML = preds.map((p, i) =>
      `<div class="pred-row">
         <span class="pred-rank">${i + 1}.</span>
         <span class="pred-nodes">
           <span class="pred-a">${p.node_a}</span>
           <span class="pred-arrow">↔</span>
           <span class="pred-b">${p.node_b}</span>
         </span>
         <span class="pred-score">${p.score_pct}</span>
       </div>`
    ).join("");
    return preds;
  } catch {
    el.innerHTML = `<div class="msg-box msg-error">Failed to load predictions.</div>`;
    return [];
  }
}

/* ══════════════ 3D Graph ══════════════ */
async function loadGraph() {
  const container = document.getElementById("graph-container");
  const loading   = document.getElementById("graph-loading");

  // Fetch predictions first (we need them to draw red edges)
  const preds = await loadGlobalPredictions();

  // Fetch nodes via health endpoint to get list
  // We'll build the graph from the predictions + metrics data already loaded
  // Use a simple fetch for the graph data
  try {
    // Get full node list from search with wildcard (hack: search empty returns error, use metrics)
    // Instead, just use the predictions nodes for the 3D graph highlights
    // For the full graph, we build a virtual graph from top predictions neighbors

    // Draw a focused graph: just the nodes involved in top predictions
    const nodeSet = new Set();
    const edges = [];
    preds.forEach(p => {
      nodeSet.add(p.node_a);
      nodeSet.add(p.node_b);
      edges.push({ a: p.node_a, b: p.node_b, score: p.score });
    });

    const nodes = Array.from(nodeSet);
    const n = nodes.length;
    const idx = Object.fromEntries(nodes.map((nd, i) => [nd, i]));

    // Random 3D positions (reproducible seed via simple LCG)
    const pos = nodes.map((_, i) => {
      const t = (i * 2.399963) % (2 * Math.PI);  // golden angle
      const z = 1 - (i / (n - 1)) * 2;
      const r = Math.sqrt(1 - z * z);
      return { x: r * Math.cos(t), y: r * Math.sin(t), z };
    });

    const palette = ["#60d0ff","#4ade80","#b794f6","#ffd700","#ff6b9d"];

    // Predicted edges (red)
    const ex = [], ey = [], ez = [];
    edges.forEach(e => {
      const ai = idx[e.a], bi = idx[e.b];
      ex.push(pos[ai].x, pos[bi].x, null);
      ey.push(pos[ai].y, pos[bi].y, null);
      ez.push(pos[ai].z, pos[bi].z, null);
    });

    const fig = [
      {
        type: "scatter3d",
        x: ex, y: ey, z: ez,
        mode: "lines",
        line: { color: "rgba(255,80,110,0.85)", width: 4 },
        hoverinfo: "none",
        name: "AI Predicted Gaps",
      },
      {
        type: "scatter3d",
        x: pos.map(p => p.x),
        y: pos.map(p => p.y),
        z: pos.map(p => p.z),
        mode: "markers+text",
        marker: {
          size: 14,
          color: nodes.map((_, i) => palette[i % palette.length]),
          line: { color: "rgba(255,255,255,0.2)", width: 1.5 },
        },
        text: nodes.map(n => n.length > 25 ? n.slice(0, 22) + "…" : n),
        hovertext: nodes.map((nd, i) => {
          const related = edges.filter(e => e.a === nd || e.b === nd);
          return `<b>${nd}</b><br>Predicted connections: ${related.length}`;
        }),
        hoverinfo: "text",
        textfont: { color: "rgba(220,220,240,0.7)", size: 9 },
        textposition: "top center",
        name: "Research Concepts",
      },
    ];

    loading.style.display = "none";
    Plotly.newPlot("graph-container", fig, {
      paper_bgcolor: "#0f1117",
      scene: {
        xaxis: { showbackground: false, showticklabels: false, gridcolor: "rgba(80,80,100,0.12)" },
        yaxis: { showbackground: false, showticklabels: false, gridcolor: "rgba(80,80,100,0.12)" },
        zaxis: { showbackground: false, showticklabels: false, gridcolor: "rgba(80,80,100,0.12)" },
        bgcolor: "#0f1117",
        camera: { eye: { x: 1.3, y: 1.3, z: 0.9 } },
      },
      margin: { l: 0, r: 0, t: 0, b: 0 },
      height: 680,
      legend: {
        bgcolor: "rgba(20,22,35,0.92)",
        bordercolor: "rgba(100,180,255,0.25)",
        borderwidth: 1,
        font: { color: "#e0e0e0", size: 12, family: "Inter" },
        x: 0.02, y: 0.98,
      },
      hovermode: "closest",
      font: { color: "#e0e0e0", family: "Inter" },
    }, { responsive: true, scrollZoom: true, displaylogo: false });

  } catch (e) {
    container.innerHTML = `<div class="loading-state"><p style="color:var(--pink)">❌ Failed to load graph: ${e.message}</p></div>`;
  }
}

/* ══════════════ Search ══════════════ */
async function runSearch() {
  const query = document.getElementById("search-input").value.trim();
  const topK  = parseInt(document.getElementById("topk-select").value);
  const area  = document.getElementById("search-results");

  if (!query) {
    area.innerHTML = `<div class="msg-box msg-warn">Please enter a search term.</div>`;
    return;
  }

  area.innerHTML = `<div class="loading-state" style="position:static;padding:40px 0;background:none">
    <div class="spinner"></div><p>Running GNN predictions for "<b>${query}</b>"…</p></div>`;

  try {
    const data = await apiFetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });

    if (data.matches_found === 0) {
      area.innerHTML = `
        <div class="msg-box msg-error">
          ❌ <b>No nodes found</b> matching "<b>${query}</b>" in the mental health knowledge graph.<br><br>
          This concept may not exist in the current training domain.
          Multi-domain support (e.g. environmental science) is planned for the next sprint.
        </div>`;
      return;
    }

    let html = "";
    if (data.matches_found > 1) {
      html += `<div class="msg-box msg-warn">Found ${data.matches_found} matching nodes. Showing predictions for each.</div>`;
    }

    data.results.forEach(group => {
      html += `
        <div class="seed-group">
          <div class="seed-title">🔵 ${group.seed_node}</div>
          <div class="seed-meta">Current graph connections: ${group.degree}</div>
          ${group.predictions.map((p, i) => `
            <div class="pred-row">
              <span class="pred-rank">${i + 1}.</span>
              <span class="pred-nodes">
                <span class="pred-a">${group.seed_node}</span>
                <span class="pred-arrow">↔</span>
                <span class="pred-b">${p.node}</span>
              </span>
              <span class="pred-score">${p.score_pct}</span>
            </div>`).join("")}
        </div>`;
    });

    area.innerHTML = html;
  } catch (e) {
    area.innerHTML = `<div class="msg-box msg-error">❌ API error: ${e.message}</div>`;
  }
}

/* ══════════════ Init ══════════════ */
document.addEventListener("DOMContentLoaded", () => {
  loadMetrics();
});
