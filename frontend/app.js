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
      ["Nodes (Concepts)", data.graph.nodes.toLocaleString()],
      ["Edges (Known Links)", data.graph.edges.toLocaleString()],
      ["Total Papers", data.dataset.total_papers.toLocaleString()],
      ["Graph Density", data.graph.density],
      ["Avg Degree", data.graph.avg_degree],
    ];
    document.getElementById("graph-stats-list").innerHTML =
      statsRows.map(([k, v]) =>
        `<div class="stat-row"><span class="stat-key">${k}</span><span class="stat-val">${v}</span></div>`
      ).join("");

    // Architecture detail
    const arch = data.architecture;
    const archRows = [
      ["Encoder", arch.encoder],
      ["Decoder", arch.decoder],
      ["Input Dim", arch.input_dim],
      ["Hidden", arch.hidden_dim],
      ["Embed Dim", arch.embedding_dim],
      ["Dropout", arch.dropout],
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
    const counts = Object.values(data.dataset.sources);
    const palette = ["#60d0ff", "#4ade80", "#ff6b9d", "#b794f6", "#ffd700"];
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
const DEFAULT_CAMERA = { eye: { x: 1.25, y: 1.25, z: 1.0 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } };
let _autoRotateTimer = null;

async function loadGraph() {
  const container = document.getElementById("graph-container");
  const loading = document.getElementById("graph-loading");

  try {
    // 1. Fetch top predictions (for red edges)
    const predsPromise = loadGlobalPredictions();
    // 2. Fetch full graph layout
    const graphPromise = apiFetch("/api/graph_data");

    const [preds, graphData] = await Promise.all([predsPromise, graphPromise]);

    const { nodes, edges } = graphData;

    // Build Plotly Traces

    // 1. Existing Known Edges (faint white/gray)
    const ex = [], ey = [], ez = [];
    edges.forEach(([uIdx, vIdx]) => {
      const u = nodes[uIdx], v = nodes[vIdx];
      ex.push(u.x, v.x, null);
      ey.push(u.y, v.y, null);
      ez.push(u.z, v.z, null);
    });

    // 2. AI Predicted Gaps (bright red)
    const px = [], py = [], pz = [];
    // node id -> index map for quick lookup
    const nodeIdxMap = Object.fromEntries(nodes.map((n, i) => [n.id, i]));

    preds.forEach(p => {
      const uIdx = nodeIdxMap[p.node_a];
      const vIdx = nodeIdxMap[p.node_b];
      if (uIdx !== undefined && vIdx !== undefined) {
        const u = nodes[uIdx], v = nodes[vIdx];
        px.push(u.x, v.x, null);
        py.push(u.y, v.y, null);
        pz.push(u.z, v.z, null);
      }
    });

    const fig = [
      {
        type: "scatter3d",
        x: ex, y: ey, z: ez,
        mode: "lines",
        line: { color: "rgba(120, 120, 130, 0.15)", width: 1.2 },
        hoverinfo: "none",
        name: "Known Connections",
      },
      {
        type: "scatter3d",
        x: px, y: py, z: pz,
        mode: "lines",
        line: { color: "rgba(250, 80, 100, 0.9)", width: 4.0 },
        hoverinfo: "none",
        name: "AI Predicted Gaps",
      },
      {
        type: "scatter3d",
        x: nodes.map(n => n.x),
        y: nodes.map(n => n.y),
        z: nodes.map(n => n.z),
        mode: "markers+text",
        marker: {
          size: nodes.map(n => n.size),
          color: nodes.map(n => n.color),
          line: { color: "rgba(255,255,255,0.3)", width: 2 },
          opacity: 1.0
        },
        text: nodes.map(n => n.id.length > 25 ? n.id.slice(0, 22) + "…" : n.id),
        hovertext: nodes.map(n => `<b>${n.id}</b><br>Connections: ${n.degree}`),
        hoverinfo: "text",
        textfont: { color: "rgba(220,220,240,0.8)", size: 10 },
        textposition: "top center",
        name: "Research Concepts",
      },
    ];

    loading.style.display = "none";
    Plotly.newPlot("graph-container", fig, {
      paper_bgcolor: "#0f1117",
      scene: {
        xaxis: { showbackground: false, showticklabels: false, showgrid: true, gridcolor: "rgba(90,90,100,0.15)", zeroline: false },
        yaxis: { showbackground: false, showticklabels: false, showgrid: true, gridcolor: "rgba(90,90,100,0.15)", zeroline: false },
        zaxis: { showbackground: false, showticklabels: false, showgrid: true, gridcolor: "rgba(90,90,100,0.15)", zeroline: false },
        bgcolor: "#0f1117",
        camera: { ...DEFAULT_CAMERA },
        dragmode: "orbit",
        aspectmode: "cube",
      },
      margin: { l: 0, r: 0, t: 0, b: 0 },
      height: 600,
      legend: {
        bgcolor: "rgba(30,30,35,0.9)",
        bordercolor: "rgba(100,180,255,0.4)",
        borderwidth: 2,
        font: { color: "#e0e0e0", size: 12, family: "Arial" },
        x: 0.02, y: 0.98,
      },
      hovermode: "closest",
    }, {
      responsive: true,
      scrollZoom: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["toImage"],
    });

  } catch (e) {
    container.innerHTML = `<div class="loading-state"><p style="color:var(--pink)">❌ Failed to load graph: ${e.message}</p></div>`;
  }
}

/* ══════════════ Graph Controls ══════════════ */
function setDragMode(mode) {
  const el = document.getElementById("graph-container");
  if (!el || !el.layout) return;
  Plotly.relayout(el, { "scene.dragmode": mode });
  document.querySelectorAll(".graph-controls .ctrl-btn").forEach(b => b.classList.remove("active"));
  const btn = document.getElementById("btn-" + mode);
  if (btn) btn.classList.add("active");
}

function resetCamera() {
  const el = document.getElementById("graph-container");
  if (!el || !el.layout) return;
  if (_autoRotateTimer) { cancelAnimationFrame(_autoRotateTimer); _autoRotateTimer = null; document.getElementById("btn-autorotate").classList.remove("active"); }
  Plotly.relayout(el, { "scene.camera": { ...DEFAULT_CAMERA } });
}

function toggleAutoRotate() {
  const btn = document.getElementById("btn-autorotate");
  if (_autoRotateTimer) {
    cancelAnimationFrame(_autoRotateTimer);
    _autoRotateTimer = null;
    btn.classList.remove("active");
    return;
  }
  btn.classList.add("active");
  let angle = 0;
  const radius = 1.6;
  function rotate() {
    angle += 0.006;
    const el = document.getElementById("graph-container");
    if (!el || !el.layout) { _autoRotateTimer = null; return; }
    Plotly.relayout(el, {
      "scene.camera.eye": { x: radius * Math.cos(angle), y: radius * Math.sin(angle), z: 0.8 },
    });
    _autoRotateTimer = requestAnimationFrame(rotate);
  }
  _autoRotateTimer = requestAnimationFrame(rotate);
}

/* ══════════════ Search ══════════════ */
async function runSearch() {
  const query = document.getElementById("search-input").value.trim();
  const topK = parseInt(document.getElementById("topk-select").value);
  const area = document.getElementById("search-results");

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
