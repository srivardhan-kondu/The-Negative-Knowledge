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
  // Lazy-load predictions when tab is first opened
  if (name === "predictions" && !window._predsLoaded) {
    loadGlobalPredictions();
    window._predsLoaded = true;
  }
}

/* ══════════════ Sidebar Toggle (mobile) ══════════════ */
function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  const overlay = document.getElementById("sidebar-overlay");
  sidebar.classList.toggle("open");
  overlay.classList.toggle("active");
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
    const palette = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#b45309"];
    Plotly.newPlot("pie-chart", [{
      type: "pie",
      labels: sources,
      values: counts,
      hole: 0.55,
      marker: { colors: palette.slice(0, sources.length), line: { color: "#ffffff", width: 2 } },
      textinfo: "label+percent",
      textfont: { color: "#1e293b", size: 12 },
    }], {
      paper_bgcolor: "transparent",
      margin: { l: 10, r: 10, t: 10, b: 10 },
      height: 320,
      showlegend: false,
      font: { color: "#1e293b", family: "Inter" },
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
         <button class="explain-btn" onclick="openExplain('${p.node_a.replace(/'/g, "\\'")}','${p.node_b.replace(/'/g, "\\'")}')" title="Why this prediction?">Why?</button>
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
    // 1. Fetch top predictions (for red edges) and graph layout in parallel
    const predsPromise = apiFetch("/api/predictions?top_k=20&n_samples=10000");
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
        line: { color: "rgba(100, 116, 139, 0.25)", width: 1.2 },
        hoverinfo: "none",
        name: "Known Connections",
      },
      {
        type: "scatter3d",
        x: px, y: py, z: pz,
        mode: "lines",
        line: { color: "rgba(220, 38, 38, 0.85)", width: 4.0 },
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
          line: { color: "rgba(30,41,59,0.2)", width: 2 },
          opacity: 1.0
        },
        text: nodes.map(n => n.id.length > 25 ? n.id.slice(0, 22) + "…" : n.id),
        hovertext: nodes.map(n => `<b>${n.id}</b><br>Connections: ${n.degree}`),
        hoverinfo: "text",
        textfont: { color: "rgba(30,41,59,0.75)", size: 10 },
        textposition: "top center",
        name: "Research Concepts",
      },
    ];

    loading.style.display = "none";
    Plotly.newPlot("graph-container", fig, {
      paper_bgcolor: "#ffffff",
      scene: {
        xaxis: { showbackground: false, showticklabels: false, showgrid: true, gridcolor: "rgba(100,116,139,0.12)", zeroline: false },
        yaxis: { showbackground: false, showticklabels: false, showgrid: true, gridcolor: "rgba(100,116,139,0.12)", zeroline: false },
        zaxis: { showbackground: false, showticklabels: false, showgrid: true, gridcolor: "rgba(100,116,139,0.12)", zeroline: false },
        bgcolor: "#ffffff",
        camera: { ...DEFAULT_CAMERA },
        dragmode: "orbit",
        aspectmode: "cube",
      },
      margin: { l: 0, r: 0, t: 0, b: 0 },
      height: container.clientHeight || 800,
      legend: {
        bgcolor: "rgba(255,255,255,0.95)",
        bordercolor: "rgba(37,99,235,0.3)",
        borderwidth: 2,
        font: { color: "#1e293b", size: 12, family: "Arial" },
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
              <button class="explain-btn" onclick="openExplain('${group.seed_node.replace(/'/g, "\\'")}','${p.node.replace(/'/g, "\\'")}')" title="Why this prediction?">Why?</button>
            </div>`).join("")}
        </div>`;
    });

    area.innerHTML = html;
  } catch (e) {
    area.innerHTML = `<div class="msg-box msg-error">❌ API error: ${e.message}</div>`;
  }
}

/* ══════════════ GNNExplainer Modal ══════════════ */

function openExplain(nodeA, nodeB) {
  const overlay = document.getElementById("explain-overlay");
  const modal = document.getElementById("explain-modal");
  const body = document.getElementById("explain-body");
  const title = document.getElementById("explain-title");

  overlay.classList.add("active");
  modal.classList.add("active");
  title.textContent = `Why: ${nodeA} ↔ ${nodeB}`;
  body.innerHTML = `<div class="loading-state" style="position:static;padding:60px 0;background:none">
    <div class="spinner"></div><p>Running GNNExplainer analysis…</p></div>`;

  apiFetch("/api/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_a: nodeA, node_b: nodeB }),
  }).then(data => renderExplanation(data))
    .catch(err => {
      body.innerHTML = `<div class="msg-box msg-error">❌ Explanation failed: ${err.message}</div>`;
    });
}

function closeExplainModal() {
  document.getElementById("explain-overlay").classList.remove("active");
  document.getElementById("explain-modal").classList.remove("active");
}

function renderExplanation(d) {
  const body = document.getElementById("explain-body");

  // Confidence color mapping
  const confColors = { very_strong: "var(--green)", strong: "#2563eb", moderate: "var(--gold)", weak: "var(--pink)" };
  const confColor = confColors[d.confidence_level] || "var(--text-muted)";

  // Attribution bars
  const topoPct = d.feature_attribution ? (d.feature_attribution.topology_pct * 100).toFixed(1) : 50;
  const semPct = d.feature_attribution ? (d.feature_attribution.semantics_pct * 100).toFixed(1) : 50;

  let html = ``;

  // ── Section 1: Score & Confidence ──
  html += `
    <div class="xai-section">
      <div class="xai-score-banner" style="border-color: ${confColor}">
        <div class="xai-score-big" style="color: ${confColor}">${d.score_pct}</div>
        <div class="xai-confidence-label">${d.confidence_level.replace('_', ' ').toUpperCase()}</div>
        <div class="xai-confidence-desc">${d.confidence_description}</div>
        <div class="xai-confidence-gauge">
          <div class="xai-gauge-bg">
            <div class="xai-gauge-zone zone-weak"></div>
            <div class="xai-gauge-zone zone-moderate"></div>
            <div class="xai-gauge-zone zone-strong"></div>
            <div class="xai-gauge-zone zone-vstrong"></div>
            <div class="xai-gauge-needle" style="left:${Math.min(100, ((d.score - 0.5) / 0.5) * 100)}%"></div>
          </div>
          <div class="xai-gauge-labels"><span>50%</span><span>60%</span><span>70%</span><span>80%</span><span>100%</span></div>
        </div>
      </div>
    </div>`;

  // ── Section 2: Gradient Feature Attribution ──
  if (d.feature_attribution) {
    const fa = d.feature_attribution;
    html += `
    <div class="xai-section">
      <h3 class="xai-heading">🧬 GNN Feature Attribution</h3>
      <p class="xai-desc">How much of the prediction comes from <b>graph structure</b> (Node2Vec) vs <b>medical meaning</b> (SciBERT).</p>
      <div class="xai-attr-bar">
        <div class="xai-attr-topo" style="width:${topoPct}%">
          <span>Topology ${topoPct}%</span>
        </div>
        <div class="xai-attr-sem" style="width:${semPct}%">
          <span>Semantics ${semPct}%</span>
        </div>
      </div>
      <div class="xai-attr-legend">
        <span><span class="legend-dot" style="background:var(--blue)"></span> Node2Vec — "similar graph neighborhoods"</span>
        <span><span class="legend-dot" style="background:var(--purple)"></span> SciBERT — "similar medical meaning"</span>
      </div>
    </div>`;

    // Top feature dimensions
    if (fa.top_features && fa.top_features.length) {
      html += `
      <div class="xai-section">
        <h3 class="xai-heading">🔍 Top Activated Feature Dimensions</h3>
        <div class="xai-features-grid">
          ${fa.top_features.map((f, i) => {
            const maxImp = fa.top_features[0].importance || 1;
            const barW = Math.max(5, (f.importance / maxImp) * 100);
            const isTopo = f.source.includes('Node2Vec');
            return `<div class="xai-feat-row">
              <span class="xai-feat-dim">dim ${f.dim}</span>
              <div class="xai-feat-bar-bg"><div class="xai-feat-bar" style="width:${barW}%;background:${isTopo ? 'var(--blue)' : 'var(--purple)'}"></div></div>
              <span class="xai-feat-src">${isTopo ? 'Topo' : 'Sem'}</span>
            </div>`;
          }).join('')}
        </div>
      </div>`;
    }
  }

  // ── Section 3: Common Neighbors ──
  html += `
    <div class="xai-section">
      <h3 class="xai-heading">🤝 Common Neighbors (${d.common_neighbor_count})</h3>
      <p class="xai-desc">Concepts already connected to <b>both</b> nodes — these are the bridge concepts that make the GNN believe a direct link should exist.</p>`;
  if (d.common_neighbors.length > 0) {
    html += `<div class="xai-chip-list">${d.common_neighbors.map(n =>
      `<span class="xai-chip">🔗 ${n.node} <small>(${n.degree} links)</small></span>`
    ).join('')}</div>`;
  } else {
    html += `<div class="xai-empty">No common neighbors — prediction relies on topology + semantic similarity, not direct bridges.</div>`;
  }
  html += `</div>`;

  // ── Section 4: Shortest Path ──
  html += `
    <div class="xai-section">
      <h3 class="xai-heading">🛤️ Shortest Path (${d.shortest_path_length > 0 ? d.shortest_path_length + ' hops' : 'unreachable'})</h3>
      <p class="xai-desc">The shortest chain of connections between the two concepts in the current knowledge graph.</p>`;
  if (d.shortest_path.length > 0) {
    html += `<div class="xai-path">${d.shortest_path.map((n, i) => {
      const isEnd = (i === 0 || i === d.shortest_path.length - 1);
      return `<span class="xai-path-node ${isEnd ? 'endpoint' : ''}">${n}</span>${i < d.shortest_path.length - 1 ? '<span class="xai-path-arrow">→</span>' : ''}`;
    }).join('')}</div>`;
  } else {
    html += `<div class="xai-empty">Nodes are not connected — they exist in separate graph components.</div>`;
  }
  html += `</div>`;

  // ── Section 5: Embedding Similarity ──
  const simPct = ((d.embedding_similarity + 1) / 2 * 100).toFixed(1); // map [-1,1] to [0,100]
  html += `
    <div class="xai-section">
      <h3 class="xai-heading">📐 GNN Embedding Similarity</h3>
      <p class="xai-desc">Cosine similarity between the 64-D GNN-learned node embeddings. Higher = model sees these concepts as more related.</p>
      <div class="xai-sim-bar-wrap">
        <div class="xai-sim-bar-bg">
          <div class="xai-sim-bar-fill" style="width:${simPct}%"></div>
        </div>
        <span class="xai-sim-val">${d.embedding_similarity.toFixed(4)}</span>
      </div>
    </div>`;

  // ── Section 6: Influential Neighbors ──
  html += `
    <div class="xai-section">
      <h3 class="xai-heading">🌟 Most Influential Neighbors</h3>
      <p class="xai-desc">Neighbors of each node whose GNN embeddings are most similar to the <i>other</i> node — they act as structural evidence for the predicted link.</p>
      <div class="xai-inf-cols">
        <div class="xai-inf-col">
          <div class="xai-inf-label">Neighbors of <b>${d.node_a}</b> most similar to <b>${d.node_b}</b></div>
          ${d.influential_neighbors_a.length ? d.influential_neighbors_a.map(n =>
            `<div class="xai-inf-row"><span class="xai-inf-node">${n.node}</span><span class="xai-inf-rel" style="color:${n.relevance > 0.5 ? 'var(--green)' : 'var(--text-muted)'}">${(n.relevance * 100).toFixed(1)}%</span></div>`
          ).join('') : '<div class="xai-empty">No neighbors</div>'}
        </div>
        <div class="xai-inf-col">
          <div class="xai-inf-label">Neighbors of <b>${d.node_b}</b> most similar to <b>${d.node_a}</b></div>
          ${d.influential_neighbors_b.length ? d.influential_neighbors_b.map(n =>
            `<div class="xai-inf-row"><span class="xai-inf-node">${n.node}</span><span class="xai-inf-rel" style="color:${n.relevance > 0.5 ? 'var(--green)' : 'var(--text-muted)'}">${(n.relevance * 100).toFixed(1)}%</span></div>`
          ).join('') : '<div class="xai-empty">No neighbors</div>'}
        </div>
      </div>
    </div>`;

  // ── Section 7: Paper Evidence ──
  const pe = d.paper_evidence;
  html += `
    <div class="xai-section">
      <h3 class="xai-heading">📚 Paper Evidence (Provenance)</h3>
      <p class="xai-desc">Research papers from the training corpus that mention each concept — showing the real-world evidence behind these graph nodes.</p>
      <div class="xai-inf-cols">
        <div class="xai-inf-col">
          <div class="xai-inf-label"><b>${d.node_a}</b> — ${pe.papers_a} papers</div>
          ${pe.titles_a.length ? pe.titles_a.map(t => `<div class="xai-paper-title">📄 ${t}</div>`).join('') : '<div class="xai-empty">No papers found</div>'}
        </div>
        <div class="xai-inf-col">
          <div class="xai-inf-label"><b>${d.node_b}</b> — ${pe.papers_b} papers</div>
          ${pe.titles_b.length ? pe.titles_b.map(t => `<div class="xai-paper-title">📄 ${t}</div>`).join('') : '<div class="xai-empty">No papers found</div>'}
        </div>
      </div>
    </div>`;

  // ── Section 8: Node Stats ──
  html += `
    <div class="xai-section">
      <h3 class="xai-heading">📊 Node Statistics</h3>
      <div class="xai-stats-row">
        <div class="xai-stat-card">
          <div class="xai-stat-label">${d.node_a}</div>
          <div class="xai-stat-val">${d.node_a_degree} connections</div>
        </div>
        <div class="xai-stat-card">
          <div class="xai-stat-label">${d.node_b}</div>
          <div class="xai-stat-val">${d.node_b_degree} connections</div>
        </div>
      </div>
    </div>`;

  // ── Disclaimer ──
  html += `
    <div class="xai-disclaimer">
      ⚠️ This explanation is generated by gradient-based GNN attribution (analogous to Grad-CAM for CNNs).
      Feature importance is computed by backpropagating the link prediction score through the GCN encoder layers.
      All explanations are model-intrinsic and require domain expert validation.
    </div>`;

  body.innerHTML = html;
}

/* ══════════════ Init ══════════════ */
document.addEventListener("DOMContentLoaded", () => {
  loadMetrics();
});
