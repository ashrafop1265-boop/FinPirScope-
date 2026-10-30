const $ = s => document.querySelector(s);
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
let METRICS = null;

/* ---------- tabs ---------- */
document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t === b));
  document.querySelectorAll(".panel").forEach(p => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
}));

/* ---------- init ---------- */
async function init() {
  const res = await fetch("/api/metrics");
  METRICS = await res.json();
  fillSelects();
  renderResults();
  renderInsights();
}

function fillSelects() {
  const f = $("#form");
  f.release_month.innerHTML = MONTHS.map((m, i) => `<option value="${i + 1}">${m}</option>`).join("");
  f.release_month.value = 6;
  const defaults = { industry: "Hollywood / English", language: "English", age_rating: "PG-13", genre: "Action" };
  for (const [name, opts] of Object.entries(METRICS.options)) {
    f[name].innerHTML = opts.map(o => `<option>${o}</option>`).join("");
    if (opts.includes(defaults[name])) f[name].value = defaults[name];
  }
}

/* ---------- search + autofill ---------- */
let timer;
$("#search").addEventListener("input", e => {
  clearTimeout(timer);
  const q = e.target.value.trim();
  if (q.length < 2) { $("#suggest").innerHTML = ""; return; }
  timer = setTimeout(async () => {
    const hits = await (await fetch("/api/search?q=" + encodeURIComponent(q))).json();
    const box = $("#suggest");
    box.innerHTML = hits.map((h, i) => `<div data-i="${i}">${h.title}<span>${h.release_year ?? ""} · ${h.industry}</span></div>`).join("");
    box.querySelectorAll("div").forEach(d => d.addEventListener("click", () => {
      fillFromMovie(hits[+d.dataset.i]); box.innerHTML = ""; $("#search").value = hits[+d.dataset.i].title;
    }));
  }, 200);
});
document.addEventListener("click", e => { if (!e.target.closest(".search")) $("#suggest").innerHTML = ""; });

function fillFromMovie(m) {
  const f = $("#form");
  const set = (k, v) => { if (v !== null && v !== undefined && f[k]) f[k].value = v; };
  ["imdb_rating", "runtime", "release_year", "release_month", "days_since_posted"].forEach(k => set(k, m[k]));
  ["industry", "language", "age_rating", "genre"].forEach(k => {
    if (![...f[k].options].some(o => o.value === m[k])) return;
    set(k, m[k]);
  });
  ["tmdb_popularity", "tmdb_vote_count", "tmdb_vote_average", "tmdb_budget"].forEach(k => {
    f[k].value = m.has_tmdb && m[k] !== null ? m[k] : "";
  });
  if (m.has_tmdb) f.querySelector("details").open = true;
  window._actual = m;  // shown next to the prediction
}

/* ---------- predict ---------- */
$("#form").addEventListener("submit", async e => {
  e.preventDefault();
  const btn = e.target.querySelector("button.primary"), err = $("#error");
  err.textContent = ""; btn.disabled = true; btn.textContent = "Predicting…";
  const body = Object.fromEntries(new FormData(e.target).entries());
  try {
    const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Prediction failed");
    renderPrediction(data);
  } catch (ex) { err.textContent = ex.message; }
  btn.disabled = false; btn.textContent = "Predict piracy";
});

function modelRows(group) {
  return group.models.map(m => {
    const p = m.prob_positive;
    const pos = m.class === 1;
    return `<div class="mrow"><span>${m.model}</span>
      <div>${p === null ? '<span class="muted" style="font-size:12px">no probability</span>' :
        `<div class="gauge"><i style="width:${(p * 100).toFixed(0)}%"></i></div>`}</div>
      <span class="pill ${pos ? "pos" : "neg"}">${m.label}</span></div>`;
  }).join("");
}

function renderPrediction(d) {
  const a = window._actual;
  let actual = "";
  if (a && a.downloads != null) {
    actual = `<p class="hint">Dataset record for “${a.title}”: ${Number(a.downloads).toLocaleString()} downloads` +
      (a.leak_days != null && !isNaN(a.leak_days) ? `, uploaded ${a.leak_days} days after release` : "") + `.</p>`;
  }
  const q1 = d.q1, q2 = d.q2;
  const p1 = q1.avg_prob_positive;
  $("#output").innerHTML = `
    <div class="card">
      <h3>Q1 · Will it be pirated heavily?</h3>
      <div class="verdict"><span class="badge ${q1.consensus.toLowerCase()}">${q1.consensus} piracy</span>
        <div class="muted">${q1.agree} models agree${p1 !== null ? `<br>Average chance of “High”: <b>${(p1 * 100).toFixed(0)}%</b>` : ""}</div></div>
      ${modelRows(q1)}${actual}
    </div>
    <div class="card">
      <h3>Q2 · Will it be pirated quickly?</h3>
      <div class="verdict"><span class="badge ${q2.consensus.toLowerCase()}">${q2.consensus} leak</span>
        <div class="muted">${q2.agree} models agree<br>“Fast” = uploaded within ${Math.round(q2.median_leak)} days of release</div></div>
      ${modelRows(q2)}
    </div>`;
}

/* ---------- results tab ---------- */
function resultsTable(el, results) {
  const best = Math.max(...results.map(r => r.accuracy));
  el.innerHTML = `<table><tr><th>Model</th><th>Accuracy</th><th>F1 (macro)</th></tr>` +
    results.map(r => `<tr class="${r.accuracy === best ? "best" : ""}"><td>${r.model}</td><td>${(r.accuracy * 100).toFixed(1)}%</td><td>${r.f1.toFixed(3)}</td></tr>`).join("") + `</table>`;
}
function confusion(el, results, labels) {
  el.innerHTML = results.map(r => {
    const c = r.confusion, mx = Math.max(...c.flat());
    const cell = v => `<td style="background:rgba(124,92,255,${(0.12 + 0.7 * v / mx).toFixed(2)})">${v}</td>`;
    return `<div class="cm"><b>${r.model}</b><table>
      <tr><td></td><td class="muted">pred ${labels[0]}</td><td class="muted">pred ${labels[1]}</td></tr>
      <tr><td class="muted">${labels[0]}</td>${cell(c[0][0])}${cell(c[0][1])}</tr>
      <tr><td class="muted">${labels[1]}</td>${cell(c[1][0])}${cell(c[1][1])}</tr></table></div>`;
  }).join("");
}
const pct = v => (v * 100).toFixed(0) + "%";

function renderResults() {
  const { q1, q2, dataset } = METRICS;
  const t = q1.tiers;
  $("#q1-note").textContent = `Top third of downloads (${Number(t["2"].min).toLocaleString()}+) vs bottom third (up to ${Number(t["0"].max).toLocaleString()}). ` +
    `${dataset.q1_rows.toLocaleString()} movies; random guessing = 50%.`;
  Charts.vbars($("#q1-bars"), q1.results.map(r => r.model), q1.results.map(r => r.accuracy), { yMin: 0.5, yMax: 1, format: pct });
  resultsTable($("#q1-table"), q1.results);
  Charts.hbars($("#q1-imp"), q1.importance.map(i => ({ label: i.feature.replace("days_since_posted", "days_online"), value: i.importance })).reverse(), { format: v => v.toFixed(3) });
  confusion($("#q1-cm"), q1.results, q1.labels);

  const knn = q1.results.find(r => r.model === "KNN");
  Charts.lines($("#knn-curve"), Object.keys(knn.k_curve), [{ name: "CV accuracy", values: Object.values(knn.k_curve), color: "#22d3ee" }], { format: v => v.toFixed(3) });

  $("#q2-note").textContent = `Fast = uploaded within ${Math.round(q2.median_leak)} days of release (median). ` +
    `${dataset.q2_rows.toLocaleString()} movies uploaded within a year of release; random guessing = 50%.`;
  Charts.vbars($("#q2-bars"), q2.results.map(r => r.model), q2.results.map(r => r.accuracy), { yMin: 0.5, yMax: 1, format: pct, color: "#22d3ee" });
  resultsTable($("#q2-table"), q2.results);
  Charts.hbars($("#q2-imp"), q2.importance.map(i => ({ label: i.feature, value: i.importance })).reverse(), { format: v => v.toFixed(3), color: "#22d3ee" });
  confusion($("#q2-cm"), q2.results, q2.labels);
}

/* ---------- insights tab ---------- */
function renderInsights() {
  const { dataset: d, eda } = METRICS;
  $("#stats").innerHTML = [
    [d.rows.toLocaleString(), "movies after cleaning"],
    [d.tmdb_matched_pct + "%", "matched with TMDB"],
    [d.q1_rows.toLocaleString(), "movies in Question 1"],
    [d.q2_rows.toLocaleString(), "movies in Question 2"],
  ].map(([a, b]) => `<div class="stat"><b>${a}</b><span>${b}</span></div>`).join("");

  const mid = h => h.counts.map((_, i) => ((h.edges[i] + h.edges[i + 1]) / 2));
  const dl = eda.downloads_log_hist;
  Charts.vbars($("#eda-dl"), mid(dl).map(v => v.toFixed(1)), dl.counts, { xLabelEvery: 4, color: "#fbbf24" });
  const lk = eda.leak_hist;
  Charts.vbars($("#eda-leak"), mid(lk).map(v => Math.round(v)), lk.counts, { xLabelEvery: 5, color: "#a78bfa" });

  const ind = eda.median_downloads_by_industry;
  Charts.hbars($("#eda-ind"), ind.labels.map((l, i) => ({ label: l, value: ind.values[i] })), { labelW: 150 });
  const il = eda.median_leak_by_industry;
  Charts.hbars($("#eda-indleak"), il.labels.map((l, i) => ({ label: l, value: il.values[i] })), { labelW: 150, color: "#22d3ee" });
  const yr = eda.median_downloads_by_year;
  Charts.lines($("#eda-year"), yr.labels, [{ name: "median downloads", values: yr.values, color: "#34d399" }], { format: Charts.fmt });
  Charts.heatmap($("#eda-corr"), eda.correlation.labels, eda.correlation.matrix);
}

init().catch(e => { document.querySelector("main").insertAdjacentHTML("afterbegin",
  `<div class="card error">Could not load model data: ${e.message}. Did you run <code>python -m piracy.train</code>?</div>`); });
