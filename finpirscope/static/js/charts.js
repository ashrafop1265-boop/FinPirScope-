/* Tiny dependency-free SVG chart helpers (no CDN needed, works offline). */
const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const COLORS = ["#7c5cff", "#22d3ee", "#34d399", "#fbbf24", "#f87171", "#a78bfa"];
  const fmt = n => Math.abs(n) >= 1e6 ? (n / 1e6).toFixed(1) + "M" : Math.abs(n) >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(+n.toFixed(2));
  const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  function svg(el, w, h, inner) {
    el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img">${inner}</svg>`;
  }

  /* horizontal bars: items = [{label, value, color?}] */
  function hbars(el, items, { max, format = fmt, color = COLORS[0], labelW = 130 } = {}) {
    const row = 26, w = 520, h = items.length * row + 6;
    const m = max || Math.max(...items.map(i => i.value), 1e-9);
    const bw = w - labelW - 60;
    svg(el, w, h, items.map((it, i) => {
      const y = i * row + 3, len = Math.max(1, (it.value / m) * bw);
      return `<text x="${labelW - 8}" y="${y + 15}" text-anchor="end">${esc(it.label)}</text>
        <rect x="${labelW}" y="${y + 3}" width="${len}" height="16" rx="4" fill="${it.color || color}"/>
        <text x="${labelW + len + 6}" y="${y + 15}" class="t">${format(it.value)}</text>`;
    }).join(""));
  }

  /* vertical bars with x labels, used for histograms and accuracy */
  function vbars(el, labels, values, { color = COLORS[0], format = fmt, yMax, yMin = 0, h = 220, xLabelEvery = 1, colors } = {}) {
    const w = 520, pl = 44, pb = 28, pt = 12, ph = h - pb - pt, pw = w - pl - 8;
    const max = yMax ?? (Math.max(...values) * 1.08 || 1);
    const bw = pw / values.length;
    let s = "";
    for (let t = 0; t <= 4; t++) {
      const v = yMin + (max - yMin) * t / 4, y = pt + ph - (ph * t / 4);
      s += `<line x1="${pl}" x2="${w - 8}" y1="${y}" y2="${y}" stroke="var(--line)"/><text x="${pl - 6}" y="${y + 4}" text-anchor="end">${format(v)}</text>`;
    }
    values.forEach((v, i) => {
      const bh = Math.max(0, (v - yMin) / (max - yMin) * ph), x = pl + i * bw;
      s += `<rect x="${x + 1.5}" y="${pt + ph - bh}" width="${Math.max(1, bw - 3)}" height="${bh}" rx="3" fill="${colors ? colors[i] : color}"><title>${esc(labels[i])}: ${format(v)}</title></rect>`;
      if (i % xLabelEvery === 0) s += `<text x="${x + bw / 2}" y="${h - 8}" text-anchor="middle">${esc(labels[i])}</text>`;
    });
    svg(el, w, h, s);
  }

  /* multi-series line chart: series = [{name, values, color?}] */
  function lines(el, xs, series, { yMin, yMax, h = 220, format = v => v.toFixed(2) } = {}) {
    const w = 520, pl = 44, pb = 28, pt = 12, ph = h - pb - pt, pw = w - pl - 12;
    const all = series.flatMap(s => s.values);
    const lo = yMin ?? Math.min(...all), hi = yMax ?? Math.max(...all);
    const X = i => pl + (xs.length === 1 ? pw / 2 : i * pw / (xs.length - 1));
    const Y = v => pt + ph - ((v - lo) / ((hi - lo) || 1)) * ph;
    let s = "";
    for (let t = 0; t <= 4; t++) {
      const v = lo + (hi - lo) * t / 4, y = Y(v);
      s += `<line x1="${pl}" x2="${w - 12}" y1="${y}" y2="${y}" stroke="var(--line)"/><text x="${pl - 6}" y="${y + 4}" text-anchor="end">${format(v)}</text>`;
    }
    const step = Math.ceil(xs.length / 8);
    xs.forEach((x, i) => { if (i % step === 0) s += `<text x="${X(i)}" y="${h - 8}" text-anchor="middle">${esc(x)}</text>`; });
    series.forEach((se, k) => {
      const c = se.color || COLORS[k];
      s += `<polyline fill="none" stroke="${c}" stroke-width="2.4" points="${se.values.map((v, i) => `${X(i)},${Y(v)}`).join(" ")}"/>`;
      s += se.values.map((v, i) => `<circle cx="${X(i)}" cy="${Y(v)}" r="2.6" fill="${c}"><title>${esc(se.name)} @ ${esc(xs[i])}: ${format(v)}</title></circle>`).join("");
    });
    svg(el, w, h, s);
    if (series.length > 1 || series[0].name)
      el.insertAdjacentHTML("beforeend", `<div class="legend">${series.map((se, k) => `<span><i style="background:${se.color || COLORS[k]}"></i>${esc(se.name)}</span>`).join("")}</div>`);
  }

  /* correlation heatmap */
  function heatmap(el, labels, M) {
    const n = labels.length, cell = 56, lw = 110, w = lw + n * cell, h = 24 + n * cell + 6;
    let s = "";
    labels.forEach((l, j) => { s += `<text x="${lw + j * cell + cell / 2}" y="14" text-anchor="middle">${esc(l.replace("days_since_posted", "days_online").slice(0, 10))}</text>`; });
    M.forEach((row, i) => {
      s += `<text x="${lw - 6}" y="${24 + i * cell + cell / 2 + 4}" text-anchor="end">${esc(labels[i].replace("days_since_posted", "days_online"))}</text>`;
      row.forEach((v, j) => {
        const a = Math.min(1, Math.abs(v)), col = v >= 0 ? `rgba(124,92,255,${a})` : `rgba(248,113,113,${a})`;
        s += `<rect x="${lw + j * cell}" y="${24 + i * cell}" width="${cell - 2}" height="${cell - 2}" rx="6" fill="${col}"/><text x="${lw + j * cell + cell / 2 - 1}" y="${24 + i * cell + cell / 2 + 4}" text-anchor="middle" class="t">${v.toFixed(2)}</text>`;
      });
    });
    svg(el, w, h, s);
  }

  return { hbars, vbars, lines, heatmap, fmt, COLORS };
})();
