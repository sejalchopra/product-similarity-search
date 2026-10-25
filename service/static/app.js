(function () {
  const BLOCK_ORDER = [
    "text", "image", "category", "audience", "brand", "color", "price", "rating", "weight"
  ];
  const PLACEHOLDER =
    "data:image/svg+xml," + encodeURIComponent(
      '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96">' +
      '<rect fill="%23d4d4d8" width="96" height="96"/>' +
      '<text x="48" y="52" text-anchor="middle" fill="%2371717a" font-size="11" font-family="sans-serif">No image</text></svg>'
    );

  let queryId = null;
  let exploreData = null;
  let evalData = null;
  let serverWeights = {};
  let sliderWeights = {};
  let maskBrand = false;
  let maskImage = false;
  let debounceTimer = null;
  let selectedResult = null;

  const el = (id) => document.getElementById(id);

  function setError(msg) {
    const e = el("error");
    if (msg) {
      e.textContent = msg;
      e.hidden = false;
    } else {
      e.hidden = true;
      e.textContent = "";
    }
  }

  function setSpinner(on) {
    el("spinner").classList.toggle("on", on);
  }

  function imgOnError(ev) {
    ev.target.onerror = null;
    ev.target.src = PLACEHOLDER;
  }

  function makeImg(url) {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = "";
    img.src = url || PLACEHOLDER;
    img.onerror = imgOnError;
    return img;
  }

  function formatPrice(p, sym) {
    if (p == null || p === "") return "—";
    return (sym || "₹") + Number(p).toFixed(2);
  }

  async function fetchJson(url) {
    const r = await fetch(url);
    if (!r.ok) {
      let detail = r.statusText;
      try {
        const j = await r.json();
        detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      } catch (_) { /* ignore */ }
      throw new Error(detail || "Request failed");
    }
    return r.json();
  }

  function topChips(blocks) {
    const items = BLOCK_ORDER.map((name) => ({
      name,
      c: blocks[name].contribution,
      v: blocks[name].value,
      imp: blocks[name].imputed,
    }));
    items.sort((a, b) => b.c - a.c);
    const chips = [];
    for (const it of items) {
      if (chips.length >= 3) break;
      if (it.c <= 0) continue;
      if (it.name === "category" && it.v >= 0.99 && !it.imp) {
        chips.push("same category");
      } else if (it.name === "audience" && it.v >= 0.99 && !it.imp) {
        chips.push("audience match");
      } else if (it.imp) {
        chips.push(it.name + " imputed");
      } else {
        const label = it.name === "text" ? "title" : it.name;
        chips.push(label + " " + it.v.toFixed(2));
      }
    }
    return chips;
  }

  function renderQueryCard(card, sym) {
    const host = el("query-card");
    host.replaceChildren();
    host.appendChild(makeImg(card.image_url));
    const body = document.createElement("div");
    const h2 = document.createElement("h2");
    h2.textContent = card.name;
    body.appendChild(h2);
    const meta = document.createElement("p");
    meta.className = "meta";
    meta.textContent = [
      card.brand || "—",
      formatPrice(card.price, sym),
      card.child_category,
      card.audience
    ].join(" · ");
    body.appendChild(meta);
    host.appendChild(body);
    el("query-section").hidden = false;
  }

  async function renderStatus(meta) {
    const strip = el("status-strip");
    strip.replaceChildren();
    const line = document.createElement("p");
    line.className = "status";
    line.textContent =
      "Scored " + meta.candidates_scored + " products · " +
      meta.title_groups_total + " title groups · " +
      meta.hidden_query_group_count + " colorway variants of this product hidden · " +
      Math.round(meta.latency_ms) + " ms · " + meta.mode + " mode · " + meta.transform;
    strip.appendChild(line);
    const exact = document.createElement("p");
    exact.className = "status ok";
    exact.textContent = "Explore always uses exact scoring.";
    strip.appendChild(exact);
    if (meta.hidden_query_group_ids && meta.hidden_query_group_ids.length) {
      const det = document.createElement("details");
      const sum = document.createElement("summary");
      sum.textContent = "Hidden variants";
      det.appendChild(sum);
      const ul = document.createElement("ul");
      let names = Object.create(null);
      try {
        const batch = await fetchJson(
          "/products?ids=" + meta.hidden_query_group_ids.map(encodeURIComponent).join(",")
        );
        batch.forEach((c) => { names[c.id] = c.name; });
      } catch (_) { /* ids only */ }
      meta.hidden_query_group_ids.forEach((id) => {
        const li = document.createElement("li");
        li.textContent = names[id] ? names[id] + " (" + id + ")" : id;
        ul.appendChild(li);
      });
      det.appendChild(ul);
      strip.appendChild(det);
    }
  }

  function fuseScore(item, weights) {
    let s = 0;
    BLOCK_ORDER.forEach((b) => {
      s += weights[b] * item.blocks[b].value;
    });
    return s;
  }

  function normalizeWeights(w) {
    const sum = BLOCK_ORDER.reduce((a, b) => a + w[b], 0);
    if (sum <= 0) return w;
    const out = {};
    BLOCK_ORDER.forEach((b) => { out[b] = w[b] / sum; });
    return out;
  }

  function sortResults(results, weights) {
    return results.slice().sort((a, b) => {
      const sa = Math.round(fuseScore(a, weights) * 1e6) / 1e6;
      const sb = Math.round(fuseScore(b, weights) * 1e6) / 1e6;
      if (sa !== sb) return sb - sa;
      const ra = a.rating == null ? -1 : a.rating;
      const rb = b.rating == null ? -1 : b.rating;
      if (ra !== rb) return rb - ra;
      const pa = a.price == null ? Infinity : a.price;
      const pb = b.price == null ? Infinity : b.price;
      if (pa !== pb) return pa - pb;
      return a.id.localeCompare(b.id);
    });
  }

  function weightsMatchServer() {
    return BLOCK_ORDER.every((b) =>
      Math.abs(sliderWeights[b] - serverWeights[b]) < 1e-6
    );
  }

  function renderResultsGrid(results, sym, flipFrom) {
    const grid = el("results-grid");
    const positions = new Map();
    if (flipFrom) {
      grid.querySelectorAll(".result").forEach((node) => {
        positions.set(node.dataset.id, node.getBoundingClientRect());
      });
    }
    grid.replaceChildren();
    results.forEach((card, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "result";
      btn.dataset.id = card.id;
      const rankEl = document.createElement("div");
      rankEl.className = "rank";
      const w = normalizeWeights(sliderWeights);
      const sc = fuseScore(card, w);
      rankEl.textContent = "#" + (i + 1) + " · " + sc.toFixed(3);
      btn.appendChild(rankEl);
      btn.appendChild(makeImg(card.image_url));
      const name = document.createElement("div");
      name.className = "name";
      name.textContent = card.name;
      btn.appendChild(name);
      const sub = document.createElement("div");
      sub.className = "meta";
      sub.textContent = (card.brand || "—") + " · " + formatPrice(card.price, sym);
      btn.appendChild(sub);
      topChips(card.blocks).forEach((t) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = t;
        btn.appendChild(chip);
      });
      btn.addEventListener("click", () => openDrawer(card));
      const cmp = document.createElement("button");
      cmp.type = "button";
      cmp.textContent = "Compare";
      cmp.addEventListener("click", (ev) => {
        ev.stopPropagation();
        selectedResult = card;
        renderCompare();
      });
      btn.appendChild(cmp);
      grid.appendChild(btn);
      if (flipFrom && positions.has(card.id)) {
        const first = positions.get(card.id);
        const last = btn.getBoundingClientRect();
        const dx = first.left - last.left;
        const dy = first.top - last.top;
        btn.style.transform = "translate(" + dx + "px," + dy + "px)";
        requestAnimationFrame(() => {
          btn.style.transform = "";
        });
      }
    });
    el("results-section").hidden = results.length === 0;
    const note = el("match-note");
    if (weightsMatchServer()) {
      note.textContent = "matches server ranking";
      note.className = "ok";
    } else {
      note.textContent = "Custom weights — only top-" + results.length + " can reorder";
      note.className = "note";
    }
  }

  function renderWeightSliders() {
    const host = el("weight-sliders");
    host.replaceChildren();
    BLOCK_ORDER.forEach((name) => {
      const lab = document.createElement("label");
      const span = document.createElement("span");
      span.textContent = name + " " + sliderWeights[name].toFixed(2);
      const input = document.createElement("input");
      input.type = "range";
      input.min = "0";
      input.max = "1";
      input.step = "0.01";
      input.value = String(sliderWeights[name]);
      input.addEventListener("input", () => {
        sliderWeights[name] = Number(input.value);
        span.textContent = name + " " + sliderWeights[name].toFixed(2);
        if (exploreData) {
          const sorted = sortResults(exploreData.results, normalizeWeights(sliderWeights));
          renderResultsGrid(sorted, exploreData.meta.currency_symbol, true);
        }
      });
      lab.appendChild(span);
      lab.appendChild(input);
      host.appendChild(lab);
    });
    const reset = document.createElement("button");
    reset.type = "button";
    reset.textContent = "Reset weights";
    reset.addEventListener("click", () => {
      sliderWeights = Object.assign({}, serverWeights);
      renderWeightSliders();
      if (exploreData) renderResultsGrid(exploreData.results, exploreData.meta.currency_symbol, true);
      syncHash();
    });
    host.appendChild(reset);
  }

  function applyPresets(name) {
    const base = Object.assign({}, serverWeights);
    if (name === "text") {
      BLOCK_ORDER.forEach((b) => { base[b] = b === "text" ? 1 : 0; });
    } else if (name === "noimage") {
      base.image = 0;
    } else if (name === "structured") {
      base.text = 0;
      base.image = 0;
    }
    sliderWeights = normalizeWeights(base);
    renderWeightSliders();
    if (exploreData) {
      renderResultsGrid(
        sortResults(exploreData.results, sliderWeights),
        exploreData.meta.currency_symbol,
        true
      );
    }
    syncHash();
  }

  function presetNdgcLabel(key) {
    if (!evalData) return "";
    if (key === "shipped" && evalData.shipped_transform && evalData.transform_experiment) {
      const row = evalData.transform_experiment[evalData.shipped_transform];
      if (row && row.ndcg10) {
        return " NDCG@10 " + row.ndcg10.mean.toFixed(3) +
          " [" + row.ndcg10.lo.toFixed(3) + "," + row.ndcg10.hi.toFixed(3) + "]";
      }
    }
    if (key === "text" && evalData.baselines && evalData.baselines.minilm_only) {
      const row = evalData.baselines.minilm_only.ndcg10;
      if (row) {
        return " NDCG@10 " + row.mean.toFixed(3) +
          " [" + row.lo.toFixed(3) + "," + row.hi.toFixed(3) + "]";
      }
    }
    return "";
  }

  function openDrawer(card) {
    const drawer = el("explain-drawer");
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    el("drawer-title").textContent = card.name;
    const w = normalizeWeights(sliderWeights);
    const sc = fuseScore(card, w);
    el("drawer-score").textContent = "Score " + sc.toFixed(6);
    const bars = el("drawer-bars");
    bars.replaceChildren();
    let math = "";
    BLOCK_ORDER.forEach((name) => {
      const b = card.blocks[name];
      const contrib = w[name] * b.value;
      math += w[name].toFixed(3) + "×" + b.value.toFixed(3) + " + ";
      const row = document.createElement("div");
      row.className = "bar-row";
      if (b.raw_sim != null) row.title = "raw: " + b.raw_sim;
      const label = document.createElement("div");
      label.className = "bar-label";
      const left = document.createElement("span");
      left.textContent = name + " w=" + w[name].toFixed(2);
      if (b.imputed) {
        const tag = document.createElement("span");
        tag.className = "tag-imputed";
        tag.textContent = b.imputed_reason || "imputed";
        left.appendChild(tag);
      }
      const right = document.createElement("span");
      right.textContent = b.value.toFixed(3) + " → " + contrib.toFixed(4);
      label.appendChild(left);
      label.appendChild(right);
      row.appendChild(label);
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill" + (b.imputed ? " imputed" : "");
      fill.style.width = Math.min(100, Math.max(0, b.value * 100)) + "%";
      track.appendChild(fill);
      row.appendChild(track);
      bars.appendChild(row);
    });
    el("drawer-math").textContent = math.slice(0, -3) + " = " + sc.toFixed(6);
    const masks = [maskBrand ? "brand" : "", maskImage ? "image" : ""].filter(Boolean);
    const curl =
      "curl -s '" + location.origin +
      "/explore?product_id=" + encodeURIComponent(queryId) +
      "&n=" + (exploreData ? exploreData.results.length : 10) +
      (masks.length ? "&mask=" + masks.join(",") : "") +
      "'";
    el("btn-copy-curl").onclick = () => navigator.clipboard.writeText(curl);
    el("raw-json").textContent = JSON.stringify(card, null, 2);
  }

  function closeDrawer() {
    el("explain-drawer").classList.remove("open");
    el("explain-drawer").setAttribute("aria-hidden", "true");
  }

  function renderCompare() {
    const panel = el("compare-panel");
    if (!exploreData || !selectedResult) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    const q = exploreData.query;
    const r = selectedResult;
    const rows = [
      ["brand", q.brand, r.brand],
      ["category", q.child_category, r.child_category],
      ["audience", q.audience, r.audience],
      ["price", q.price, r.price],
      ["rating", q.rating, r.rating],
      ["weight", q.weight_g, r.weight_g],
      ["image", q.image_url ? "present" : "missing", r.image_url ? "present" : "missing"],
      ["color", q.color_missing ? "missing" : "present", r.color_missing ? "missing" : "present"],
    ];
    const table = el("compare-table");
    table.replaceChildren();
    const head = document.createElement("tr");
    ["Field", "Query", "Result", ""].forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      head.appendChild(th);
    });
    table.appendChild(head);
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      [row[0], String(row[1] ?? "—"), String(row[2] ?? "—")].forEach((cell) => {
        const td = document.createElement("td");
        td.textContent = cell;
        tr.appendChild(td);
      });
      const mark = document.createElement("td");
      const a = row[1];
      const b = row[2];
      if (a == null && b == null) mark.textContent = "—";
      else if (a === b) mark.textContent = "match";
      else if (a == null || b == null) mark.textContent = "missing";
      else mark.textContent = "mismatch";
      tr.appendChild(mark);
      table.appendChild(tr);
    });
  }

  function renderEvidence() {
    const host = el("evidence-body");
    host.replaceChildren();
    if (!evalData) {
      host.textContent = "Load /eval failed or eval.json missing.";
      return;
    }
    const p = document.createElement("p");
    p.textContent = "generated by make eval · " + evalData.eval_queries + " held-out queries · image coverage " + evalData.image_coverage;
    host.appendChild(p);
    if (evalData.baselines) {
      Object.keys(evalData.baselines).forEach((name) => {
        const b = evalData.baselines[name];
        const line = document.createElement("p");
        line.textContent = name + " NDCG@10 " + b.ndcg10.mean.toFixed(3) +
          " [" + b.ndcg10.lo.toFixed(3) + "," + b.ndcg10.hi.toFixed(3) + "]";
        host.appendChild(line);
      });
    }
    if (evalData.pair_active_rates) {
      const pr = document.createElement("p");
      pr.textContent = "Pair-active: " + JSON.stringify(evalData.pair_active_rates);
      host.appendChild(pr);
    }
  }

  function syncHash() {
    if (!queryId) return;
    const n = el("num-similar").value;
    let h = "#id=" + encodeURIComponent(queryId) + "&n=" + n;
    if (!weightsMatchServer()) {
      h += "&w=" + encodeURIComponent(
        BLOCK_ORDER.map((b) => b + ":" + sliderWeights[b].toFixed(3)).join(",")
      );
    }
    location.hash = h;
  }

  function parseWeightHash(w) {
    if (!w) return null;
    const out = {};
    w.split(",").forEach((pair) => {
      const i = pair.indexOf(":");
      if (i < 0) return;
      const k = pair.slice(0, i);
      const v = Number(pair.slice(i + 1));
      if (BLOCK_ORDER.includes(k) && !Number.isNaN(v)) out[k] = v;
    });
    return Object.keys(out).length ? out : null;
  }

  function parseHash() {
    const raw = location.hash.replace(/^#/, "");
    if (!raw) return;
    const params = new URLSearchParams(raw);
    const id = params.get("id");
    const n = params.get("n");
    if (n) el("num-similar").value = n;
    const wh = parseWeightHash(params.get("w"));
    if (id) {
      selectProduct(id, false).then(() => {
        if (wh) {
          sliderWeights = wh;
          renderWeightSliders();
          if (exploreData) {
            renderResultsGrid(
              sortResults(exploreData.results, normalizeWeights(sliderWeights)),
              exploreData.meta.currency_symbol,
              false
            );
          }
        }
      });
    }
  }

  async function selectProduct(id, pushHash) {
    setError("");
    el("short-note").hidden = true;
    closeDrawer();
    setSpinner(true);
    try {
      queryId = id;
      const n = Number(el("num-similar").value);
      const masks = [maskBrand ? "brand" : "", maskImage ? "image" : ""].filter(Boolean);
      let url = "/explore?product_id=" + encodeURIComponent(id) + "&n=" + n;
      if (masks.length) url += "&mask=" + masks.join(",");
      exploreData = await fetchJson(url);
      serverWeights = exploreData.meta.weights;
      sliderWeights = Object.assign({}, serverWeights);
      renderQueryCard(exploreData.query, exploreData.meta.currency_symbol);
      await renderStatus(exploreData.meta);
      el("weights-section").hidden = false;
      renderWeightSliders();
      renderResultsGrid(exploreData.results, exploreData.meta.currency_symbol, false);
      el("mode-label").textContent = exploreData.meta.mode;
      el("transform-label").textContent = exploreData.meta.transform;
      if (pushHash !== false) syncHash();
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setSpinner(false);
    }
  }

  async function loadMeta() {
    try {
      const meta = await fetchJson("/meta");
      el("mode-label").textContent = meta.mode;
      el("transform-label").textContent = meta.transform;
      const imgNote = el("image-tuning-note");
      if (imgNote && meta.explore_scoring_note) {
        imgNote.textContent = meta.explore_scoring_note;
        imgNote.hidden = false;
      }
    } catch (_) {
      el("mode-label").textContent = "unknown";
    }
  }

  async function loadEval() {
    try {
      evalData = await fetchJson("/eval");
      renderEvidence();
      ["shipped", "text", "noimage", "structured"].forEach((key) => {
        const btn = el("preset-" + key);
        if (btn) btn.textContent = btn.dataset.label + presetNdgcLabel(key === "shipped" ? "shipped" : key === "text" ? "text" : "");
      });
    } catch (_) {
      evalData = null;
    }
  }

  async function runSearch(q) {
    const dd = el("dropdown");
    if (!q) {
      dd.classList.remove("open");
      dd.replaceChildren();
      return;
    }
    const items = await fetchJson("/products/search?q=" + encodeURIComponent(q) + "&limit=10");
    dd.replaceChildren();
    items.forEach((item) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = item.name + (item.brand ? " — " + item.brand : "");
      b.addEventListener("click", () => {
        dd.classList.remove("open");
        el("search").value = item.name;
        selectProduct(item.id);
      });
      dd.appendChild(b);
    });
    dd.classList.toggle("open", items.length > 0);
  }

  el("search").addEventListener("input", (ev) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => runSearch(ev.target.value.trim()), 250);
  });

  el("btn-search").addEventListener("click", () => {
    const q = el("search").value.trim();
    runSearch(q).then(() => {
      const first = el("dropdown").querySelector("button");
      if (first) first.click();
    });
  });

  el("btn-random").addEventListener("click", async () => {
    const card = await fetchJson("/products/random");
    el("search").value = card.name;
    await selectProduct(card.id);
  });

  el("num-similar").addEventListener("change", () => {
    if (queryId) selectProduct(queryId);
  });

  el("mask-brand").addEventListener("change", (ev) => {
    maskBrand = ev.target.checked;
    if (queryId) selectProduct(queryId);
  });
  el("mask-image").addEventListener("change", (ev) => {
    maskImage = ev.target.checked;
    if (queryId) selectProduct(queryId);
  });

  el("drawer-close").addEventListener("click", closeDrawer);
  el("explain-drawer").addEventListener("click", (ev) => {
    if (ev.target === el("explain-drawer")) closeDrawer();
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") closeDrawer();
  });
  document.addEventListener("click", (ev) => {
    if (!ev.target.closest(".search-wrap")) {
      el("dropdown").classList.remove("open");
    }
  });

  ["shipped", "text", "noimage", "structured"].forEach((key) => {
    const btn = el("preset-" + key);
    if (!btn) return;
    btn.addEventListener("click", () => {
      if (key === "shipped") {
        sliderWeights = Object.assign({}, serverWeights);
        renderWeightSliders();
        if (exploreData) renderResultsGrid(exploreData.results, exploreData.meta.currency_symbol, true);
      } else if (key === "text") applyPresets("text");
      else if (key === "noimage") applyPresets("noimage");
      else applyPresets("structured");
    });
  });

  loadMeta();
  loadEval();
  parseHash();
})();
