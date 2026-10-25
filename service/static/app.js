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
      c: blocks[name] ? blocks[name].contribution : 0,
      v: blocks[name] ? blocks[name].value : 0,
      imp: blocks[name] ? blocks[name].imputed : false,
    })).filter((it) => blocks[it.name]);
    items.sort((a, b) => b.c - a.c);
    const chips = [];
    for (const it of items) {
      if (chips.length >= 3) break;
      if (it.c <= 0) continue;
      if (it.imp) {
        chips.push({ text: it.name + " imputed", imputed: true });
      } else if (it.name === "category" && it.v >= 0.99) {
        chips.push({ text: "same category", imputed: false });
      } else if (it.name === "audience" && it.v >= 0.99) {
        chips.push({ text: "audience match", imputed: false });
      } else {
        const label = it.name === "text" ? "title" : it.name;
        chips.push({ text: label + " " + it.v.toFixed(2), imputed: false });
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

  function renderStatus(meta) {
    const strip = el("status-strip");
    strip.replaceChildren();
    const line = document.createElement("p");
    line.className = "status";
    const ms = typeof meta.latency_ms === "number" ? Math.round(meta.latency_ms) + " ms" : "—";
    const cached = meta.cache_hit === true ? " (cached)" : "";
    line.textContent =
      "Scored " + meta.candidates_scored + " products · " +
      meta.title_groups_total + " title groups · " +
      meta.hidden_query_group_count + " colorway variants of this product hidden · " +
      ms + cached + " · " + meta.mode + " mode · " + meta.transform;
    strip.appendChild(line);
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
      const btn = document.createElement("div");
      btn.className = "result";
      btn.setAttribute("role", "button");
      btn.tabIndex = 0;
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
        chip.className = t.imputed ? "chip imputed" : "chip";
        chip.textContent = t.text;
        btn.appendChild(chip);
      });
      btn.addEventListener("click", () => openDrawer(card));
      btn.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          openDrawer(card);
        }
      });
      const cmp = document.createElement("button");
      cmp.type = "button";
      cmp.className = "compare-btn";
      cmp.textContent = "Compare";
      cmp.addEventListener("click", (ev) => {
        ev.stopPropagation();
        selectedResult = card;
        renderCompare();
        el("compare-panel").scrollIntoView({ behavior: "smooth", block: "start" });
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
      note.textContent = "✓ matches server ranking";
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

  const MISSING = "not in eval.json";
  const CATEGORICAL = { brand: true, category: true, audience: true };

  function dig(obj, path) {
    let cur = obj;
    for (let i = 0; i < path.length; i++) {
      if (cur == null || typeof cur !== "object" || !(path[i] in cur)) return undefined;
      cur = cur[path[i]];
    }
    return cur;
  }

  function fmtCi(ci) {
    if (!ci || typeof ci.mean !== "number" || typeof ci.lo !== "number" || typeof ci.hi !== "number") {
      return null;
    }
    return ci.mean.toFixed(3) + " [" + ci.lo.toFixed(3) + ", " + ci.hi.toFixed(3) + "]";
  }

  function ciZeroSentence(ci) {
    if (!ci || typeof ci.lo !== "number" || typeof ci.hi !== "number") return MISSING;
    return (ci.lo > 0 || ci.hi < 0) ? "CI excludes 0" : "CI includes 0";
  }

  function fmtPaired(ci) {
    const base = fmtCi(ci);
    if (!base) return MISSING;
    return base + " · " + ciZeroSentence(ci);
  }

  function fmt3(v) {
    return typeof v === "number" ? v.toFixed(3) : MISSING;
  }

  function presetMetric(ci) {
    const formatted = fmtCi(ci);
    return formatted ? " " + formatted : " " + MISSING;
  }

  function applyPresetLabels() {
    const shipped = el("preset-shipped");
    const text = el("preset-text");
    if (shipped) {
      const ci = dig(evalData, ["test_results", "fusion", "ndcg10"]);
      shipped.textContent = shipped.dataset.label + (evalData ? presetMetric(ci) : "");
    }
    if (text) {
      const ci = dig(evalData, ["test_results", "baselines", "minilm_only", "ndcg10"]);
      text.textContent = text.dataset.label + (evalData ? presetMetric(ci) : "");
    }
    ["noimage", "structured"].forEach((key) => {
      const btn = el("preset-" + key);
      if (!btn) return;
      btn.textContent = btn.dataset.label;
      btn.title = "not evaluated";
    });
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

  function sideText(block, card, sym) {
    if (block === "text") return card.name || "—";
    if (block === "image") return card.image_url ? "present" : "missing";
    if (block === "category") return card.child_category || "—";
    if (block === "audience") return card.audience || "—";
    if (block === "brand") return card.brand || "—";
    if (block === "color") return card.color_missing ? "missing" : "present";
    if (block === "price") return formatPrice(card.price, sym);
    if (block === "rating") return card.rating == null ? "—" : String(card.rating);
    if (block === "weight") return card.weight_g == null ? "—" : String(card.weight_g);
    return "—";
  }

  function blockReading(name, block) {
    if (!block || typeof block.value !== "number") return "—";
    const v = block.value;
    if (CATEGORICAL[name]) {
      if (block.imputed) return "imputed";
      if (name === "audience" && Math.abs(v - 0.5) < 1e-6) return "partial";
      if (v >= 0.99) return "match";
      return "mismatch";
    }
    if (block.imputed) return v.toFixed(3) + " · imputed";
    if (v >= 0.8) return v.toFixed(3) + " · similar";
    if (v >= 0.5) return v.toFixed(3) + " · somewhat";
    return v.toFixed(3) + " · different";
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
    const sym = exploreData.meta ? exploreData.meta.currency_symbol : "";
    const table = el("compare-table");
    table.replaceChildren();
    const head = document.createElement("tr");
    ["Block", "Query", "Result", "Value", "Reading"].forEach((h) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = h;
      head.appendChild(th);
    });
    table.appendChild(head);
    BLOCK_ORDER.forEach((name) => {
      const block = r.blocks ? r.blocks[name] : null;
      const tr = document.createElement("tr");
      const cells = [
        name,
        sideText(name, q, sym),
        sideText(name, r, sym),
        block && typeof block.value === "number" ? block.value.toFixed(3) : "—",
        blockReading(name, block),
      ];
      cells.forEach((text, i) => {
        const cell = document.createElement(i === 0 ? "th" : "td");
        if (i === 0) cell.scope = "row";
        cell.textContent = text;
        tr.appendChild(cell);
      });
      table.appendChild(tr);
    });
  }

  function countText(used, full) {
    const u = typeof used === "number" ? String(used) : MISSING;
    const f = typeof full === "number" ? String(full) : MISSING;
    return u + " / " + f;
  }

  function runLabel(data) {
    if (data.smoke === true) return "smoke";
    if (data.capped === true) return "capped run (not smoke)";
    if (data.smoke == null && data.capped == null) return MISSING;
    return "full eval";
  }

  function coverageText(data) {
    if (typeof data.image_coverage !== "number") return MISSING;
    return (data.image_coverage * 100).toFixed(1) + "%";
  }

  function addKvRow(table, label, value) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.scope = "row";
    th.textContent = label;
    const td = document.createElement("td");
    td.textContent = value;
    tr.appendChild(th);
    tr.appendChild(td);
    table.appendChild(tr);
  }

  function makeKvTable() {
    const table = document.createElement("table");
    table.className = "kv";
    return table;
  }

  function renderEvidenceMissing() {
    const host = el("evidence-body");
    host.replaceChildren();
    const p = document.createElement("p");
    p.textContent = "evaluation file not found";
    host.appendChild(p);
  }

  function renderEvidence() {
    const host = el("evidence-body");
    host.replaceChildren();
    if (!evalData) {
      renderEvidenceMissing();
      return;
    }
    const data = evalData;
    const split = data.split && typeof data.split === "object" ? data.split : null;
    const head = document.createElement("p");
    const tuneUsed = split ? split.tune_queries : data.tune_queries;
    const valUsed = split ? split.validation_queries : data.validation_queries;
    const testUsed = split ? split.test_queries : data.test_queries;
    const tuneFull = split ? split.tune_queries_full : undefined;
    const valFull = split ? split.validation_queries_full : undefined;
    const testFull = split ? split.test_queries_full : undefined;
    head.textContent =
      runLabel(data) +
      " · tune " + countText(tuneUsed, tuneFull) +
      ", val " + countText(valUsed, valFull) +
      ", test " + countText(testUsed, testFull) +
      " · image coverage " + coverageText(data);
    host.appendChild(head);

    const metrics = makeKvTable();
    const fusion = dig(data, ["test_results", "fusion"]);
    addKvRow(metrics, "Shipped fusion NDCG@10", fmtCi(fusion && fusion.ndcg10) || MISSING);
    addKvRow(metrics, "Shipped fusion hit@10", fmtCi(fusion && fusion.hit10) || MISSING);
    const mini = dig(data, ["test_results", "baselines", "minilm_only"]);
    const pop = dig(data, ["test_results", "baselines", "popularity"]);
    addKvRow(metrics, "MiniLM-only (text only) NDCG@10", fmtCi(mini && mini.ndcg10) || MISSING);
    addKvRow(metrics, "MiniLM-only (text only) hit@10", fmtCi(mini && mini.hit10) || MISSING);
    addKvRow(metrics, "Popularity NDCG@10", fmtCi(pop && pop.ndcg10) || MISSING);
    addKvRow(metrics, "Popularity hit@10", fmtCi(pop && pop.hit10) || MISSING);
    addKvRow(
      metrics,
      "Fusion − MiniLM, validation",
      fmtPaired(dig(data, ["paired_ndcg_fusion_minus_minilm", "validation"]))
    );
    addKvRow(
      metrics,
      "Fusion − MiniLM, test",
      fmtPaired(dig(data, ["paired_ndcg_fusion_minus_minilm", "test"]))
    );
    const shippedName = typeof data.shipped_transform === "string" ? data.shipped_transform : MISSING;
    addKvRow(
      metrics,
      "raw − " + shippedName,
      fmtPaired(dig(data, ["transform_selection_meta", "paired_vs_default", "raw"]))
    );
    addKvRow(
      metrics,
      "percentile − " + shippedName,
      fmtPaired(dig(data, ["transform_selection_meta", "paired_vs_default", "percentile"]))
    );
    const guards = dig(data, ["guardrails"]);
    let tuning = MISSING;
    if (guards && typeof guards === "object" && "accepted_tuned" in guards) {
      tuning = guards.accepted_tuned ? "accepted" : "rejected";
      if (!guards.accepted_tuned) {
        tuning += " · " + (typeof guards.rejection_reason === "string" && guards.rejection_reason
          ? guards.rejection_reason
          : MISSING);
      }
    }
    addKvRow(metrics, "Weight tuning", tuning);
    addKvRow(
      metrics,
      "Same-child share, default",
      fmt3(dig(data, ["guardrails", "validation_default", "same_child_share10"]))
    );
    addKvRow(
      metrics,
      "Same-child share, tuned candidate",
      fmt3(dig(data, ["guardrails", "validation_tuned_candidate", "same_child_share10"]))
    );
    addKvRow(
      metrics,
      "Variant A − binary, same-child (test)",
      fmtPaired(dig(data, [
        "variant_guardrails_binary_vs_A", "test", "same_child_share10", "paired_A_minus_binary"
      ]))
    );
    host.appendChild(metrics);

    const pairHead = document.createElement("p");
    pairHead.className = "section-label";
    pairHead.textContent = "Pair-active rates";
    host.appendChild(pairHead);
    const rates = data.pair_active_rates;
    const pairTable = makeKvTable();
    const headRow = document.createElement("tr");
    ["Block", "Share of random pairs where both have the attribute"].forEach((label) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = label;
      headRow.appendChild(th);
    });
    pairTable.appendChild(headRow);
    if (!rates || typeof rates !== "object") {
      addKvRow(pairTable, MISSING, MISSING);
    } else {
      const names = Object.keys(rates);
      const ordered = BLOCK_ORDER.filter((name) => names.indexOf(name) >= 0)
        .concat(names.filter((name) => BLOCK_ORDER.indexOf(name) < 0));
      if (!ordered.length) addKvRow(pairTable, MISSING, MISSING);
      ordered.forEach((name) => {
        const rate = rates[name];
        const shown = typeof rate === "number" ? (rate * 100).toFixed(1) + "%" : MISSING;
        addKvRow(pairTable, name, shown);
      });
    }
    host.appendChild(pairTable);
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
      applyPresetLabels();
    } catch (_) {
      evalData = null;
      renderEvidenceMissing();
      applyPresetLabels();
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
