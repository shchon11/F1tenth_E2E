/* Leaderboard interaction: cohort switch, search, sort, per-row details.
   All dynamic text is written with textContent -- nothing from the data ever becomes markup. */
(function () {
  "use strict";

  var model = JSON.parse(document.getElementById("leaderboard-data").textContent);
  var metrics = model.metrics || [];
  var cohorts = model.cohorts || [];
  var byId = {};
  cohorts.forEach(function (c) { byId[c.id] = c; });

  var state = {
    cohort: byId[model.default_cohort] ? model.default_cohort : (cohorts[0] || {}).id,
    query: "",
    sort: model.default_rank_metric,
    order: "best",
    open: {}
  };

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) { n.className = cls; }
    if (text !== undefined && text !== null) { n.textContent = String(text); }
    return n;
  }
  function metricSpec(key) {
    for (var i = 0; i < metrics.length; i++) { if (metrics[i].key === key) { return metrics[i]; } }
    return metrics[0];
  }
  function cohort() { return byId[state.cohort]; }
  /* Display strings are built once, in Python, and reused verbatim: JavaScript's toFixed and
     Python's format disagree on an exact half (36/64 is 56.25%), and the two views must not print
     the same measurement differently. Nothing here re-rounds a value. */
  function display(spec, cell) {
    if (cell.value === null || cell.value === undefined) { return "N/A"; }
    return cell.display;
  }
  function short(sha) { return sha ? String(sha).slice(0, 12) : "none"; }

  /* ------------------------------------------------------------------ controls */

  function buildCohortOptions() {
    var box = document.getElementById("cohort-options");
    box.textContent = "";
    cohorts.forEach(function (c) {
      var id = "cohort-" + c.id;
      var label = el("label");
      var radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "cohort";
      radio.value = c.id;
      radio.id = id;
      radio.checked = c.id === state.cohort;
      radio.addEventListener("change", function () {
        if (!radio.checked) { return; }
        state.cohort = c.id;
        state.open = {};
        renderCohort();
      });
      label.appendChild(radio);
      label.appendChild(el("span", null, c.label));
      label.appendChild(el("span", "count", c.n_systems + " systems"));
      box.appendChild(label);
    });
  }

  function buildRankSelect() {
    var sel = document.getElementById("rank-metric");
    sel.textContent = "";
    metrics.forEach(function (m) {
      var o = document.createElement("option");
      o.value = m.key;
      o.textContent = m.label + (m.better === "up" ? " (higher is better)" : " (lower is better)");
      o.selected = m.key === state.sort;
      sel.appendChild(o);
    });
    sel.addEventListener("change", function () { state.sort = sel.value; renderTable(); });
    var dir = document.getElementById("direction");
    dir.value = state.order;
    dir.addEventListener("change", function () { state.order = dir.value; renderTable(); });
    var q = document.getElementById("q");
    q.addEventListener("input", function () { state.query = q.value; renderTable(); });
    document.getElementById("controls").addEventListener("submit", function (e) {
      e.preventDefault();
    });
  }

  function renderStrip() {
    var c = cohort();
    var strip = document.getElementById("strip");
    strip.textContent = "";
    var trials = [];
    Object.keys(c.expected_trials).sort().forEach(function (k) {
      trials.push(k + " " + c.expected_trials[k]);
    });
    var items = [
      ["Systems", String(c.n_systems), false],
      ["Scenario cells", String(c.scenario_cells) + " per system", false],
      ["Trials per system", String(c.trials_per_system) + " (" + trials.join(" · ") + ")", false],
      ["Friction", c.suite.solo_mus.length + " levels, fixed per episode", false]
    ];
    items.forEach(function (it) {
      var li = el("li");
      li.appendChild(el("span", "k", it[0]));
      li.appendChild(el("span", it[2] ? "v mono" : "v", it[1]));
      strip.appendChild(li);
    });
    document.getElementById("cohort-desc").textContent = c.description || "";
  }

  /* ------------------------------------------------------------------ table */

  function columnCount() { return 3 + metrics.length; }

  function buildHead() {
    var row = document.getElementById("head-row");
    row.textContent = "";
    var spec = metricSpec(state.sort);
    var rank = el("th", "col-rank", "Rank");
    rank.scope = "col";
    rank.title = "Competition rank by " + spec.label;
    row.appendChild(rank);

    var sys = el("th", "col-system");
    sys.scope = "col";
    sys.textContent = "Checkpoint";
    row.appendChild(sys);

    metrics.forEach(function (m) {
      var th = el("th", "metric");
      th.scope = "col";
      var active = m.key === state.sort;
      if (active) {
        var ascending = m.better === "up" ? state.order === "worst" : state.order === "best";
        th.setAttribute("aria-sort", ascending ? "ascending" : "descending");
      }
      var b = document.createElement("button");
      b.type = "button";
      var dirWords = m.better === "up" ? "higher is better" : "lower is better";
      b.title = m.label + " — " + dirWords + ". " + m.description + " Click to sort; click again " +
        "to reverse.";
      var label = el("span", "label", m.label + " ");
      label.appendChild(el("span", "arrow", m.better === "up" ? "↑" : "↓"));
      b.appendChild(label);
      b.appendChild(el("span", "sr", dirWords));
      if (active) {
        b.appendChild(el("span", "sortmark", state.order === "best" ? " ▼" : " ▲"));
        b.appendChild(el("span", "sr", state.order === "best" ? "best first" : "worst first"));
      }
      b.addEventListener("click", function () {
        if (state.sort === m.key) {
          state.order = state.order === "best" ? "worst" : "best";
        } else {
          state.sort = m.key;
          state.order = "best";
        }
        document.getElementById("rank-metric").value = state.sort;
        document.getElementById("direction").value = state.order;
        renderTable();
      });
      th.appendChild(b);
      row.appendChild(th);
    });

    var more = el("th", "col-more");
    more.scope = "col";
    more.textContent = "Details";
    row.appendChild(more);
  }

  function matches(s, q) {
    if (!q) { return true; }
    var hay = [s.system_id, s.name, s.training, s.note, s.controller_arm].join(" ").toLowerCase();
    return hay.indexOf(q) !== -1;
  }

  function orderedIds() {
    var c = cohort();
    var order = (c.order || {})[state.sort] || c.systems.map(function (s) { return s.system_id; });
    var sys = {};
    c.systems.forEach(function (s) { sys[s.system_id] = s; });
    var ranked = [], missing = [];
    order.forEach(function (sid) {
      var cell = sys[sid].metrics[state.sort];
      (cell && cell.value !== null && cell.value !== undefined ? ranked : missing).push(sid);
    });
    if (state.order === "worst") { ranked.reverse(); }
    return ranked.concat(missing);          // unmeasured rows stay last in both directions
  }

  function detailPanel(s) {
    var c = cohort();
    var panel = el("div", "panel");
    panel.appendChild(el("h3", null, s.name));
    var dl = el("dl", "kv");
    function put(k, v, mono) {
      dl.appendChild(el("dt", null, k));
      dl.appendChild(el("dd", mono ? "mono" : null, v));
    }
    put("System id", s.system_id, true);
    put("Training", s.training);
    if (s.note) { put("Note", s.note); }
    put("Controller", s.controller_arm + (s.cross_runtime
      ? " (declared cross-runtime: legacy-trained weights under this controller)" : ""));
    put("Checkpoint sha256", s.checkpoint_sha256, true);
    put("Estimator sha256", s.estimator_sha256 || "none", true);
    if (s.roster_note) { put("Roster note", s.roster_note); }
    put("Suite pin", c.suite.version + " · freeze " + c.suite.freeze_sha256, true);
    put("Benchmark source digest", c.source_digest_sha256, true);
    put("Solo completion", s.counts.solo[0] + " / " + s.counts.solo[1] + " trials");
    put("Low-mu completion", s.counts.low_mu[0] + " / " + s.counts.low_mu[1] +
      " trials at mu=" + c.suite.low_mu);
    put("Avoidance", s.counts.avoidance[0] + " / " + s.counts.avoidance[1] + " approaches");
    put("Overtaking", s.counts.overtaking[0] + " / " + s.counts.overtaking[1] + " races");
    if (s.distance_km !== null && s.distance_km !== undefined) {
      put("Distance travelled", s.distance_km + " km");
    }
    var pace = s.pace || {};
    put("Paired pace vs reference", s.is_reference
      ? "This is the reference system."
      : (pace.value === null || pace.value === undefined
        ? "N/A (" + (pace.reason || "unmeasured") + ")"
        : pace.display + " of solo trials completed by both this system and the reference. " +
          "Descriptive only: pace is not a rank metric and its sample differs for every pair."));
    var fails = Object.keys(s.failures || {});
    if (fails.length) {
      put("Recorded failure reasons", fails.sort().map(function (k) {
        return k + " " + s.failures[k];
      }).join(" · "));
    }
    put("Evidence", (c.evidence || []).map(function (e) {
      return e.path + " (" + e.n_cells + " cells)";
    }).join(" · "), true);
    panel.appendChild(dl);
    panel.appendChild(el("p", "note", c.suite.reused_maps_note));
    return panel;
  }

  function renderTable() {
    buildHead();
    var c = cohort();
    var body = document.getElementById("body");
    body.textContent = "";
    var spec = metricSpec(state.sort);
    var q = state.query.trim().toLowerCase();
    var sys = {};
    c.systems.forEach(function (s) { sys[s.system_id] = s; });
    var ids = orderedIds().filter(function (sid) { return matches(sys[sid], q); });

    var caption = document.getElementById("table-caption");
    caption.textContent = c.label + " — " + ids.length + " of " + c.n_systems +
      " checkpoints shown, ranked by " + spec.label +
      (spec.better === "up" ? " (higher is better)" : " (lower is better)") + ".";
    document.getElementById("filter-note").textContent =
      "Search filters rows only: ranks are competition ranks over all " + c.n_systems +
      " checkpoints in this cohort and never change.";

    if (!ids.length) {
      var tr = el("tr"), td = el("td");
      td.colSpan = columnCount();
      var box = el("div", "empty");
      box.appendChild(el("strong", null, q
        ? "No checkpoint in this cohort matches that search."
        : "This cohort has no checkpoints to show."));
      box.appendChild(el("span", null, q
        ? "Searching matches the name, system id, training recipe and controller arm."
        : "Nothing was validated for this cohort."));
      if (q) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = "Clear search";
        btn.addEventListener("click", function () {
          state.query = "";
          document.getElementById("q").value = "";
          renderTable();
          document.getElementById("q").focus();
        });
        box.appendChild(btn);
      }
      td.appendChild(box);
      tr.appendChild(td);
      body.appendChild(tr);
      return;
    }

    ids.forEach(function (sid) {
      var s = sys[sid];
      var tr = el("tr", "system-row");

      var rankTd = el("td", "col-rank");
      var rank = s.metrics[state.sort].rank;
      rankTd.appendChild(el("span", rank === null || rank === undefined ? "rank none" : "rank",
        rank === null || rank === undefined ? "—" : String(rank)));
      tr.appendChild(rankTd);

      var nameTd = el("td", "col-system");
      var name = el("span", "name", s.name);
      nameTd.appendChild(name);
      if (s.is_reference) { nameTd.appendChild(el("span", "badge", "reference")); }
      nameTd.appendChild(el("span", "train", s.training));
      tr.appendChild(nameTd);

      metrics.forEach(function (m) {
        var cell = s.metrics[m.key];
        var td = el("td", "metric" + (cell.rank === 1 ? " best" : "") +
          (cell.value === null || cell.value === undefined ? " na" : ""));
        td.appendChild(el("span", "v", display(m, cell)));
        if (cell.value === null || cell.value === undefined) {
          td.appendChild(el("span", "reason", cell.reason || "unmeasured"));
        } else if (m.kind === "pct") {
          td.appendChild(el("span", "x", cell.counts));
        }
        tr.appendChild(td);
      });

      var moreTd = el("td", "col-more");
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "more";
      var panelId = "details-" + c.id + "-" + sid.replace(/[^A-Za-z0-9_-]/g, "_");
      btn.setAttribute("aria-controls", panelId);
      btn.setAttribute("aria-expanded", state.open[sid] ? "true" : "false");
      btn.textContent = state.open[sid] ? "Hide" : "Details";
      moreTd.appendChild(btn);
      tr.appendChild(moreTd);
      body.appendChild(tr);

      var dtr = el("tr", "details-row");
      dtr.id = panelId;
      var dtd = el("td");
      dtd.colSpan = columnCount();
      dtd.appendChild(detailPanel(s));
      dtr.appendChild(dtd);
      dtr.hidden = !state.open[sid];
      body.appendChild(dtr);

      btn.addEventListener("click", function () {
        state.open[sid] = !state.open[sid];
        btn.setAttribute("aria-expanded", state.open[sid] ? "true" : "false");
        btn.textContent = state.open[sid] ? "Hide" : "Details";
        dtr.hidden = !state.open[sid];
      });
    });
  }

  /* ------------------------------------------------------------------ static prose */

  function renderMethod() {
    var ol = document.getElementById("method");
    ol.textContent = "";
    (model.method || []).forEach(function (t) { ol.appendChild(el("li", null, t)); });
    var ul = document.getElementById("limits");
    ul.textContent = "";
    (model.limitations || []).forEach(function (t) { ul.appendChild(el("li", null, t)); });
    document.getElementById("regen").textContent = model.command || "";
  }

  function renderEvidence() {
    var box = document.getElementById("evidence");
    box.textContent = "";
    cohorts.forEach(function (c) {
      box.appendChild(el("h3", null, c.label));
      var dl = el("dl", "kv");
      [["Protocol", c.n_systems + " systems · " + c.scenario_cells + " scenario cells per system · "
        + c.trials_per_system + " trials per system · " + c.n_cells + " system-cells and "
        + c.n_trials + " trial outcomes in total"],
       ["Solo friction levels", c.suite.solo_mus.join(" · ") + " (fixed for the whole episode; "
        + "low-mu completion is measured at " + c.suite.low_mu + ")"],
       ["Suite pin", c.suite.version + " · freeze " + c.suite.freeze_sha256],
       ["Benchmark source digest", c.source_digest_sha256],
       ["Reference system for paired pace", c.reference_system]].forEach(function (kv) {
        dl.appendChild(el("dt", null, kv[0]));
        dl.appendChild(el("dd", null, kv[1]));
      });
      box.appendChild(dl);
      var ul = el("ul");
      (c.evidence || []).forEach(function (e) {
        var li = el("li");
        var a = document.createElement("a");
        a.className = "evfile";
        a.href = e.href || e.path;         /* a relative path from this file, never a URL scheme */
        a.textContent = e.path;
        li.appendChild(a);
        li.appendChild(el("span", "evmeta", " — " + e.n_cells + " cells, sha256 " +
          e.sha256));
        ul.appendChild(li);
      });
      box.appendChild(ul);
    });
  }

  function renderCohort() {
    renderStrip();
    renderTable();
  }

  if (!cohorts.length) {
    document.getElementById("board").appendChild(
      el("p", "note warn", "This report contains no validated cohort."));
    return;
  }
  buildCohortOptions();
  buildRankSelect();
  renderMethod();
  renderEvidence();
  renderCohort();
})();
