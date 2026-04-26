// Leon S. Kennedy tactical training console.

const QUOTES = [
  "Rookie mistake. Skip warm-ups, skip results.",
  "Hasta luego, plateau.",
  "Got a body to maintain. Mission first.",
  "What's holding you up? Get those reps.",
  "Time to put my training to the test.",
  "Tougher than I thought... so push harder.",
  "Saved your progress. Don't waste it.",
  "Stranger... got a workout for you.",
  "Game's not over. One more set.",
  "Where's everyone going? Bingo? Squat day."
];

// ------- routing -----------------------------------------------------------
const tabs = document.querySelectorAll(".tab");
const views = document.querySelectorAll(".view");
tabs.forEach(t => t.addEventListener("click", () => {
  tabs.forEach(x => x.classList.remove("active"));
  views.forEach(x => x.classList.remove("active"));
  t.classList.add("active");
  document.getElementById("view-" + t.dataset.view).classList.add("active");
  setStatus(`SWITCHED TO ${t.textContent}`);
  if (t.dataset.view === "overload") loadOverload();
  if (t.dataset.view === "supps") loadSupps();
}));

// ------- HUD ---------------------------------------------------------------
function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}
function rotateQuote() {
  const q = QUOTES[Math.floor(Math.random() * QUOTES.length)];
  document.getElementById("leon-quote").textContent = "« " + q + " »";
}
setInterval(rotateQuote, 12000); rotateQuote();

setInterval(() => {
  document.getElementById("clock").textContent = new Date().toLocaleTimeString("en-GB");
}, 1000);

// ------- API helpers -------------------------------------------------------
async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

function showError(container, err) {
  container.innerHTML = `<div class="err">⚠ ${err.message}</div>`;
  setStatus("ERROR");
}

// ------- INVENTORY: workouts ----------------------------------------------
async function loadWorkouts() {
  const grid = document.getElementById("workouts-grid");
  grid.innerHTML = `<div class="loading">DECRYPTING SAVE FILE...</div>`;
  try {
    const data = await api("/api/workouts");
    document.getElementById("match-count").textContent = data.matched;
    document.getElementById("total-count").textContent = data.total_in_account;
    if (!data.workouts.length) {
      grid.innerHTML = `<div class="loading">NO ENTRIES MATCH LABEL "${data.label}". Try a different LEON_LABEL.</div>`;
      return;
    }
    grid.innerHTML = "";
    data.workouts.forEach(w => grid.appendChild(workoutCard(w)));
    setStatus(`${data.matched} OPERATIONS LOADED`);
  } catch (e) { showError(grid, e); }
}

function workoutCard(w) {
  const card = document.createElement("div");
  card.className = "inv-card";
  const date = w.start_time ? new Date(w.start_time).toLocaleDateString() : "—";
  const duration = (w.start_time && w.end_time)
    ? Math.round((new Date(w.end_time) - new Date(w.start_time)) / 60000) + "m"
    : "—";
  let setCount = 0, volume = 0;
  (w.exercises || []).forEach(ex => (ex.sets || []).forEach(s => {
    if (s.weight_kg != null && s.reps != null) {
      setCount += 1;
      volume += s.weight_kg * s.reps;
    }
  }));
  const exList = (w.exercises || []).slice(0, 6).map(ex => {
    const sets = (ex.sets || []).filter(s => s.weight_kg != null);
    const top = sets.length
      ? sets.reduce((a, b) => (a.weight_kg * a.reps > b.weight_kg * b.reps ? a : b))
      : null;
    const right = top ? `${top.weight_kg}kg×${top.reps}` : `${(ex.sets || []).length}set`;
    return `<li><span>${ex.title}</span><span>${right}</span></li>`;
  }).join("");
  card.innerHTML = `
    <div class="stamp">v2</div>
    <div class="title">${escapeHtml(w.title || "UNTITLED OP")}</div>
    <div class="meta">${date} &middot; ${duration} &middot; ${setCount} SETS &middot; ${Math.round(volume)} kg</div>
    <ul>${exList}</ul>
  `;
  return card;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// ------- FILES: progressive overload + stats ------------------------------
async function loadOverload() {
  const tbody = document.querySelector("#overload-table tbody");
  tbody.innerHTML = `<tr><td colspan="6" class="loading">CALCULATING...</td></tr>`;
  const lookback = document.getElementById("lookback").value || 5;
  try {
    const [ov, st] = await Promise.all([
      api(`/api/overload?lookback=${lookback}`),
      api("/api/stats"),
    ]);
    if (!ov.projections.length) {
      tbody.innerHTML = `<tr><td colspan="6" class="loading">Need ≥2 sessions per exercise. Log more.</td></tr>`;
    } else {
      tbody.innerHTML = ov.projections.map(p => {
        const trendClass = p.trend_kg_per_session > 0 ? "color:var(--olive-bright)" : "color:var(--blood-bright)";
        const arrow = p.trend_kg_per_session > 0 ? "▲" : (p.trend_kg_per_session < 0 ? "▼" : "—");
        return `<tr>
          <td>${escapeHtml(p.title)}</td>
          <td>${p.last_top}</td>
          <td>${p.last_e1rm} kg</td>
          <td style="${trendClass}">${arrow} ${p.trend_kg_per_session} kg</td>
          <td>${p.projected_e1rm} kg</td>
          <td><strong>${p.suggested_next}</strong></td>
        </tr>`;
      }).join("");
    }

    // stat cells
    const s = st.stats;
    const cells = [
      { l: "WORKOUTS", v: s.workout_count },
      { l: "TOTAL VOLUME", v: s.total_volume_kg.toLocaleString(), u: "kg" },
      { l: "SETS", v: s.total_sets },
      { l: "REPS", v: s.total_reps },
      { l: "AVG DURATION", v: s.avg_duration_min, u: "min" },
      { l: "TRAINING DAYS", v: s.training_days },
      { l: "LAST 7 DAYS", v: s.last_7_days },
      { l: "LAST 30 DAYS", v: s.last_30_days },
    ];
    document.getElementById("stat-row").innerHTML = cells.map(c =>
      `<div class="stat-cell"><div class="label">${c.l}</div>
       <div class="value">${c.v}<span class="unit">${c.u || ""}</span></div></div>`
    ).join("");

    drawVolume(st.volume_timeline);
    drawBalance(st.muscle_balance);

    // PRs
    document.querySelector("#pr-table tbody").innerHTML = (s.prs || []).map(pr =>
      `<tr><td>${escapeHtml(pr.exercise)}</td><td>${pr.e1rm} kg</td>
       <td>${pr.weight}kg × ${pr.reps}</td>
       <td>${pr.ts ? new Date(pr.ts).toLocaleDateString() : "—"}</td></tr>`
    ).join("");

    setStatus("REPORT FILED");
  } catch (e) { showError(tbody, e); }
}

function drawVolume(series) {
  const c = document.getElementById("volume-chart");
  if (!c) return;
  const ctx = c.getContext("2d");
  const w = c.clientWidth, h = c.height;
  c.width = w;
  ctx.clearRect(0, 0, w, h);
  if (!series.length) {
    ctx.fillStyle = "#c9bf94"; ctx.font = "12px monospace";
    ctx.fillText("NO DATA", 12, 24);
    return;
  }
  const max = Math.max(...series.map(s => s.volume)) || 1;
  const barW = (w - 60) / series.length;
  // axes
  ctx.strokeStyle = "#4d5b35";
  ctx.beginPath(); ctx.moveTo(50, 10); ctx.lineTo(50, h - 24); ctx.lineTo(w - 8, h - 24); ctx.stroke();
  // bars
  series.forEach((s, i) => {
    const bh = ((s.volume / max) * (h - 40)) | 0;
    const x = 52 + i * barW;
    const y = h - 24 - bh;
    const grad = ctx.createLinearGradient(0, y, 0, y + bh);
    grad.addColorStop(0, "#c52a2a");
    grad.addColorStop(1, "#6b3a1a");
    ctx.fillStyle = grad;
    ctx.fillRect(x, y, barW - 4, bh);
    ctx.fillStyle = "#c9bf94";
    ctx.font = "9px monospace";
    ctx.save();
    ctx.translate(x + barW / 2, h - 8);
    ctx.rotate(-Math.PI / 6);
    ctx.fillText(s.week, -20, 0);
    ctx.restore();
  });
  // y-axis label
  ctx.fillStyle = "#8aa15a"; ctx.font = "10px monospace";
  ctx.fillText(Math.round(max).toLocaleString() + " kg", 4, 14);
  ctx.fillText("0", 38, h - 26);
}

function drawBalance(rows) {
  const total = rows.reduce((a, b) => a + b.volume, 0) || 1;
  document.getElementById("balance-bars").innerHTML = rows.map(r => {
    const pct = (r.volume / total) * 100;
    return `<div class="bar-row">
      <div class="name">${r.group}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(1)}%"></div></div>
      <div class="val">${Math.round(r.volume).toLocaleString()} kg</div>
    </div>`;
  }).join("");
}

// ------- CASE: supplements -------------------------------------------------
const SUPPS = [
  { field: "creatine_g", label: "CREATINE", unit: "g", presets: [1, 2.5, 5] },
  { field: "protein_g", label: "PROTEIN", unit: "g", presets: [25, 40, 60] },
  { field: "electrolytes_mg_sodium", label: "SODIUM", unit: "mg", presets: [200, 500, 1000] },
  { field: "electrolytes_mg_potassium", label: "POTASSIUM", unit: "mg", presets: [200, 500, 1000] },
  { field: "water_ml", label: "WATER", unit: "ml", presets: [250, 500, 750] },
];

async function loadSupps() {
  const grid = document.getElementById("case-grid");
  grid.innerHTML = `<div class="loading">UNPACKING CASE...</div>`;
  try {
    const data = await api("/api/supplements");
    renderSupps(data);
  } catch (e) { showError(grid, e); }
}

function renderSupps(data) {
  const grid = document.getElementById("case-grid");
  document.getElementById("streak-pill").textContent =
    `CREATINE STREAK: ${data.streaks.creatine} DAY${data.streaks.creatine === 1 ? "" : "S"}`;
  grid.innerHTML = "";
  SUPPS.forEach(s => {
    const cur = data.today[s.field] || 0;
    const target = data.targets[s.field] || 1;
    const pct = Math.min(100, (cur / target) * 100);
    const hit = cur >= target;
    const card = document.createElement("div");
    card.className = "case-item" + (hit ? " hit" : "");
    card.innerHTML = `
      <h3>${s.label}</h3>
      <div class="case-current">${cur} <span style="font-size:14px;color:#c9bf94">${s.unit}</span></div>
      <div class="case-target">/ ${target} ${s.unit} ${hit ? "✓ TARGET MET" : ""}</div>
      <div class="case-bar"><div style="width:${pct}%"></div></div>
      <div class="case-actions">
        ${s.presets.map(p => `<button class="btn" data-add="${s.field}" data-amt="${p}">+${p}${s.unit}</button>`).join("")}
        <button class="btn-ghost" data-reset="${s.field}">RESET</button>
      </div>
    `;
    grid.appendChild(card);
  });
  // history
  document.querySelector("#supps-history tbody").innerHTML = data.history.map(h => `
    <tr>
      <td>${h.date}</td>
      <td style="color:${h.creatine_hit ? "var(--olive-bright)" : "var(--khaki)"}">${h.creatine_g} g ${h.creatine_hit ? "✓" : ""}</td>
      <td style="color:${h.protein_hit ? "var(--olive-bright)" : "var(--khaki)"}">${h.protein_g} g ${h.protein_hit ? "✓" : ""}</td>
      <td>${h.water_ml} ml</td>
    </tr>`).join("");
}

document.addEventListener("click", async (e) => {
  const add = e.target.dataset.add;
  const reset = e.target.dataset.reset;
  if (add) {
    const amt = parseFloat(e.target.dataset.amt);
    const data = await api("/api/supplements/add", {
      method: "POST",
      body: JSON.stringify({ field: add, amount: amt }),
    });
    renderSupps(data);
    setStatus(`+${amt} ${add}`);
  } else if (reset) {
    const data = await api("/api/supplements/reset", {
      method: "POST",
      body: JSON.stringify({ field: reset }),
    });
    renderSupps(data);
  }
});

// ------- TIMER (countdown) -------------------------------------------------
let timerEnd = null, timerRemaining = 120, timerInt = null;
function fmt(sec) {
  sec = Math.max(0, Math.ceil(sec));
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
function renderTimer() {
  const el = document.getElementById("timer-display");
  let remaining = timerRemaining;
  if (timerEnd) remaining = (timerEnd - Date.now()) / 1000;
  el.textContent = fmt(remaining);
  el.classList.toggle("warn", remaining > 0 && remaining <= 5);
  if (remaining <= 0 && timerEnd) {
    el.classList.add("done");
    el.classList.remove("warn");
    clearInterval(timerInt);
    timerInt = null; timerEnd = null;
    timerRemaining = 0;
    beep();
    setStatus("REST OVER. NEXT SET.");
  }
}
function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    [880, 660, 880].forEach((f, i) => {
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.frequency.value = f; o.type = "square";
      g.gain.value = 0.05;
      o.connect(g); g.connect(ctx.destination);
      o.start(ctx.currentTime + i * 0.18);
      o.stop(ctx.currentTime + i * 0.18 + 0.15);
    });
  } catch (_) {}
}
document.querySelectorAll(".preset").forEach(b =>
  b.addEventListener("click", () => {
    timerRemaining = parseInt(b.dataset.preset);
    timerEnd = null;
    document.getElementById("timer-display").classList.remove("done", "warn");
    renderTimer();
  }));
document.getElementById("timer-start").onclick = () => {
  if (timerRemaining <= 0) timerRemaining = 120;
  timerEnd = Date.now() + timerRemaining * 1000;
  document.getElementById("timer-display").classList.remove("done");
  if (timerInt) clearInterval(timerInt);
  timerInt = setInterval(renderTimer, 250);
  renderTimer();
};
document.getElementById("timer-pause").onclick = () => {
  if (!timerEnd) return;
  timerRemaining = (timerEnd - Date.now()) / 1000;
  timerEnd = null;
  if (timerInt) { clearInterval(timerInt); timerInt = null; }
};
document.getElementById("timer-reset").onclick = () => {
  timerEnd = null; timerRemaining = 120;
  if (timerInt) { clearInterval(timerInt); timerInt = null; }
  document.getElementById("timer-display").classList.remove("done", "warn");
  renderTimer();
};
document.getElementById("timer-set-custom").onclick = () => {
  timerRemaining = parseInt(document.getElementById("timer-custom").value) || 120;
  timerEnd = null;
  renderTimer();
};
renderTimer();

// ------- STOPWATCH ---------------------------------------------------------
let swStart = null, swElapsed = 0, swInt = null, lapNum = 0, lastLap = 0;
function fmtSw(ms) {
  const totalCs = Math.floor(ms / 100);
  const m = Math.floor(totalCs / 600);
  const s = Math.floor((totalCs / 10) % 60);
  const cs = totalCs % 10;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${cs}`;
}
function renderSw() {
  const ms = swStart ? swElapsed + (Date.now() - swStart) : swElapsed;
  document.getElementById("stopwatch-display").textContent = fmtSw(ms);
}
document.getElementById("sw-start").onclick = () => {
  if (swStart) {
    swElapsed += Date.now() - swStart;
    swStart = null;
    if (swInt) { clearInterval(swInt); swInt = null; }
    document.getElementById("sw-start").textContent = "START";
  } else {
    swStart = Date.now();
    swInt = setInterval(renderSw, 100);
    document.getElementById("sw-start").textContent = "STOP";
  }
};
document.getElementById("sw-lap").onclick = () => {
  const ms = swStart ? swElapsed + (Date.now() - swStart) : swElapsed;
  const split = ms - lastLap;
  lastLap = ms;
  lapNum += 1;
  const li = document.createElement("li");
  li.innerHTML = `<span>SET ${String(lapNum).padStart(2, "0")}</span><span>${fmtSw(split)} (total ${fmtSw(ms)})</span>`;
  document.getElementById("lap-list").prepend(li);
};
document.getElementById("sw-reset").onclick = () => {
  swStart = null; swElapsed = 0; lapNum = 0; lastLap = 0;
  if (swInt) { clearInterval(swInt); swInt = null; }
  document.getElementById("sw-start").textContent = "START";
  document.getElementById("lap-list").innerHTML = "";
  renderSw();
};
renderSw();

document.getElementById("refresh-btn").onclick = loadWorkouts;
document.getElementById("overload-refresh").onclick = loadOverload;

// ------- boot --------------------------------------------------------------
loadWorkouts();
