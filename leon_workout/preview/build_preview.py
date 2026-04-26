"""Bake the CSS into a single self-contained preview.html with mock data.

Run from repo root:
    python leon_workout/preview/build_preview.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSS = (ROOT / "static" / "css" / "leon.css").read_text()
OUT = Path(__file__).parent / "index.html"

MOCK = {
    "workouts": [
        {"title": "Push Day v2", "date": "2026-04-24", "dur": "62m", "sets": 18, "vol": 6420,
         "items": [("Bench Press", "85kg×5"), ("OHP", "55kg×6"), ("Incline DB", "30kg×8"),
                   ("Tricep Pushdown", "40kg×10"), ("Lateral Raise", "12kg×12")]},
        {"title": "Pull Day v2", "date": "2026-04-22", "dur": "58m", "sets": 16, "vol": 5780,
         "items": [("Deadlift", "140kg×3"), ("Pull-Up", "BW+10×6"), ("Barbell Row", "80kg×6"),
                   ("Face Pull", "25kg×15"), ("Bicep Curl", "16kg×10")]},
        {"title": "Legs Day v2", "date": "2026-04-20", "dur": "71m", "sets": 17, "vol": 8240,
         "items": [("Back Squat", "120kg×5"), ("RDL", "100kg×8"), ("Leg Press", "180kg×10"),
                   ("Hip Thrust", "120kg×8"), ("Calf Raise", "80kg×15")]},
        {"title": "Push Day v2", "date": "2026-04-17", "dur": "60m", "sets": 18, "vol": 6280,
         "items": [("Bench Press", "82.5kg×5"), ("OHP", "52.5kg×6"), ("Incline DB", "28kg×8"),
                   ("Tricep Pushdown", "37.5kg×10")]},
        {"title": "Pull Day v2", "date": "2026-04-15", "dur": "55m", "sets": 16, "vol": 5640,
         "items": [("Deadlift", "137.5kg×3"), ("Pull-Up", "BW+7.5×6"), ("Barbell Row", "77.5kg×6")]},
        {"title": "Legs Day v2", "date": "2026-04-13", "dur": "68m", "sets": 17, "vol": 8060,
         "items": [("Back Squat", "117.5kg×5"), ("RDL", "97.5kg×8"), ("Leg Press", "175kg×10")]},
    ],
    "overload": [
        ("Back Squat", "120kg × 5", "140.0 kg", "▲ 2.50 kg", "147.5 kg", "122.5kg × 5"),
        ("Bench Press", "85kg × 5", "99.2 kg", "▲ 1.25 kg", "103.7 kg", "86.25kg × 5"),
        ("Deadlift", "140kg × 3", "154.0 kg", "▲ 2.50 kg", "161.5 kg", "142.5kg × 3"),
        ("OHP", "55kg × 6", "66.0 kg", "▲ 0.75 kg", "68.2 kg", "55.0kg × 6"),
        ("Barbell Row", "80kg × 6", "96.0 kg", "▲ 0.50 kg", "97.5 kg", "80.0kg × 6"),
        ("Incline DB", "30kg × 8", "38.0 kg", "— 0.00 kg", "38.0 kg", "30.0kg × 8"),
        ("Tricep Pushdown", "40kg × 10", "53.3 kg", "▼ -0.30 kg", "52.7 kg", "40.0kg × 10"),
    ],
    "stats": [
        ("WORKOUTS", "32", ""), ("TOTAL VOLUME", "203,840", "kg"),
        ("SETS", "548", ""), ("REPS", "4,127", ""),
        ("AVG DURATION", "62.4", "min"), ("TRAINING DAYS", "29", ""),
        ("LAST 7 DAYS", "4", ""), ("LAST 30 DAYS", "14", ""),
    ],
    "prs": [
        ("Deadlift", "154.0 kg", "140kg × 3", "2026-04-22"),
        ("Back Squat", "140.0 kg", "120kg × 5", "2026-04-20"),
        ("Bench Press", "99.2 kg", "85kg × 5", "2026-04-24"),
        ("Hip Thrust", "144.0 kg", "120kg × 8", "2026-04-20"),
        ("Barbell Row", "96.0 kg", "80kg × 6", "2026-04-22"),
        ("Pull-Up", "82.0 kg", "BW+10kg × 6", "2026-04-22"),
        ("OHP", "66.0 kg", "55kg × 6", "2026-04-24"),
    ],
    "balance": [("PUSH", 48230), ("PULL", 51970), ("LEGS", 88240), ("CORE", 15400)],
    "volume_weeks": [
        ("W12", 14820), ("W13", 16200), ("W14", 15640), ("W15", 17880),
        ("W16", 18920), ("W17", 20480), ("W18", 19320), ("W19", 22140),
    ],
    "supps": [
        {"label": "CREATINE", "unit": "g", "cur": 5, "target": 5, "presets": [1, 2.5, 5]},
        {"label": "PROTEIN", "unit": "g", "cur": 142, "target": 160, "presets": [25, 40, 60]},
        {"label": "SODIUM", "unit": "mg", "cur": 1800, "target": 2000, "presets": [200, 500, 1000]},
        {"label": "POTASSIUM", "unit": "mg", "cur": 750, "target": 1000, "presets": [200, 500, 1000]},
        {"label": "WATER", "unit": "ml", "cur": 2250, "target": 3000, "presets": [250, 500, 750]},
    ],
    "supp_history": [
        ("2026-04-18", "5 g ✓", "165 g ✓", "2800 ml"),
        ("2026-04-19", "5 g ✓", "172 g ✓", "3100 ml"),
        ("2026-04-20", "5 g ✓", "158 g", "2900 ml"),
        ("2026-04-21", "5 g ✓", "164 g ✓", "3200 ml"),
        ("2026-04-22", "5 g ✓", "171 g ✓", "2700 ml"),
        ("2026-04-23", "5 g ✓", "159 g", "2400 ml"),
        ("2026-04-24", "5 g ✓", "142 g", "2250 ml"),
    ],
}


def workout_card(w):
    items = "".join(
        f'<li><span>{n}</span><span>{r}</span></li>' for n, r in w["items"][:6]
    )
    return f'''<div class="inv-card">
      <div class="stamp">v2</div>
      <div class="title">{w["title"]}</div>
      <div class="meta">{w["date"]} · {w["dur"]} · {w["sets"]} SETS · {w["vol"]} kg</div>
      <ul>{items}</ul>
    </div>'''


def overload_row(r):
    title, last, e1rm, trend, proj, nxt = r
    color = "var(--olive-bright)" if "▲" in trend else (
        "var(--blood-bright)" if "▼" in trend else "var(--khaki)")
    return (f'<tr><td>{title}</td><td>{last}</td><td>{e1rm}</td>'
            f'<td style="color:{color}">{trend}</td><td>{proj}</td>'
            f'<td><strong>{nxt}</strong></td></tr>')


def stat_cell(label, value, unit):
    return (f'<div class="stat-cell"><div class="label">{label}</div>'
            f'<div class="value">{value}<span class="unit">{unit}</span></div></div>')


def pr_row(p):
    ex, e1rm, top, date = p
    return f'<tr><td>{ex}</td><td>{e1rm}</td><td>{top}</td><td>{date}</td></tr>'


def bar_row(name, vol, total):
    pct = (vol / total) * 100
    return (f'<div class="bar-row"><div class="name">{name}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="val">{vol:,} kg</div></div>')


def supp_card(s):
    pct = min(100, (s["cur"] / s["target"]) * 100)
    hit = s["cur"] >= s["target"]
    presets = "".join(
        f'<button class="btn">+{p}{s["unit"]}</button>' for p in s["presets"]
    )
    return f'''<div class="case-item{' hit' if hit else ''}">
      <h3>{s["label"]}</h3>
      <div class="case-current">{s["cur"]} <span style="font-size:14px;color:#c9bf94">{s["unit"]}</span></div>
      <div class="case-target">/ {s["target"]} {s["unit"]} {'✓ TARGET MET' if hit else ''}</div>
      <div class="case-bar"><div style="width:{pct}%"></div></div>
      <div class="case-actions">{presets}<button class="btn-ghost">RESET</button></div>
    </div>'''


def main():
    workouts_html = "\n".join(workout_card(w) for w in MOCK["workouts"])
    overload_html = "\n".join(overload_row(r) for r in MOCK["overload"])
    stats_html = "\n".join(stat_cell(*s) for s in MOCK["stats"])
    prs_html = "\n".join(pr_row(p) for p in MOCK["prs"])
    balance_total = sum(v for _, v in MOCK["balance"])
    balance_html = "\n".join(bar_row(n, v, balance_total) for n, v in MOCK["balance"])
    supps_html = "\n".join(supp_card(s) for s in MOCK["supps"])
    history_html = "\n".join(
        f'<tr><td>{d}</td><td style="color:var(--olive-bright)">{c}</td>'
        f'<td>{p}</td><td>{w}</td></tr>'
        for d, c, p, w in MOCK["supp_history"]
    )
    weeks_js = ",".join(f'{{w:"{w}",v:{v}}}' for w, v in MOCK["volume_weeks"])

    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>LEON // Preview (mock data)</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="scanlines"></div>
  <div class="vignette"></div>

  <header class="hud">
    <div class="hud-left">
      <span class="brand">RPD · TACTICAL TRAINING</span>
      <span class="brand-sub">PREVIEW · MOCK DATA · NO API</span>
    </div>
    <nav class="hud-tabs">
      <button class="tab active" data-view="inventory">INVENTORY</button>
      <button class="tab" data-view="overload">FILES</button>
      <button class="tab" data-view="supps">CASE</button>
      <button class="tab" data-view="timer">TIMER</button>
      <button class="tab" data-view="map">MAP</button>
    </nav>
    <div class="hud-right">
      <span class="label-pill">LABEL: <code>v2</code></span>
      <span class="time" id="clock">--:--:--</span>
    </div>
  </header>

  <main>
    <section class="view active" id="view-inventory">
      <div class="panel">
        <div class="panel-header">
          <h2>INVENTORY — LOGGED OPERATIONS</h2>
          <div class="meter">6 / 32 ENTRIES</div>
        </div>
        <div class="grid-inv">{workouts_html}</div>
      </div>
    </section>

    <section class="view" id="view-overload">
      <div class="panel">
        <div class="panel-header"><h2>FIELD REPORT — PROGRESSIVE OVERLOAD</h2>
          <div class="meter">LOOKBACK 5</div></div>
        <table class="dossier">
          <thead><tr><th>EXERCISE</th><th>LAST TOP</th><th>e1RM</th>
            <th>TREND/SESSION</th><th>PROJECTED e1RM</th><th>NEXT TARGET</th></tr></thead>
          <tbody>{overload_html}</tbody>
        </table>

        <div class="panel-header"><h2>STATISTICS DOSSIER</h2></div>
        <div class="stat-row">{stats_html}</div>

        <div class="panel-header"><h2>VOLUME OVER TIME</h2></div>
        <canvas id="vc" height="160"></canvas>

        <div class="panel-header"><h2>MUSCLE BALANCE</h2></div>
        <div>{balance_html}</div>

        <div class="panel-header"><h2>PERSONAL RECORDS</h2></div>
        <table class="dossier">
          <thead><tr><th>EXERCISE</th><th>e1RM</th><th>TOP SET</th><th>DATE</th></tr></thead>
          <tbody>{prs_html}</tbody>
        </table>
      </div>
    </section>

    <section class="view" id="view-supps">
      <div class="panel">
        <div class="panel-header"><h2>ATTACHÉ CASE — SUPPLY STACK</h2>
          <div class="meter"><span>CREATINE STREAK: 7 DAYS</span></div></div>
        <div class="case-grid">{supps_html}</div>
        <div class="panel-header"><h2>7-DAY LOG</h2></div>
        <table class="dossier">
          <thead><tr><th>DATE</th><th>CREATINE</th><th>PROTEIN</th><th>WATER</th></tr></thead>
          <tbody>{history_html}</tbody>
        </table>
      </div>
    </section>

    <section class="view" id="view-timer">
      <div class="panel timer-panel">
        <div class="panel-header"><h2>MISSION CLOCK</h2></div>
        <div class="timer-display" id="td">02:00</div>
        <div class="timer-controls">
          <button class="btn preset" data-p="60">0:60</button>
          <button class="btn preset" data-p="90">1:30</button>
          <button class="btn preset" data-p="120">2:00</button>
          <button class="btn preset" data-p="180">3:00</button>
          <button class="btn preset" data-p="300">5:00</button>
        </div>
        <div class="timer-controls">
          <button class="btn primary" id="ts">START</button>
          <button class="btn" id="tp">PAUSE</button>
          <button class="btn" id="tr">RESET</button>
        </div>
        <div class="panel-header" style="margin-top:32px;"><h2>STOPWATCH</h2></div>
        <div class="timer-display small">14:32.7</div>
        <ul class="lap-list">
          <li><span>SET 04</span><span>00:48.2 (total 14:32.7)</span></li>
          <li><span>SET 03</span><span>03:21.4 (total 13:44.5)</span></li>
          <li><span>SET 02</span><span>02:58.9 (total 10:23.1)</span></li>
          <li><span>SET 01</span><span>02:44.3 (total 07:24.2)</span></li>
        </ul>
      </div>
    </section>

    <section class="view" id="view-map">
      <div class="panel">
        <div class="panel-header"><h2>OPERATIONS MAP</h2></div>
        <p class="readme">This is the offline preview with mock data. The
          live app pulls real workouts from your Hevy account, filtered to
          titles containing <code>v2</code>.</p>
        <ul class="readme">
          <li><strong>INVENTORY</strong> — every v2 workout as an inventory slot.</li>
          <li><strong>FILES</strong> — projected progressive overload, stats, PRs, weekly volume.</li>
          <li><strong>CASE</strong> — daily creatine/protein/electrolyte/water log with streaks.</li>
          <li><strong>TIMER</strong> — rest countdown + workout stopwatch.</li>
        </ul>
        <p class="readme dim">"Saved your progress. Don't waste it."</p>
      </div>
    </section>
  </main>

  <footer class="hud-bottom">
    <span id="status">SYSTEM ONLINE — PREVIEW MODE</span>
    <span class="quote">« Mock data. Deploy for the real thing. »</span>
  </footer>

  <script>
    // tabs
    document.querySelectorAll(".tab").forEach(t => t.onclick = () => {{
      document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      document.querySelectorAll(".view").forEach(x => x.classList.remove("active"));
      t.classList.add("active");
      document.getElementById("view-" + t.dataset.view).classList.add("active");
    }});
    // clock
    setInterval(() => document.getElementById("clock").textContent =
      new Date().toLocaleTimeString("en-GB"), 1000);
    // tiny working countdown for the timer view
    let rem = 120, end = null, iv = null;
    const fmt = s => {{ s = Math.max(0, Math.ceil(s));
      return String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0"); }};
    const td = document.getElementById("td");
    const tick = () => {{
      const r = end ? (end - Date.now())/1000 : rem;
      td.textContent = fmt(r);
      td.classList.toggle("warn", r > 0 && r <= 5);
      if (r <= 0 && end) {{ td.classList.add("done"); clearInterval(iv); iv = null; end = null; rem = 0; }}
    }};
    document.querySelectorAll(".preset").forEach(b => b.onclick = () => {{
      rem = +b.dataset.p; end = null; td.classList.remove("done","warn"); tick();
    }});
    document.getElementById("ts").onclick = () => {{
      if (rem <= 0) rem = 120;
      end = Date.now() + rem*1000;
      td.classList.remove("done");
      if (iv) clearInterval(iv); iv = setInterval(tick, 250); tick();
    }};
    document.getElementById("tp").onclick = () => {{
      if (!end) return; rem = (end-Date.now())/1000; end = null;
      if (iv) {{ clearInterval(iv); iv = null; }}
    }};
    document.getElementById("tr").onclick = () => {{
      end = null; rem = 120; if (iv) {{ clearInterval(iv); iv = null; }}
      td.classList.remove("done","warn"); tick();
    }};
    tick();
    // simple bar chart for weekly volume
    (function () {{
      const c = document.getElementById("vc");
      const series = [{weeks_js}];
      const ctx = c.getContext("2d");
      const w = c.clientWidth || 800, h = 160; c.width = w;
      const max = Math.max(...series.map(s => s.v));
      const barW = (w-60) / series.length;
      ctx.strokeStyle = "#4d5b35";
      ctx.beginPath(); ctx.moveTo(50,10); ctx.lineTo(50,h-24); ctx.lineTo(w-8,h-24); ctx.stroke();
      series.forEach((s,i) => {{
        const bh = (s.v/max) * (h-40) | 0;
        const x = 52 + i*barW, y = h-24-bh;
        const g = ctx.createLinearGradient(0,y,0,y+bh);
        g.addColorStop(0,"#c52a2a"); g.addColorStop(1,"#6b3a1a");
        ctx.fillStyle = g; ctx.fillRect(x, y, barW-4, bh);
        ctx.fillStyle = "#c9bf94"; ctx.font = "9px monospace";
        ctx.save(); ctx.translate(x+barW/2, h-8); ctx.rotate(-Math.PI/6);
        ctx.fillText(s.w, -16, 0); ctx.restore();
      }});
      ctx.fillStyle = "#8aa15a"; ctx.font = "10px monospace";
      ctx.fillText(max.toLocaleString()+" kg", 4, 14);
    }})();
  </script>
</body>
</html>'''
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
