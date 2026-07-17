"""Build the interactive web dashboard from ``models/prediction.json``.

Produces two artifacts from one template:
* ``web/index.html`` — a full standalone document (open via file:// or GitHub Pages).
* ``web/_artifact_body.html`` — the same inner content without the document
  shell, for publishing as a claude.ai Artifact (which supplies its own <head>).

Run:  python web/build_dashboard.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "models" / "prediction.json"
OUT_STANDALONE = ROOT / "web" / "index.html"
OUT_BODY = ROOT / "web" / "_artifact_body.html"
CREST = ROOT / "web" / "assets" / "crest.jpg"


def _crest_data_uri() -> str:
    """Inline the club crest as a data URI (works in both the standalone page
    and the self-contained Artifact, which blocks relative image loads)."""
    if not CREST.exists():
        return ""
    b64 = base64.b64encode(CREST.read_bytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _line_of(position: str | None) -> str:
    p = (position or "").lower()
    if "goalkeeper" in p:
        return "GK"
    if "back" in p or "defender" in p:
        return "DEF"
    if "forward" in p or "striker" in p or "winger" in p:
        return "FWD"
    return "MID"


def _merge_players(pred: dict) -> list[dict]:
    """Join squad bios with goalscorer + anytime predictions by name."""
    goals = {g["player"]: g for g in pred.get("goalscorers", [])}
    anytime = {a["player"]: a["p_anytime"] for a in pred.get("anytime_scorers", [])}
    players = []
    for p in pred.get("squad", []):
        name = p["name"]
        g = goals.get(name, {})
        players.append({
            "name": name,
            "position": p.get("position"),
            "age": p.get("age"),
            "nationality": p.get("nationality"),
            "line": _line_of(p.get("position")),
            "exp_goals": g.get("exp_goals", 0.0),
            "p_top": g.get("p_team_top_scorer", 0.0),
            "goals_last": g.get("goals_last_season", 0),
            "is_pen_taker": g.get("is_pen_taker", False),
            "new": g.get("new_or_scoreless", False),
            "p_anytime": anytime.get(name, 0.0),
        })
    # order each line by predicted goals so the sharp end reads first
    return players


def build() -> None:
    pred = json.loads(PRED.read_text(encoding="utf-8"))
    data = {
        "club": pred["club"],
        "crest": _crest_data_uri(),
        "season": pred["target_season"],
        "generated_at": pred["generated_at"],
        "promotion": pred["promotion_probability"],
        "range": pred["promotion_range"],
        "p_direct": pred["p_direct"],
        "p_playoff_reached": pred["p_playoff_reached"],
        "p_playoff_won": pred["p_playoff_won"],
        "se": pred["monte_carlo_se"],
        "mean_position": pred["mean_position"],
        "mean_points": pred["mean_points"],
        "mean_goals_for": pred.get("mean_goals_for", 0),
        "strength_rank": pred["strength_rank_in_group"],
        "group_size": len(pred["group"]),
        "position_distribution": pred["position_distribution"],
        "ensemble_members": pred["ensemble_members"],
        "strength_agreement": pred.get("strength_agreement", {}),
        "newcomers": pred.get("newcomers", []),
        "projected_table": pred.get("projected_table", []),
        "players": _merge_players(pred),
        "goalscorers": [g for g in pred.get("goalscorers", [])
                        if not g["player"].startswith("(other")],
        "pichichi": pred.get("pichichi_race", []),
        "sensitivity": pred.get("sensitivity", []),
        "backtest": pred.get("backtest"),
        "bootstrap": pred.get("bootstrap"),
        "model_params": pred.get("model_params", {}),
        "notes": pred.get("notes", []),
    }
    inner = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    OUT_BODY.write_text(inner, encoding="utf-8")
    OUT_STANDALONE.write_text(_STANDALONE.replace("<!--INNER-->", inner), encoding="utf-8")
    print(f"Wrote {OUT_STANDALONE} and {OUT_BODY}")


_STANDALONE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CD Binéfar — Promotion & Goalscorer Forecast</title>
</head>
<body>
<!--INNER-->
</body>
</html>
"""

# The inner content: <style> + markup + <script>. Valid inside both the
# standalone shell and the Artifact wrapper.
TEMPLATE = r"""
<style>
/* CD Binéfar identity: committed royal-blue-and-gold club colours — a deep
   Binéfar navy ground carries the page, crest-gold is the single hot accent,
   white ink, a sky-blue for interactive/data marks and pitch green from the
   crest field. Condensed display for big figures, restrained. */
:root{
  --bg:#0a1e46; --card:#0e2755; --card2:#123068; --fg:#f2f6ff; --muted:#a2b6dc;
  --line:#24407e; --accent:#f4b731; --accent-2:#ffcf5c; --accent-soft:rgba(244,183,49,.15);
  --blue:#6ea0ff; --accent2:#6ea0ff; --good:#3ddc97; --warn:#e0a05a;
  --pitch1:#1a5137; --pitch2:#143f2b;
  --shadow:0 1px 2px rgba(3,10,28,.5),0 14px 34px rgba(3,10,28,.45);
  --radius:12px; --maxw:1180px;
  --display:"Archivo","Arial Narrow","Roboto Condensed","Oswald","Helvetica Neue Condensed",Impact,system-ui,sans-serif;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono","SFMono-Regular",Menlo,Consolas,"JetBrains Mono","Liberation Mono",monospace;
}
*{box-sizing:border-box}
.bf-root{
  background:linear-gradient(180deg,#0c2350 0%,var(--bg) 40%);
  background-attachment:fixed;
  color:var(--fg); min-height:100vh;
  font-family:var(--sans); line-height:1.5; -webkit-font-smoothing:antialiased;
}
.bf-root *{font-variant-numeric:tabular-nums}
.wrap{max-width:var(--maxw); margin:0 auto; padding:clamp(16px,3vw,32px)}
.eyebrow{font-family:var(--mono); font-size:10.5px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); font-weight:600}
h1,h2,h3{margin:0; text-wrap:balance; letter-spacing:-.01em; font-family:var(--display); font-weight:600}
.card{background:var(--card); border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow);
  transition:transform .3s cubic-bezier(.16,1,.3,1),border-color .3s,box-shadow .3s}
.card.chart-card:hover,.card.tile:hover,.card.gauge-card:hover{transform:translateY(-3px); border-color:color-mix(in srgb,var(--accent) 45%,var(--line)); box-shadow:var(--shadow),0 0 22px rgba(244,183,49,.12)}
.section{margin-top:clamp(20px,3.5vw,40px)}
.section-h{display:flex; align-items:baseline; gap:12px; margin-bottom:14px}
.section-h h2{font-size:clamp(21px,2.6vw,27px); font-weight:700; text-transform:uppercase; letter-spacing:.02em}
.section-h .note{color:var(--muted); font-size:13px}
/* scroll-reveal */
.reveal{opacity:0; transform:translateY(18px); transition:opacity .6s cubic-bezier(.16,1,.3,1),transform .6s cubic-bezier(.16,1,.3,1)}
.reveal.in{opacity:1; transform:none}

/* header */
header.top{display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap}
.brand{display:flex; align-items:center; gap:14px}
.crest{width:52px;height:52px;object-fit:contain;border-radius:10px;
  background:radial-gradient(circle at 50% 40%, #fff, #eef2f8);
  padding:3px;box-shadow:var(--shadow),0 0 16px rgba(244,183,49,.20)}
.brand h1{font-size:clamp(23px,3vw,31px); font-weight:700; text-transform:uppercase; letter-spacing:.02em}
.brand p{margin:3px 0 0; color:var(--muted); font-size:13px; font-family:var(--mono); letter-spacing:.01em}
.updated{font-family:var(--mono); font-size:11px; letter-spacing:.08em; color:var(--muted);
  border:1px solid var(--line); border-radius:999px; padding:7px 13px; background:var(--card)}
.updated span{color:var(--fg)}

/* auto-generated narrative summary */
.lede{margin:20px 0 0; max-width:68ch; font-size:clamp(15px,1.7vw,18px); line-height:1.6;
  color:var(--fg); text-wrap:pretty}
.lede b{font-weight:700}
.lede .em{color:var(--accent); font-weight:700}

/* hero */
.hero{display:grid; grid-template-columns:minmax(260px,360px) 1fr; gap:18px; margin-top:22px}
@media(max-width:760px){.hero{grid-template-columns:1fr}}
.gauge-card{padding:22px; display:flex; flex-direction:column; align-items:center; text-align:center}
.gauge-wrap{position:relative; width:210px; height:210px}
.gauge-num{position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center}
.gauge-num b{font-family:var(--display); font-size:50px; font-weight:700; letter-spacing:-.02em; line-height:1; color:var(--accent)}
.gauge-num span{color:var(--muted); font-size:12px; margin-top:4px}
.gauge-sub{margin-top:14px; font-size:13px; color:var(--muted)}
.gauge-sub b{color:var(--fg)}
/* signature: the "ascenso" ladder — promotion framed as climbing tier 5 -> 4 */
.ascenso{margin-top:18px; width:100%; display:flex; flex-direction:column; gap:6px}
.ascenso .rung{display:flex; align-items:center; gap:10px; padding:8px 10px; border-radius:8px;
  font-family:var(--mono); font-size:11px; letter-spacing:.02em; border:1px solid var(--line)}
.ascenso .rung .tier{font-family:var(--display); font-weight:700; font-size:15px; width:18px; text-align:center}
.ascenso .rung .lab{flex:1; text-align:left; text-transform:uppercase; letter-spacing:.06em}
.ascenso .rung .tag{color:var(--muted); text-transform:uppercase; letter-spacing:.08em; font-size:10px}
.ascenso .target{background:color-mix(in srgb,var(--accent) 12%,transparent); border-color:color-mix(in srgb,var(--accent) 40%,transparent); color:var(--accent)}
.ascenso .target .tier{color:var(--accent)}
.ascenso .now{background:color-mix(in srgb,var(--blue) 10%,transparent); border-color:color-mix(in srgb,var(--blue) 32%,transparent)}
.ascenso .now .tier{color:var(--blue)}
.tiles{display:grid; grid-template-columns:repeat(3,1fr); gap:14px}
@media(max-width:520px){.tiles{grid-template-columns:repeat(2,1fr)}}
.tile{padding:16px}
.tile .k{font-family:var(--mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-weight:600}
.tile .v{font-family:var(--display); font-size:27px; font-weight:500; margin-top:6px; letter-spacing:-.01em}
.tile .s{font-size:12px; color:var(--muted); margin-top:2px}
.route{grid-column:1/-1; padding:16px 18px; display:flex; gap:22px; flex-wrap:wrap; align-items:center}
.route .rt{display:flex; flex-direction:column; gap:6px; min-width:150px; flex:1}
.route .rt .lab{font-size:12px; color:var(--muted); display:flex; justify-content:space-between}
.route .rt .lab b{color:var(--fg)}
.bar{height:8px; border-radius:999px; background:var(--card2); overflow:hidden}
.bar>i{display:block; height:100%; border-radius:999px}

/* pitch */
.pitch-grid{display:grid; grid-template-columns:1.5fr 1fr; gap:18px}
@media(max-width:820px){.pitch-grid{grid-template-columns:1fr}}
.pitch{position:relative; border-radius:var(--radius); overflow:hidden; padding:14px 10px;
  background:
    repeating-linear-gradient(0deg,var(--pitch1) 0 44px,var(--pitch2) 44px 88px);
  min-height:520px; display:flex; flex-direction:column; justify-content:space-between; gap:6px}
.pitch::before{content:""; position:absolute; inset:10px; border:2px solid rgba(255,255,255,.35); border-radius:10px; pointer-events:none}
.pitch::after{content:""; position:absolute; left:50%; top:10px; bottom:10px; width:2px; transform:translateX(-50%); background:rgba(255,255,255,.001)}
.pline{display:flex; justify-content:center; gap:10px; flex-wrap:wrap; position:relative; z-index:1}
.pline .rowlab{position:absolute; left:8px; top:-2px; font-size:10px; letter-spacing:.12em; color:rgba(255,255,255,.65); font-weight:700}
.shirt{border:none; cursor:pointer; background:transparent; display:flex; flex-direction:column; align-items:center; gap:4px; width:74px; padding:4px}
.disc{border-radius:50%; display:grid; place-items:center; font-weight:700; font-family:var(--mono);
  background:linear-gradient(135deg,#ffffff,#dfe7f5); color:var(--blue);
  border:2px solid rgba(255,255,255,.9); box-shadow:0 4px 10px rgba(0,0,0,.3); transition:transform .12s}
.shirt:hover .disc,.shirt:focus-visible .disc{transform:translateY(-3px) scale(1.05)}
.shirt.pen .disc{outline:2px solid var(--accent); outline-offset:2px; box-shadow:0 0 12px rgba(244,183,49,.55)}
.shirt .nm{color:#fff; font-size:11px; font-weight:700; text-shadow:0 1px 3px rgba(0,0,0,.5); max-width:74px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.shirt .gg{color:rgba(255,255,255,.85); font-size:10px; text-shadow:0 1px 3px rgba(0,0,0,.5)}
.legend-mini{display:flex; align-items:center; gap:8px; color:rgba(255,255,255,.85); font-size:11px; position:relative; z-index:1; justify-content:center; padding-top:4px}
.legend-mini i{background:#fff;border-radius:50%;display:inline-block;opacity:.9}

/* player detail */
.detail{padding:18px; display:flex; flex-direction:column; gap:12px}
.detail .empty{color:var(--muted); font-size:14px; text-align:center; margin:auto 0}
.detail h3{font-size:21px; font-weight:600}
.detail .meta{color:var(--muted); font-size:13px}
.pill{display:inline-flex; align-items:center; gap:6px; font-size:11px; font-weight:700; padding:3px 9px; border-radius:999px; font-family:var(--mono); letter-spacing:.03em;
  background:color-mix(in srgb,var(--blue) 10%,transparent); color:var(--blue); border:1px solid color-mix(in srgb,var(--blue) 25%,transparent)}
.pill.pen{background:color-mix(in srgb,var(--accent) 14%,transparent); color:var(--accent); border-color:color-mix(in srgb,var(--accent) 35%,transparent)}
.dstats{display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:4px}
.dstat{background:var(--card2); border-radius:12px; padding:12px}
.dstat .k{font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; font-weight:700}
.dstat .v{font-family:var(--display); font-size:23px; font-weight:500; margin-top:3px}

/* charts */
.chart-row{display:grid; grid-template-columns:1fr 1fr; gap:18px}
@media(max-width:820px){.chart-row{grid-template-columns:1fr}}
.chart-card{padding:18px}
.hbars{display:flex; flex-direction:column; gap:9px; margin-top:4px}
.hbar{display:grid; grid-template-columns:120px 1fr 42px; align-items:center; gap:10px; font-size:13px}
.hbar .nm{color:var(--fg); overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.hbar .track{background:var(--card2); border-radius:999px; height:14px; overflow:hidden}
.hbar .track>i{display:block; height:100%; border-radius:999px; background:var(--accent)}
.hbar .val{text-align:right; color:var(--muted); font-weight:700}
.posbars{display:flex; align-items:flex-end; gap:3px; height:180px; margin-top:8px}
.posbars .pb{flex:1; display:flex; flex-direction:column; align-items:center; justify-content:flex-end; gap:5px; height:100%}
.posbars .pb .col{width:100%; border-radius:5px 5px 0 0; background:var(--muted); min-height:2px; transition:filter .12s}
.posbars .pb.po .col{background:var(--accent)}
.posbars .pb.champ .col{background:var(--good)}
.posbars .pb:hover .col{filter:brightness(1.12)}
.posbars .pb .lab{font-size:9px; color:var(--muted)}
tbl,.tbl{width:100%}
table.tbl{border-collapse:collapse; width:100%; font-size:13px}
table.tbl th{ text-align:left; color:var(--muted); font-weight:700; font-size:11px; text-transform:uppercase; letter-spacing:.06em; padding:6px 8px; border-bottom:1px solid var(--line)}
table.tbl td{padding:7px 8px; border-bottom:1px solid var(--line)}
table.tbl tr:last-child td{border-bottom:none}
table.tbl .minibar{height:7px; border-radius:999px; background:var(--accent2); display:inline-block; vertical-align:middle}
.hl{background:color-mix(in srgb,var(--accent) 12%,transparent)}
/* team explorer panel (click a projected-table row) */
.teamx{background:var(--card2); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:14px; display:none}
.teamx.on{display:block}
.teamx .hd{display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:10px}
.teamx .hd .nm{font-family:var(--display); font-size:20px; font-weight:700; text-transform:uppercase; letter-spacing:.01em}
.teamx .hd .chips{display:flex; gap:6px; flex-wrap:wrap; margin-left:auto}
.teamx .chip{font-family:var(--mono); font-size:11px; padding:3px 8px; border-radius:6px; background:var(--card); border:1px solid var(--line); color:var(--muted)}
.teamx .chip b{color:var(--fg)}
.teamx .dist{display:flex; align-items:flex-end; gap:2px; height:90px}
.teamx .dist .b{flex:1; display:flex; flex-direction:column; align-items:center; justify-content:flex-end; height:100%; gap:3px}
.teamx .dist .b .col{width:100%; border-radius:3px 3px 0 0; background:var(--accent2); min-height:2px}
.teamx .dist .b.po .col{background:var(--accent)}
.teamx .dist .b.ch .col{background:var(--good)}
.teamx .dist .b .n{font-size:8px; color:var(--muted); font-family:var(--mono)}
/* position heatmap strip */
.heat{display:flex; gap:1px; min-width:230px}
.heat .cell{flex:1; height:18px; border-radius:2px; position:relative}
.heat .cut{width:0; border-left:2px dashed var(--muted); margin:0 1px; opacity:.6}
.heat-x{display:flex; gap:1px; min-width:230px; margin-top:3px}
.heat-x span{flex:1; text-align:center; font-size:8px; color:var(--muted); font-family:var(--mono)}
table.tbl td.strip{padding:6px 8px}
.ens{display:flex; flex-direction:column; gap:10px}
.ens .em{display:grid; grid-template-columns:170px 1fr 46px; gap:10px; align-items:center; font-size:13px}
.ens .em .track{background:var(--card2); border-radius:999px; height:12px; overflow:hidden}
.ens .em .track>i{display:block;height:100%;border-radius:999px;background:var(--accent2)}
.agree{display:flex; gap:8px; flex-wrap:wrap; margin-top:12px}
.chip{font-size:12px; background:var(--card2); border:1px solid var(--line); border-radius:999px; padding:5px 11px; color:var(--muted)}
.chip b{color:var(--good)}
.sens{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:4px}
@media(max-width:720px){.sens{grid-template-columns:repeat(2,1fr)}}
.sens .sp{background:var(--card2); border-radius:12px; padding:12px}
.sens .sp .t{font-size:11px; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:.05em}
.spark{display:flex; align-items:flex-end; gap:4px; height:56px; margin-top:8px}
.spark .sc{flex:1; background:var(--accent2); border-radius:3px 3px 0 0; min-height:3px; opacity:.85}
.spark-x{display:flex; gap:4px; margin-top:4px}
.spark-x span{flex:1; text-align:center; font-size:9px; color:var(--muted); overflow:hidden}
footer{margin-top:40px; color:var(--muted); font-size:12.5px; line-height:1.6}
footer b{color:var(--fg)}
footer .cols{display:grid; grid-template-columns:repeat(3,1fr); gap:18px; margin-top:10px}
@media(max-width:720px){footer .cols{grid-template-columns:1fr}}
.tooltip{position:fixed; z-index:50; pointer-events:none; background:var(--fg); color:var(--bg);
  padding:6px 9px; border-radius:8px; font-size:12px; font-weight:600; opacity:0; transition:opacity .1s; box-shadow:var(--shadow); max-width:220px}
:where(button,a,[tabindex]):focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:6px}
@media (prefers-reduced-motion:reduce){*{transition:none!important; animation:none!important}
  .reveal{opacity:1!important; transform:none!important}}
</style>

<div class="bf-root">
<div class="wrap">
  <header class="top">
    <div class="brand">
      <img class="crest" id="crest" alt="CD Binéfar crest" width="52" height="52">
      <div>
        <h1 id="clubName">CD Binéfar</h1>
        <p id="subtitle">Promotion &amp; goalscorer forecast · Tercera Federación Grupo 17</p>
      </div>
    </div>
    <div class="updated">Updated <span id="genTag">—</span></div>
  </header>

  <p class="lede" id="lede"></p>

  <!-- HERO -->
  <div class="hero">
    <div class="card gauge-card">
      <div class="eyebrow">Probability of promotion</div>
      <div class="gauge-wrap" style="margin-top:12px">
        <svg viewBox="0 0 120 120" width="210" height="210" id="gauge" aria-hidden="true"></svg>
        <div class="gauge-num"><b id="promoNum">–</b><span>any route</span></div>
      </div>
      <div class="gauge-sub">Ensemble range <b id="promoRange">–</b><br>Monte-Carlo SE <b id="promoSE">–</b></div>
      <div class="ascenso" aria-label="Promotion is a climb from tier 5 to tier 4">
        <div class="rung target"><span class="tier">4</span><span class="lab">Segunda Federación</span><span class="tag">target ↑</span></div>
        <div class="rung now"><span class="tier">5</span><span class="lab">Tercera Fed · Grupo 17</span><span class="tag">now</span></div>
      </div>
    </div>
    <div style="display:flex; flex-direction:column; gap:14px">
      <div class="tiles" id="tiles"></div>
      <div class="card route" id="routeStrip"></div>
    </div>
  </div>

  <!-- LINEUP -->
  <section class="section">
    <div class="section-h"><h2>Squad &amp; lineup</h2><span class="note">click a player · disc size = predicted goals · gold ring = penalty taker</span></div>
    <div class="pitch-grid">
      <div class="pitch" id="pitch"></div>
      <div class="card detail" id="detail"><div class="empty">Select a player on the pitch to see their 26/27 projection.</div></div>
    </div>
  </section>

  <!-- CHARTS -->
  <section class="section">
    <div class="chart-row">
      <div class="card chart-card">
        <div class="section-h"><h2>Predicted goalscorers</h2><span class="note">expected goals, 26/27</span></div>
        <div class="hbars" id="goalscorers"></div>
      </div>
      <div class="card chart-card">
        <div class="section-h"><h2>Finishing position</h2><span class="note">simulated distribution</span></div>
        <div class="posbars" id="posbars"></div>
        <div style="display:flex;gap:16px;margin-top:12px;font-size:11px;color:var(--muted)">
          <span><i style="display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--good)"></i> champion</span>
          <span><i style="display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--accent)"></i> play-off (2–5)</span>
          <span><i style="display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--muted)"></i> no promotion</span>
        </div>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="chart-row">
      <div class="card chart-card">
        <div class="section-h"><h2>Pichichi race</h2><span class="note">top scorer of the whole group</span></div>
        <div style="overflow-x:auto"><table class="tbl" id="pichichi"></table></div>
      </div>
      <div class="card chart-card">
        <div class="section-h"><h2>Model ensemble</h2><span class="note">promotion probability by variant</span></div>
        <div class="ens" id="ensemble"></div>
        <div class="agree" id="agreement"></div>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="card chart-card">
      <div class="section-h"><h2>Projected final table</h2><span class="note">finishing-position probability heatmap · 50k simulated seasons · click a row to explore a team · click headers to sort</span></div>
      <div class="teamx" id="teamx"></div>
      <div style="overflow-x:auto"><table class="tbl" id="projtable"></table></div>
    </div>
  </section>

  <section class="section">
    <div class="card chart-card">
      <div class="section-h"><h2>Sensitivity</h2><span class="note">how the promotion probability moves with each modelling assumption</span></div>
      <div class="sens" id="sensitivity"></div>
    </div>
  </section>

  <footer>
    <div><b id="clubFoot">CD Binéfar</b> · new joiners this season: <span id="newcomers"></span></div>
    <div class="cols">
      <div><b>Method.</b> Ensemble of time-weighted Dixon-Coles / independent-Poisson goals models → Monte-Carlo season simulation (+ territorial play-off) → Dirichlet-multinomial goalscorer allocation. <span id="btline"></span></div>
      <div><b>Data.</b> Sofascore (results, standings, goal incidents), Futbolme (goleadores), Transfermarkt (squad), Regional Preferente (promoted teams).</div>
      <div><b>Caveats.</b> No xG/minutes exist at tier 5; goalscorer shares are shrunk hard. Promotion is a rare, high-variance event — read the number as a calibrated probability. <span id="genat"></span></div>
    </div>
  </footer>
</div>
<div class="tooltip" id="tooltip"></div>
</div>

<script>
const DATA = /*__DATA__*/;
const $ = (s,r=document)=>r.querySelector(s);
const pct = x => (x*100).toFixed(1).replace(/\.0$/,'')+'%';
const pct0 = x => Math.round(x*100)+'%';

/* motion engine: crest, count-up, scroll-reveal, reduced-motion aware */
const RM = matchMedia('(prefers-reduced-motion:reduce)').matches;
if(DATA.crest){ const c=$('#crest'); if(c)c.src=DATA.crest; }
{ const gt=$('#genTag'); if(gt) gt.textContent=new Date(DATA.generated_at).toISOString().slice(0,10); }
function countUp(el,to,fmt,dur){ if(!el)return; if(RM){el.textContent=fmt(to);return;}
  dur=dur||1100; const t0=performance.now();
  const step=t=>{const p=Math.min(1,(t-t0)/dur), e=1-Math.pow(1-p,3);
    el.textContent=fmt(to*e); if(p<1)requestAnimationFrame(step); else el.textContent=fmt(to);};
  requestAnimationFrame(step);
}
const _revObs = ('IntersectionObserver' in window && !RM)
  ? new IntersectionObserver(es=>es.forEach(x=>{if(x.isIntersecting){x.target.classList.add('in');_revObs.unobserve(x.target);}}),{threshold:.12})
  : null;
function reveal(el){ if(!el)return; if(_revObs){el.classList.add('reveal');_revObs.observe(el);} }

/* tooltip */
const tip=$('#tooltip');
function showTip(e,html){tip.innerHTML=html;tip.style.opacity=1;moveTip(e);}
function moveTip(e){const p=12;tip.style.left=Math.min(e.clientX+p,innerWidth-tip.offsetWidth-8)+'px';tip.style.top=(e.clientY+p)+'px';}
function hideTip(){tip.style.opacity=0;}

/* header + hero */
$('#clubName').textContent=DATA.club;
$('#clubFoot').textContent=DATA.club;
$('#subtitle').textContent=`Promotion & goalscorer forecast · Tercera Federación Grupo 17 · ${DATA.season}`;
countUp($('#promoNum'), DATA.promotion, v=>pct(v), 1300);
$('#promoRange').textContent=`${pct(DATA.range[0])} – ${pct(DATA.range[1])}`;
$('#promoSE').textContent='±'+pct(DATA.se);
if(DATA.bootstrap){const gs=document.querySelector('.gauge-sub');
  gs.innerHTML+=`<br><span title="Bootstrap over ${DATA.bootstrap.n_boot} refits — reflects that tier-5 ratings are estimated from small samples">Parameter-uncertainty 90% CI <b>${pct(DATA.bootstrap.ci90[0])} – ${pct(DATA.bootstrap.ci90[1])}</b></span>`;}
$('#newcomers').textContent=(DATA.newcomers||[]).join(', ')||'—';
$('#genat').textContent='Generated '+new Date(DATA.generated_at).toISOString().slice(0,10)+'.';
if(DATA.backtest){const b=DATA.backtest;$('#btline').textContent=`Walk-forward log-loss ${b.log_loss} vs ${b.baseline_log_loss} baseline (${b.n_matches.toLocaleString()} matches).`;}

/* auto-generated plain-language summary (Opta-style storytelling) */
(function(){
  const promo=DATA.promotion, direct=DATA.p_direct, playoff=Math.max(0, promo-direct);
  // Honest route framing: name the larger promotion path, or "evenly split".
  // route clause always shows both promotion contributions (group vs play-off)
  const gpart=`winning the group (${pct(direct)})`, ppart=`the play-off route (${pct(playoff)})`;
  let route;
  if (promo<=0) route='an outside shot at either route';
  else if (Math.abs(direct-playoff) < 0.35*promo) route=`split roughly evenly between ${gpart} and ${ppart}`;
  else if (direct>playoff) route=`mostly from ${gpart}, less from ${ppart}`;
  else route=`mostly from ${ppart}, less from ${gpart}`;
  const scorer=(DATA.goalscorers||[]).find(g=>!g.player.startsWith('(other'));
  const scorerBit = scorer
    ? ` <b>${scorer.player}</b> is the favourite for top scorer (~${Math.round(scorer.exp_goals)} goals).`
    : '';
  const band = DATA.bootstrap
    ? ` It is a low-confidence call: allowing for small-sample noise, the honest range is <b>${pct(DATA.bootstrap.ci90[0])}–${pct(DATA.bootstrap.ci90[1])}</b>.`
    : '';
  $('#lede').innerHTML =
    `<b>${DATA.club}</b> are projected to finish with a mean of <b>${DATA.mean_position.toFixed(1)}` +
    ` of ${DATA.group_size}</b> (≈${Math.round(DATA.mean_points)} pts) in ${DATA.season}, ` +
    `giving a <span class="em">${pct(promo)}</span> chance of promotion — ${route}. ` +
    `They reach the top-five play-off ${pct0(DATA.p_playoff_reached)} of the time, but must then ` +
    `win it and clear a national phase to go up.${scorerBit}${band}`;
})();

/* gauge (donut, value on a 0–20% context scale for legibility of a small p) */
(function(){
  const svg=$('#gauge'), C=60, R=48, circ=2*Math.PI*R;
  const scaleMax=Math.max(0.2, DATA.range[1]*1.4);
  const frac=Math.min(1, DATA.promotion/scaleMax);
  const rlo=DATA.range[0]/scaleMax, rhi=DATA.range[1]/scaleMax;
  const arc=(from,to,color,w,dash)=>{
    const el=document.createElementNS('http://www.w3.org/2000/svg','circle');
    el.setAttribute('cx',C);el.setAttribute('cy',C);el.setAttribute('r',R);
    el.setAttribute('fill','none');el.setAttribute('stroke',color);el.setAttribute('stroke-width',w);
    el.setAttribute('stroke-linecap','round');
    el.setAttribute('stroke-dasharray',`${circ*(to-from)} ${circ}`);
    el.setAttribute('stroke-dashoffset',`${-circ*from}`);
    el.setAttribute('transform',`rotate(-90 ${C} ${C})`);
    if(dash)el.setAttribute('opacity',.35);
    svg.appendChild(el);
  };
  arc(0,1,'var(--card2)',10);                 // track
  arc(rlo,rhi,'var(--accent)',10,true);         // ensemble range (faint)
  arc(0,frac,'var(--accent)',10);               // mean value
})();

/* tiles */
const tiles=[
  {k:'Mean finish',v:DATA.mean_position.toFixed(1),s:`of ${DATA.group_size} teams`},
  {k:'Reach play-off',v:pct0(DATA.p_playoff_reached),s:'top-5 finish'},
  {k:'Mean points',v:Math.round(DATA.mean_points),s:'over 34 games'},
  {k:'Win title',v:pct(DATA.p_direct),s:'direct promotion'},
  {k:'Goals scored',v:Math.round(DATA.mean_goals_for),s:'projected'},
  {k:'Strength rank',v:'#'+DATA.strength_rank,s:`in the group`},
];
$('#tiles').innerHTML=tiles.map(t=>`<div class="card tile"><div class="k">${t.k}</div><div class="v">${t.v}</div><div class="s">${t.s}</div></div>`).join('');

/* route strip */
const routes=[
  {lab:'Direct (champion)',v:DATA.p_direct,c:'var(--good)'},
  {lab:'Reach play-off',v:DATA.p_playoff_reached,c:'var(--accent)'},
  {lab:'Win territorial play-off',v:DATA.p_playoff_won,c:'var(--accent2)'},
];
$('#routeStrip').innerHTML=routes.map(r=>`<div class="rt"><div class="lab"><span>${r.lab}</span><b>${pct(r.v)}</b></div><div class="bar"><i style="width:${Math.min(100,r.v*100/0.6)}%;background:${r.c}"></i></div></div>`).join('');

/* pitch / lineup */
(function(){
  const lines=[['FWD','Forwards'],['MID','Midfield'],['DEF','Defence'],['GK','Goalkeepers']];
  const maxG=Math.max(1,...DATA.players.map(p=>p.exp_goals));
  const html=lines.map(([code,label])=>{
    const ps=DATA.players.filter(p=>p.line===code).sort((a,b)=>b.exp_goals-a.exp_goals);
    if(!ps.length)return '';
    const chips=ps.map(p=>{
      const sz=28+Math.round(26*Math.sqrt(p.exp_goals/maxG));
      const init=p.name.split(' ').map(w=>w[0]).slice(0,2).join('').toUpperCase();
      return `<button class="shirt${p.is_pen_taker?' pen':''}" data-name="${encodeURIComponent(p.name)}">
        <span class="disc" style="width:${sz}px;height:${sz}px;font-size:${Math.max(10,sz/3)}px">${init}</span>
        <span class="nm">${p.name}</span><span class="gg">${p.exp_goals.toFixed(1)} g</span></button>`;
    }).join('');
    return `<div class="pline"><span class="rowlab">${label}</span>${chips}</div>`;
  }).join('');
  $('#pitch').innerHTML=html+`<div class="legend-mini"><i style="width:8px;height:8px"></i>fewer goals&nbsp;&nbsp;<i style="width:16px;height:16px"></i>more goals</div>`;
  $('#pitch').querySelectorAll('.shirt').forEach(b=>b.addEventListener('click',()=>{
    showPlayer(decodeURIComponent(b.dataset.name));
    $('#pitch').querySelectorAll('.shirt').forEach(s=>s.style.opacity=.6);
    b.style.opacity=1;
  }));
})();
/* default the detail panel to the top predicted scorer */
(function(){const top=[...DATA.players].sort((a,b)=>b.exp_goals-a.exp_goals)[0]; if(top)showPlayer(top.name);})();

function showPlayer(name){
  const p=DATA.players.find(x=>x.name===name); if(!p)return;
  const flag=p.nationality?` · ${p.nationality}`:'';
  $('#detail').innerHTML=`
    <div>
      <h3>${p.name}</h3>
      <div class="meta">${p.position||''}${p.age?` · ${p.age} yrs`:''}${flag}</div>
      <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">
        ${p.is_pen_taker?'<span class="pill pen">Penalty taker</span>':''}
        ${p.new?'<span class="pill">New / no goals last season</span>':''}
        <span class="pill">Line: ${p.line}</span>
      </div>
    </div>
    <div class="dstats">
      <div class="dstat"><div class="k">Expected goals 26/27</div><div class="v">${p.exp_goals.toFixed(1)}</div></div>
      <div class="dstat"><div class="k">Goals 25/26</div><div class="v">${p.goals_last}</div></div>
      <div class="dstat"><div class="k">Team top scorer</div><div class="v">${pct0(p.p_top)}</div></div>
      <div class="dstat"><div class="k">Scores in a match</div><div class="v">${pct0(p.p_anytime)}</div></div>
    </div>`;
}

/* goalscorer bars */
(function(){
  const gs=DATA.goalscorers.slice(0,10);
  const maxG=Math.max(1,...gs.map(g=>g.exp_goals));
  $('#goalscorers').innerHTML=gs.map(g=>`
    <div class="hbar" data-tip="${g.player}: ${g.exp_goals} exp goals · ${pct0(g.p_team_top_scorer)} to lead the team">
      <span class="nm">${g.player}</span>
      <span class="track"><i style="width:${g.exp_goals/maxG*100}%"></i></span>
      <span class="val">${g.exp_goals.toFixed(1)}</span></div>`).join('');
  $('#goalscorers').querySelectorAll('.hbar').forEach(el=>{
    el.addEventListener('mousemove',e=>showTip(e,el.dataset.tip));el.addEventListener('mouseleave',hideTip);});
})();

/* position distribution */
(function(){
  const d=DATA.position_distribution, maxP=Math.max(...d);
  $('#posbars').innerHTML=d.map((p,i)=>{
    const pos=i+1, cls=pos===1?'champ':(pos<=5?'po':'');
    return `<div class="pb ${cls}" data-tip="Finish ${pos}: ${pct(p)}"><div class="col" style="height:${p/maxP*100}%"></div><div class="lab">${pos}</div></div>`;
  }).join('');
  $('#posbars').querySelectorAll('.pb').forEach(el=>{
    el.addEventListener('mousemove',e=>showTip(e,el.dataset.tip));el.addEventListener('mouseleave',hideTip);});
})();

/* pichichi table */
(function(){
  const p=DATA.pichichi.slice(0,10); if(!p.length){$('#pichichi').innerHTML='<tr><td>No data</td></tr>';return;}
  const maxProb=Math.max(...p.map(x=>x.p_pichichi))||1;
  $('#pichichi').innerHTML='<tr><th>Player (team)</th><th>Exp goals</th><th>P(top scorer)</th></tr>'+
    p.map(x=>{const hl=x.player_team.includes(DATA.club)?' class="hl"':'';
      return `<tr${hl}><td>${x.player_team}</td><td>${x.exp_goals.toFixed(1)}</td>
      <td><span class="minibar" style="width:${Math.max(3,x.p_pichichi/maxProb*90)}px"></span> ${pct0(x.p_pichichi)}</td></tr>`;}).join('');
})();

/* ensemble */
(function(){
  const m=DATA.ensemble_members, vals=Object.values(m), mx=Math.max(...vals);
  $('#ensemble').innerHTML=Object.entries(m).map(([k,v])=>`
    <div class="em"><span>${k}</span><span class="track"><i style="width:${v/mx*100}%"></i></span><span style="text-align:right;color:var(--muted);font-weight:700">${pct(v)}</span></div>`).join('');
  $('#agreement').innerHTML=Object.entries(DATA.strength_agreement||{}).map(([k,v])=>`<span class="chip">${k} <b>${(+v).toFixed(2)}</b></span>`).join('');
})();

/* projected final table with a finishing-position heatmap strip (538-style) */
(function(){
  const rows=DATA.projected_table||[]; if(!rows.length){return;}
  const n=(rows[0].pos_dist||[]).length;
  // shared colour scale so cells are comparable across the whole table
  const maxCell=Math.max(0.001,...rows.flatMap(r=>r.pos_dist||[]));
  const cell=(p)=>{
    const t=Math.min(1,p/maxCell);
    // sequential single-hue coral ramp on a faint ground (accessible: value also in tooltip)
    return `background:color-mix(in srgb, var(--accent) ${Math.round(t*100)}%, var(--card2))`;
  };
  let sortKey='mean_points',desc=true;
  const cols=[
    {k:'mean_points',lab:'Pts'},{k:'p_champion',lab:'Champ'},{k:'p_top5',lab:'Top-5'},
  ];
  function strip(r){
    const cells=(r.pos_dist||[]).map((p,i)=>{
      const cut=(i===5)?'<span class="cut"></span>':'';
      return `${cut}<span class="cell" style="${cell(p)}" data-tip="${r.team} — finish ${i+1}: ${pct(p)}"></span>`;
    }).join('');
    return `<div class="heat">${cells}</div>`;
  }
  function render(){
    const sorted=[...rows].sort((a,b)=>{const d=a[sortKey]<b[sortKey]?-1:(a[sortKey]>b[sortKey]?1:0);return desc?-d:d;});
    const th=cols.map(c=>`<th data-k="${c.k}" style="cursor:pointer;text-align:right">${c.lab}${sortKey===c.k?(desc?' ▾':' ▴'):''}</th>`).join('');
    const head=`<tr><th>#</th><th data-k="team" style="cursor:pointer">Team</th><th>Finishing position 1 → ${n} (colour = probability)</th>${th}</tr>`;
    const body=sorted.map((r,i)=>{
      const hl=r.is_target?' class="hl"':'';
      return `<tr${hl} data-team="${encodeURIComponent(r.team)}" style="cursor:pointer" tabindex="0" role="button" aria-label="Explore ${r.team}">`+
        `<td style="color:var(--muted)">${i+1}</td><td>${r.team}</td>`+
        `<td class="strip">${strip(r)}</td>`+
        `<td style="text-align:right">${r.mean_points.toFixed(1)}</td>`+
        `<td style="text-align:right">${pct0(r.p_champion)}</td>`+
        `<td style="text-align:right">${pct0(r.p_top5)}</td></tr>`;
    }).join('');
    const el=$('#projtable'); el.innerHTML=head+body;
    el.querySelectorAll('th[data-k]').forEach(h=>h.addEventListener('click',()=>{
      const k=h.dataset.k; if(k===sortKey)desc=!desc; else {sortKey=k;desc=(k!=='team'&&k!=='mean_position');} render();}));
    el.querySelectorAll('.cell').forEach(c=>{
      c.addEventListener('mousemove',e=>showTip(e,c.dataset.tip)); c.addEventListener('mouseleave',hideTip);});
    el.querySelectorAll('tr[data-team]').forEach(tr=>{
      const pick=()=>showTeam(decodeURIComponent(tr.dataset.team));
      tr.addEventListener('click', pick);
      tr.addEventListener('keydown', e=>{ if(e.key==='Enter'||e.key===' '){e.preventDefault(); pick();}});
    });
  }

  // team explorer: full finishing-position distribution for any clicked team
  const byTeam=Object.fromEntries(rows.map(r=>[r.team,r]));
  function showTeam(team){
    const r=byTeam[team]; if(!r)return;
    const mx=Math.max(...r.pos_dist)||1;
    const bars=r.pos_dist.map((p,i)=>{
      const pos=i+1, cls=pos===1?'ch':(pos<=5?'po':'');
      return `<div class="b ${cls}" data-tip="Finish ${pos}: ${pct(p)}"><div class="col" style="height:${p/mx*100}%"></div><div class="n">${pos}</div></div>`;
    }).join('');
    const host=$('#teamx'); host.className='teamx on';
    host.innerHTML=`<div class="hd"><span class="nm">${r.team}</span>
      <span class="chips">
        <span class="chip">mean pts <b>${r.mean_points.toFixed(1)}</b></span>
        <span class="chip">mean finish <b>${r.mean_position.toFixed(1)}</b></span>
        <span class="chip">P(champion) <b>${pct(r.p_champion)}</b></span>
        <span class="chip">P(top-5) <b>${pct(r.p_top5)}</b></span>
      </span></div><div class="dist">${bars}</div>`;
    host.querySelectorAll('.b').forEach(b=>{
      b.addEventListener('mousemove',e=>showTip(e,b.dataset.tip)); b.addEventListener('mouseleave',hideTip);});
  }
  render();
  const tgt=rows.find(r=>r.is_target); if(tgt)showTeam(tgt.team);
})();

/* sensitivity small multiples */
(function(){
  const rows=DATA.sensitivity; if(!rows.length){$('#sensitivity').innerHTML='';return;}
  const params=[...new Set(rows.map(r=>r.parameter))];
  const nice={half_life_days:'Rating memory (days)',l2:'Shrinkage (L2)',newcomer_penalty:'Newcomer penalty',national_conversion:'Play-off conversion'};
  $('#sensitivity').innerHTML=params.map(pn=>{
    const sub=rows.filter(r=>r.parameter===pn);
    const mx=Math.max(...sub.map(s=>s.p_promotion)), mn=Math.min(...sub.map(s=>s.p_promotion));
    const bars=sub.map(s=>{const h=mx===mn?60:20+80*((s.p_promotion-mn)/(mx-mn));
      return `<div class="sc" style="height:${h}%" title="${s.value}: ${pct(s.p_promotion)}"></div>`;}).join('');
    const xs=sub.map(s=>`<span>${s.value}</span>`).join('');
    return `<div class="sp"><div class="t">${nice[pn]||pn}</div><div class="spark">${bars}</div><div class="spark-x">${xs}</div></div>`;
  }).join('');
})();

window.addEventListener('scroll',hideTip,{passive:true});

/* scroll-reveal: sections + hero panels ease in as they enter view */
document.querySelectorAll('.section, .hero .card').forEach(reveal);
</script>
"""


if __name__ == "__main__":
    build()
