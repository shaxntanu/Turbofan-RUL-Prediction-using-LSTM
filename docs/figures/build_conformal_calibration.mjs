import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

/*
Concept: explain split-conformal RUL intervals as a calibration comparator.
Sorted validation absolute residuals set q95; a caliper places the same radius
around an illustrative prediction and physically stops at the RUL bounds.
This is a visualization of software over simulated C-MAPSS data, not hardware.

Figure parts list (62 purposeful solids):
  Base (13): plate, four feet, four foot pads, engraved name plate, and three
        rulers for percentile rank, q radius, and RUL cycles.
  Calibration rack (23): tray, two guide rails, two supports, eight equal rank
        pegs, eight peg caps, and a selector carriage with comb.
  Quantile gauge (7): two supports, back plate, q track, and a three-piece
        sliding carriage with pointer.
  Interval comparator (19): two guide rails, four supports, two end stops with
        two caps, two three-piece moving jaws, and a three-piece prediction
        carriage.

The pegs are an ordering analogy, not individual observations or a sample
count. Only values in web/public/results/benchmark.json are shown as results.
*/

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../..");
const ANATOMY_DIR = resolve(
  process.env.ANATOMY_SKILL_DIR ?? join(homedir(), ".agents", "skills", "anatomy"),
);
const LICENSE = readFileSync(join(ANATOMY_DIR, "LICENSE"), "utf8").trim();
const kitPath = join(ANATOMY_DIR, "kit", "iso-kit.mjs");
if (!existsSync(kitPath)) {
  throw new Error(`Anatomy kit not found at ${kitPath}. Set ANATOMY_SKILL_DIR to its installed directory.`);
}
const k = await import(pathToFileURL(kitPath).href);

const WIDTH = 960;
const HEIGHT = 440;
const BASE = { x: -208, y: -72, w: 416, d: 144, r: 7 };
const FOOT_Z = -16;
const BASE_Z = -8;
const BASE_TOP = 0;
const RUL_FROM = 18;
const RUL_SPAN = 150;
const RUL_TO = RUL_FROM + RUL_SPAN;
const Q_FROM = -48;
const Q_SPAN = 54;
const Q_MAX = 50;
const DEFAULT_CENTER = 32;

const benchmark = JSON.parse(
  readFileSync(join(ROOT, "web", "public", "results", "benchmark.json"), "utf8"),
);
const SNAPSHOT = Object.fromEntries(
  Object.entries(benchmark.datasets).map(([name, result]) => {
    const uncertainty = result.uncertainty;
    if (
      uncertainty?.nominal_coverage !== 0.95 ||
      !Number.isFinite(uncertainty.interval_radius_cycles) ||
      !Number.isFinite(uncertainty.picp)
    ) {
      throw new Error(`Missing 95% interval values in the committed ${name} benchmark snapshot.`);
    }
    return [name, {
      model: result.selected_model,
      radius: uncertainty.interval_radius_cycles,
      observedCoverage: uncertainty.picp,
      intervalWidth: uncertainty.mpi_width_cycles,
    }];
  }),
);

const box = (plan, z, height, P, tone = "mid", bevel = 0.5) =>
  k.solidSvg(k.slabOf(plan, z, height, P, 8, bevel), { tone });
const dot = (point, tone = "mid", size = 0.55) =>
  k.dotsSvg([point], { size, tone });
const line = (path, tone = "lo", free = undefined) =>
  k.lineSvg(path, { tone, ...(free ? { free } : {}) });
const xForRul = (value) => RUL_FROM + (RUL_SPAN * value) / 125;
const xForQ = (value) => Q_FROM + (Q_SPAN * value) / Q_MAX;
const xForPercent = (value) => -175 + (105 * value) / 100;
const defaultData = SNAPSHOT.FD001;
const defaultRadius = defaultData.radius;
const defaultLow = Math.max(0, DEFAULT_CENTER - defaultRadius);
const defaultHigh = Math.min(125, DEFAULT_CENTER + defaultRadius);
const defaultQX = xForQ(defaultRadius);
const defaultCenterX = xForRul(DEFAULT_CENTER);
const defaultLowX = xForRul(defaultLow);
const defaultHighX = xForRul(defaultHigh);

const extent = [
  ...k.boxCorners(BASE, FOOT_Z, BASE_TOP),
  [-38, 18, 44],
  [RUL_TO + 6, 20, 36],
  [Q_FROM + Q_SPAN, 16, 22],
];
const P = k.fitProjection(extent, WIDTH, HEIGHT, { pad: 30, azimuth: 42 });
const at = (point) => k.iso(point, P);
const xAxis = k.axisVector(P, [1, 0, 0]);
const movedByX = (x) => `translate(${(xAxis[0] * x).toFixed(3)} ${(xAxis[1] * x).toFixed(3)})`;
const byDepth = (points) =>
  [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));

function baseAssembly() {
  const out = [`<path class="iso-halo" d="${k.haloOf(BASE, FOOT_Z, 16, P)}"/>`];
  const corners = k.corners(BASE, 15);
  for (const [x, y] of byDepth(corners)) {
    out.push(box({ x: x - 7, y: y - 6, w: 14, d: 12, r: 2 }, FOOT_Z, 4, P, "lo", 1));
    out.push(box({ x: x - 5, y: y - 4, w: 10, d: 8, r: 1.5 }, FOOT_Z + 4, 4, P, "mid", 0.8));
    out.push(dot([x, y, BASE_Z], "hi", 0.65));
    out.push(dot([x + 2, y, BASE_Z], "lo", 0.38));
  }
  out.push(box(BASE, BASE_Z, BASE_TOP - BASE_Z, P, "mid", 1.5));
  out.push(line(k.planOutline(k.insetPlan(BASE, 5), BASE_TOP, P), "lo"));

  const nameplate = { x: -198, y: 45, w: 40, d: 14, r: 1.5 };
  out.push(box(nameplate, BASE_TOP, 1, P, "lo", 0.4));
  out.push(k.faceTextSvg(k.topMatrix([-178, 52], BASE_TOP + 1, P, "x"), "Q95", {
    size: 2.6,
    tone: "hi",
    anchor: "middle",
  }));
  const percentStrip = { x: -178, y: 29, w: 108, d: 9, r: 1.2 };
  out.push(box(percentStrip, BASE_TOP, 1.4, P, "lo", 0.3));
  const percentTicks = k.topTicks(-175, -73, 5, 5, 29.8, 1.4, [1.5, 3.5], P);
  out.push(line(percentTicks.minor, "lo"), line(percentTicks.major, "mid"));
  const qStrip = { x: Q_FROM - 3, y: -31, w: Q_SPAN + 6, d: 12, r: 1.4 };
  out.push(box(qStrip, BASE_TOP, 1.2, P, "lo", 0.3));
  const qTicks = k.topTicks(Q_FROM, Q_FROM + Q_SPAN, 5.4, 2, -30, 1.2, [1.5, 3.2], P);
  out.push(line(qTicks.minor, "lo"), line(qTicks.major, "mid"));
  const rulStrip = { x: RUL_FROM - 4, y: 34, w: RUL_SPAN + 8, d: 15, r: 1.3 };
  out.push(box(rulStrip, BASE_TOP, 1.5, P, "lo", 0.3));
  const rulTicks = k.topTicks(RUL_FROM, RUL_TO, 6, 5, 34.8, 1.5, [1.7, 4], P);
  out.push(line(rulTicks.minor, "lo"), line(rulTicks.major, "mid"));
  return out.join("");
}

function validationRack() {
  const out = [];
  const tray = { x: -183, y: -24, w: 116, d: 28, r: 2.5 };
  out.push(box(tray, BASE_TOP, 4, P, "mid", 0.8));
  out.push(box({ x: -180, y: -20, w: 110, d: 3, r: 0.7 }, 4, 2, P, "hi", 0.4));
  out.push(box({ x: -180, y: 1, w: 110, d: 3, r: 0.7 }, 4, 2, P, "hi", 0.4));
  for (const x of [-175, -75]) {
    out.push(box({ x, y: -25, w: 7, d: 31, r: 1 }, BASE_TOP, 11, P, "mid", 0.7));
    out.push(dot([x + 3.5, -9, 11], "lo", 0.5));
  }

  const slots = Array.from({ length: 8 }, (_, index) => -172 + index * 13.2);
  for (const x of byDepth(slots.map((value) => [value, -9]))) {
    const px = x[0];
    out.push(box({ x: px - 2.2, y: -13, w: 4.4, d: 8, r: 0.6 }, 6, 12, P, "hi", 0.5));
    out.push(box({ x: px - 3.4, y: -12, w: 6.8, d: 4, r: 0.6 }, 14, 2, P, "mid", 0.5));
    out.push(dot([px, -11, 16], "lo", 0.42));
  }

  const selectorX = xForPercent(95);
  const selector = [
    box({ x: selectorX - 3.8, y: -20, w: 7.6, d: 25, r: 1.4 }, 8, 6, P, "mid", 0.7),
    box({ x: selectorX - 1.2, y: -18, w: 2.4, d: 20, r: 0.5 }, 14, 10, P, "hi", 0.4),
    dot([selectorX, -7, 24], "mid", 0.55),
  ];
  out.push(`<g data-part="rank-detent" transform="${movedByX(0)}">${selector.join("")}</g>`);
  return out.join("");
}

function quantileGauge() {
  const out = [];
  const x = Q_FROM + Q_SPAN / 2;
  for (const y of [-16, 9]) {
    out.push(box({ x: x - 4, y: y - 4, w: 8, d: 8, r: 1 }, BASE_TOP, 12, P, "mid", 0.6));
    out.push(dot([x, y, 12], "lo", 0.45));
  }
  const plate = { x: Q_FROM - 3, y: -18, w: Q_SPAN + 6, d: 31, r: 2.2 };
  out.push(box(plate, 12, 5, P, "mid", 0.8));
  out.push(box({ x: Q_FROM, y: -12, w: Q_SPAN, d: 18, r: 1 }, 17, 2, P, "lo", 0.4));
  const ticks = k.topTicks(Q_FROM + 1, Q_FROM + Q_SPAN - 1, 5.4, 2, -11, 19, [2, 4], P);
  out.push(line(ticks.minor, "lo"), line(ticks.major, "hi"));
  const gaugeScrews = [
    [Q_FROM - 0.5, -15], [Q_FROM - 0.5, 10],
    [Q_FROM + Q_SPAN + 0.5, -15], [Q_FROM + Q_SPAN + 0.5, 10],
  ];
  for (const [sx, sy] of gaugeScrews) out.push(dot([sx, sy, 17], "lo", 0.5));

  const carriage = [
    box({ x: defaultQX - 4, y: -15, w: 8, d: 25, r: 1.2 }, 19, 5, P, "lit", 0.7),
    box({ x: defaultQX - 1.1, y: -11, w: 2.2, d: 19, r: 0.4 }, 24, 7, P, "hi", 0.3),
    box({ x: defaultQX - 6, y: -9, w: 12, d: 4, r: 0.8 }, 31, 2, P, "mid", 0.5),
    dot([defaultQX, -7, 33], "hi", 0.55),
    dot([defaultQX - 3.5, -6, 33], "lo", 0.42),
    dot([defaultQX + 3.5, -6, 33], "lo", 0.42),
  ];
  out.push(`<g data-part="quantile-carriage" data-anchor="${defaultQX}" transform="${movedByX(0)}">${carriage.join("")}</g>`);
  return out.join("");
}

function intervalComparator() {
  const out = [];
  for (const y of [8, 64]) {
    out.push(box({ x: RUL_FROM - 2, y, w: RUL_SPAN + 4, d: 4, r: 0.8 }, BASE_TOP, 4, P, "mid", 0.5));
    for (const x of [RUL_FROM + 8, RUL_TO - 8]) {
      out.push(box({ x: x - 3, y: y - 3, w: 6, d: 10, r: 0.8 }, BASE_TOP, 10, P, "hi", 0.5));
      out.push(dot([x, y + 2, 10], "lo", 0.45));
    }
  }
  for (const [x, y] of [[RUL_FROM - 1, 17], [RUL_TO - 5, 17]]) {
    out.push(box({ x, y, w: 6, d: 29, r: 1.2 }, BASE_TOP, 17, P, "mid", 0.8));
    out.push(box({ x: x - 1, y: y + 4, w: 8, d: 4, r: 0.8 }, 17, 3, P, "hi", 0.4));
    out.push(dot([x + 3, y + 6, 20], "lo", 0.45));
  }

  function jawAt(worldX, facing) {
    const toothX = facing < 0 ? worldX - 2 : worldX + 4;
    return [
      box({ x: worldX - 2, y: 20, w: 6, d: 31, r: 1.1 }, 2, 24, P, "hi", 0.7),
      box({ x: toothX, y: 24, w: 3.5, d: 23, r: 0.6 }, 12, 6, P, "mid", 0.5),
      box({ x: worldX - 3, y: 17, w: 8, d: 4, r: 0.6 }, 26, 2, P, "mid", 0.4),
      dot([worldX + 1, 22, 27], "lo", 0.48),
      dot([worldX + 1, 45, 27], "lo", 0.48),
    ].join("");
  }
  out.push(`<g data-part="lower-jaw" data-anchor="${defaultLowX}" transform="${movedByX(0)}">${jawAt(defaultLowX, 1)}</g>`);
  out.push(`<g data-part="upper-jaw" data-anchor="${defaultHighX}" transform="${movedByX(0)}">${jawAt(defaultHighX, -1)}</g>`);

  const centre = [
    box({ x: defaultCenterX - 4.5, y: 22, w: 9, d: 26, r: 1.4 }, 4, 5, P, "mid", 0.7),
    box({ x: defaultCenterX - 2.2, y: 27, w: 4.4, d: 15, r: 0.8 }, 9, 17, P, "hi", 0.6),
    box({ x: defaultCenterX - 6, y: 31, w: 12, d: 7, r: 1 }, 26, 3, P, "mid", 0.5),
    dot([defaultCenterX, 34, 29], "lo", 0.45),
    dot([defaultCenterX - 3.5, 34, 29], "lo", 0.45),
    dot([defaultCenterX + 3.5, 34, 29], "lo", 0.45),
  ];
  out.push(`<g data-part="prediction-carriage" data-anchor="${defaultCenterX}" transform="${movedByX(0)}">${centre.join("")}</g>`);
  return out.join("");
}

function figureBody(theme) {
  const body = [baseAssembly(), validationRack(), quantileGauge(), intervalComparator()];
  return k.figureSvg({
    width: WIDTH,
    height: HEIGHT,
    standalone: theme,
    label:
      "A conceptual conformal calibration comparator. Sorted validation residuals select the 95 percent radius q; the radius gauge and two caliper jaws show the clipped interval around an illustrative RUL prediction on a zero to 125 cycle scale. The rank pegs are schematic and do not show actual residuals.",
    body,
  });
}

function initialReadout(dataset) {
  const result = SNAPSHOT[dataset] ?? SNAPSHOT.FD001;
  return `${dataset} · ${result.model} · q95 ${result.radius.toFixed(1)} cycles`;
}

function buildSvg() {
  const svg = figureBody("light");
  const notice = `<!--\n${LICENSE}\n-->\n`;
  writeFileSync(join(HERE, "conformal-calibration.svg"), notice + svg, "utf8");
  console.log("Wrote docs/figures/conformal-calibration.svg");
}

function buildHtml() {
  const svg = figureBody(false);
  const figure = k.plateHtml({
    fig: "Fig 1",
    title: "Split-conformal RUL intervals",
    hint: "Move the centre · arrow keys work too",
    readout: initialReadout("FD001"),
    keys: [
      { mark: "flat", label: "Sorted validation absolute residuals (rank slots are schematic)" },
      { mark: "lit", label: "q95 radius from the selected stored run" },
      { mark: "raised", label: "Caliper limits: clamp ŷ ± q95 to 0–125 cycles" },
    ],
    caption:
      "Conceptual software bench for simulated C-MAPSS, not aircraft hardware. q95 is the k-th sorted validation absolute residual, with k = min(ceil((n+1) × 0.95), n). Move illustrative ŷ to see [max(0, ŷ−q95), min(125, ŷ+q95)]. Saved q95 and observed test coverage come from the selected benchmark run; coverage is not a guarantee.",
    body: `<div class="iso-stage" data-stage tabindex="0" role="slider" aria-label="Illustrative RUL prediction centre. Use arrow keys or move the pointer to inspect the conformal interval." aria-valuemin="0" aria-valuemax="125" aria-valuenow="${DEFAULT_CENTER}">${svg}</div>`,
  });
  const dataJson = JSON.stringify(SNAPSHOT);
  const controls = `
    <section class="calibration-controls" aria-label="Calibration figure controls">
      <label class="subset-control" data-standalone-only>Dataset subset
        <select id="subset-select">${Object.keys(SNAPSHOT).map((name) => `<option value="${name}">${name}</option>`).join("")}</select>
      </label>
      <label class="prediction-control" for="prediction-slider">Illustrative prediction centre <span id="prediction-value">${DEFAULT_CENTER}</span> cycles
        <input id="prediction-slider" type="range" min="0" max="125" step="1" value="${DEFAULT_CENTER}" />
      </label>
      <p id="interval-summary" aria-live="polite"></p>
      <p id="snapshot-summary"></p>
    </section>`;
  const css = `
    <style>
      .calibration-controls{display:grid;grid-template-columns:minmax(130px,190px) 1fr;align-items:end;gap:16px;margin:18px 0 0;padding:16px 18px;border:1px solid var(--anatomy-faint);border-radius:12px;background:var(--anatomy-card);color:var(--anatomy-text)}
      .calibration-controls label{display:grid;gap:7px;font-size:12px}
      .calibration-controls select,.calibration-controls input{font:inherit;color:var(--anatomy-strong)}
      .calibration-controls select{min-height:36px;padding:6px 9px;border:1px solid var(--anatomy-faint);border-radius:7px;background:var(--anatomy-paper)}
      .calibration-controls input{width:100%;accent-color:var(--anatomy-hi)}
      #interval-summary{grid-column:1/-1;margin:0;color:var(--anatomy-strong);font-family:var(--anatomy-mono);font-size:12px}
      #snapshot-summary{grid-column:1/-1;margin:-9px 0 0;color:var(--anatomy-muted);font-size:11px}
      [data-standalone-only][hidden]{display:none}
      body.embedded .iso-page{width:100%;margin:0}
      body.embedded .iso-figure{margin:0}
      body.embedded .iso-plate{padding:16px 20px}
      body.embedded .iso-plate-corner[data-corner="fig"],
      body.embedded .iso-plate-corner[data-corner="hint"]{display:none}
      body.embedded .calibration-controls{grid-template-columns:1fr;gap:8px;margin:10px 0 0;padding:10px 14px}
      body.embedded #interval-summary,
      body.embedded #snapshot-summary{grid-column:1;margin:0}
      @media(max-width:560px){.calibration-controls{grid-template-columns:1fr;gap:12px;padding:13px}.calibration-controls label{font-size:11px}#interval-summary,#snapshot-summary{grid-column:1}}
      @media(max-width:560px){body.embedded .iso-plate{padding:14px 8px 10px}body.embedded .iso-plate-corner[data-corner="title"]{font-size:9px;letter-spacing:.05em}body.embedded .calibration-controls{padding:10px}}
    </style>`;
  const live = `
    const DATA = ${dataJson};
    const params = new URLSearchParams(location.search);
    const embedded = window.parent !== window;
    document.body.classList.toggle("embedded", embedded);
    const requestedTheme = params.get("theme");
    document.body.dataset.theme = requestedTheme === "light" || requestedTheme === "dark"
      ? requestedTheme
      : (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const stage = document.querySelector("[data-stage]");
    const svg = stage.querySelector("svg");
    const readout = document.querySelector("[data-readout]");
    const slider = document.querySelector("#prediction-slider");
    const subset = document.querySelector("#subset-select");
    const centerPart = svg.querySelector('[data-part="prediction-carriage"]');
    const lowPart = svg.querySelector('[data-part="lower-jaw"]');
    const highPart = svg.querySelector('[data-part="upper-jaw"]');
    const qPart = svg.querySelector('[data-part="quantile-carriage"]');
    const valueLabel = document.querySelector("#prediction-value");
    const intervalSummary = document.querySelector("#interval-summary");
    const snapshotSummary = document.querySelector("#snapshot-summary");
    const xAxis = ${JSON.stringify(xAxis)};
    const RUL_FROM = ${RUL_FROM}, RUL_SPAN = ${RUL_SPAN};
    const Q_FROM = ${Q_FROM}, Q_SPAN = ${Q_SPAN}, Q_MAX = ${Q_MAX};
    const anchors = {
      center: Number(centerPart.dataset.anchor),
      low: Number(lowPart.dataset.anchor),
      high: Number(highPart.dataset.anchor),
      q: Number(qPart.dataset.anchor),
    };
    const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
    const xForRul = (value) => RUL_FROM + RUL_SPAN * value / 125;
    const xForQ = (value) => Q_FROM + Q_SPAN * value / Q_MAX;
    const translate = (part, worldX, anchor) => {
      const dx = worldX - anchor;
      part.setAttribute("transform", "translate(" + (xAxis[0] * dx).toFixed(3) + " " + (xAxis[1] * dx).toFixed(3) + ")");
    };
    let dataset = DATA[params.get("dataset")] ? params.get("dataset") : "FD001";
    let radius = DATA[dataset].radius;
    let target = ${DEFAULT_CENTER};
    let current = target;
    let currentRadius = radius;
    let frame = 0, last = 0, visible = false, touring = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    let elapsed = 0, idleTimer = 0;
    const tour = [12, 62, 113];
    const stageBox = () => svg.getBoundingClientRect();
    function paint() {
      const low = Math.max(0, current - currentRadius);
      const high = Math.min(125, current + currentRadius);
      translate(centerPart, xForRul(current), anchors.center);
      translate(lowPart, xForRul(low), anchors.low);
      translate(highPart, xForRul(high), anchors.high);
      translate(qPart, xForQ(currentRadius), anchors.q);
      valueLabel.textContent = String(Math.round(current));
      slider.value = String(Math.round(current));
      intervalSummary.textContent = "Illustrative interval  [" + low.toFixed(1) + ", " + high.toFixed(1) + "] cycles  ·  q95 " + currentRadius.toFixed(2) + " cycles";
      const result = DATA[dataset];
      readout.textContent = dataset + " · " + result.model + " · q95 " + result.radius.toFixed(1) + " cycles";
      snapshotSummary.textContent = "Saved " + Math.round(0.95 * 100) + "% nominal interval · " + (result.observedCoverage * 100).toFixed(2) + "% observed test coverage for " + dataset + " (" + result.model + ").";
      stage.setAttribute("aria-valuenow", String(Math.round(current)));
      stage.setAttribute("aria-valuetext", "Illustrative centre " + Math.round(current) + " cycles; interval " + low.toFixed(1) + " to " + high.toFixed(1) + " cycles.");
    }
    function tick(now) {
      const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
      last = now;
      if (touring) { elapsed += dt; target = tour[Math.floor(elapsed / 2.4) % tour.length]; }
      const ease = matchMedia("(prefers-reduced-motion: reduce)").matches ? 1 : 1 - Math.exp(-dt / 0.28);
      current += (target - current) * ease;
      currentRadius += (radius - currentRadius) * ease;
      if (Math.abs(target - current) < 0.03) current = target;
      if (Math.abs(radius - currentRadius) < 0.01) currentRadius = radius;
      paint();
      const settled = !touring && current === target && currentRadius === radius;
      frame = visible && !settled ? requestAnimationFrame(tick) : 0;
    }
    const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
    const steer = (value) => {
      touring = false;
      target = clamp(Number(value), 0, 125);
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        if (!matchMedia("(prefers-reduced-motion: reduce)").matches) { touring = true; elapsed = 0; run(); }
      }, 4200);
      run();
    };
    function setDataset(name) {
      if (!DATA[name]) return;
      dataset = name;
      radius = DATA[name].radius;
      if (subset) subset.value = name;
      history.replaceState(null, "", "?dataset=" + name + (requestedTheme ? "&theme=" + requestedTheme : ""));
      run();
      if (!visible) { currentRadius = radius; paint(); }
    }
    if (subset) {
      subset.value = dataset;
      if (embedded) subset.closest("label").hidden = true;
      subset.addEventListener("change", () => setDataset(subset.value));
    }
    slider.addEventListener("input", () => steer(slider.value));
    stage.addEventListener("pointermove", (event) => {
      const rect = stageBox();
      const share = clamp((event.clientX - rect.left) / rect.width, 0, 1);
      steer(125 * share);
    });
    stage.addEventListener("pointerleave", () => { clearTimeout(idleTimer); idleTimer = setTimeout(() => { if (!matchMedia("(prefers-reduced-motion: reduce)").matches) touring = true; run(); }, 4200); });
    stage.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowDown", "ArrowRight", "ArrowUp", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? 125 : target + ((event.key === "ArrowRight" || event.key === "ArrowUp") ? 1 : -1);
      steer(next);
    });
    new IntersectionObserver((entries) => {
      visible = entries[entries.length - 1].isIntersecting;
      if (visible) run();
    }, { rootMargin: "100px 0px" }).observe(stage);
    if (subset && embedded) subset.closest("label").hidden = true;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)");
    reduced.addEventListener?.("change", (event) => { touring = !event.matches; run(); });
    paint();
  `;
  const html = k.pageHtml({
    title: "Split-conformal RUL calibration",
    theme: "dark",
    width: 1000,
    body: `${css}<div class="calibration-page">${figure}${controls}</div>`,
    script: live,
  });
  const notice = `<!--\n${LICENSE}\n-->\n`;
  writeFileSync(join(ROOT, "web", "public", "conformal-calibration.html"), notice + html, "utf8");
  console.log("Wrote web/public/conformal-calibration.html");
}

const mode = process.argv[2];
if (mode === "--svg") buildSvg();
else if (mode === "--html") buildHtml();
else if (!mode || mode === "--all") { buildSvg(); buildHtml(); }
else throw new Error("Usage: node docs/figures/build_conformal_calibration.mjs [--svg|--html|--all]");
