import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../vendor/iso-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "../../../../");
const OUTPUT = resolve(ROOT, "web/public/presentation/figures/rul-workbench-anatomy.svg");

const WIDTH = 1400;
const HEIGHT = 520;
const PLATE = { x: -205, y: -70, w: 410, d: 140, r: 7 };
const P = k.fitProjection(
  [...k.boxCorners(PLATE, -10, 0), [-177, -42, 38], [150, 0, 60]],
  WIDTH,
  HEIGHT,
  { pad: 42, azimuth: 42, elevation: 30 },
);
const at = (point) => k.iso(point, P);
const S = (paths, style = {}) => k.solidSvg(paths, style);
const L = (path, style = {}) => k.lineSvg(path, style);
const solids = [];
let partCount = 0;

function addSolid(center, paths, style) {
  solids.push({ depth: k.depthOf(center, P), markup: S(paths, style) });
  partCount += 1;
}

function slab(plan, z, height, style = {}, bevel = 0.8) {
  const center = [plan.x + plan.w / 2, plan.y + plan.d / 2, z + height / 2];
  addSolid(center, k.slabOf(plan, z, height, P, 6, bevel), style);
}

function cylinder(x, y, radius, z, height, style = {}) {
  addSolid([x, y, z + height / 2], k.cylinder(x, y, radius, z, height, P, 28, 0.35), style);
}

const baseFoot = -10;
for (const [x, y] of k.corners(PLATE, 18)) {
  cylinder(x, y, 7, baseFoot, 5, { tone: "lo" });
}
slab(PLATE, -5, 8, { tone: "mid" }, 2);
for (const [x, y] of k.corners(PLATE, 12)) {
  cylinder(x, y, 2.2, 3, 1.3, { tone: "hi" });
}

const wires = [
  k.wireSvg(at([-148, -61, 4]), at([-122, -61, 4]), { tone: "lit", size: 1.1 }),
  k.wireSvg(at([-57, -61, 4]), at([-27, -61, 4]), { tone: "mid", size: 1.1 }),
  k.wireSvg(at([-27, -61, 4]), at([-27, -51, 13]), { tone: "blue", ends: "to", size: 1.1 }),
  k.wireSvg(at([-27, -61, 4]), at([-27, 8, 13]), { tone: "green", ends: "to", size: 1.1 }),
  k.wireSvg(at([47, -50, 13]), at([125, -50, 13]), { tone: "blue", ends: "from", size: 1.1 }),
  k.wireSvg(at([47, 48, 13]), at([125, 48, 13]), { tone: "green", ends: "from", size: 1.1 }),
  k.wireSvg(at([125, -50, 13]), at([140, -22, 20]), { tone: "blue", ends: "to", size: 1.1 }),
  k.wireSvg(at([125, 48, 13]), at([140, 22, 20]), { tone: "green", ends: "to", size: 1.1 }),
];

const sensors = [];
for (let column = 0; column < 3; column += 1) {
  for (let row = 0; row < 5; row += 1) {
    const x = -177 + column * 14;
    const y = -42 + row * 21;
    sensors.push({ x, y });
    cylinder(x, y, 2.1, 3, 23, { tone: "mid" });
    cylinder(x, y, 3.1, 26, 2.6, { tone: "hi" });
  }
}

const cycleTiles = [];
for (let row = 0; row < 5; row += 1) {
  for (let column = 0; column < 6; column += 1) {
    const plan = {
      x: -119 + column * 10,
      y: -48 + row * 22,
      w: 8,
      d: 18,
      r: 1.2,
    };
    cycleTiles.push({ plan, selected: row === 2 && column === 3 });
    slab(plan, 3, 2, { tone: "mid", lit: row === 2 && column === 3 }, 0.35);
  }
}

const ridgeBoard = { x: -26, y: -51, w: 76, d: 42, r: 4 };
slab(ridgeBoard, 3, 8, { tone: "mid" }, 1.2);
for (let index = 0; index < 5; index += 1) {
  slab(
    { x: -20 + index * 13, y: -43, w: 8, d: 24, r: 1.4 },
    11,
    4,
    { tone: index === 0 ? "blue" : "lo" },
    0.45,
  );
}

const lstmBoard = { x: -26, y: 8, w: 76, d: 42, r: 4 };
slab(lstmBoard, 3, 8, { tone: "mid" }, 1.2);
for (let layer = 0; layer < 2; layer += 1) {
  const plan = { x: -17 + layer * 32, y: 16, w: 25, d: 26, r: 2.2 };
  slab(plan, 11, 4, { tone: layer === 1 ? "green" : "lo" }, 0.6);
  const nodes = [];
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      nodes.push(at([plan.x + 5 + column * 7.5, plan.y + 5 + row * 8, 15.4]));
    }
  }
  solids.push({ depth: k.depthOf([plan.x + 12, plan.y + 13, 15], P), markup: k.dotsSvg(nodes, { size: 1, tone: "hi" }) });
}

const gauge = [140, 0];
cylinder(gauge[0], gauge[1], 4.5, 3, 36, { tone: "mid" });
cylinder(gauge[0], gauge[1], 20, 39, 9, { tone: "hi" });
const dialRing = k.ring(gauge[0], gauge[1], 16, 48, P, 48);
const dialNeedle = k.pathOf([at([gauge[0] - 2, gauge[1], 48.4]), at([gauge[0] + 10, gauge[1] + 5, 48.4])]);
const details = [
  L(k.planOutline(k.insetPlan(PLATE, 6), 3, P), { tone: "lo" }),
  L(dialRing, { tone: "mid" }),
  L(dialNeedle, { tone: "lit", free: "end" }),
  k.dotsSvg(sensors.map(({ x, y }) => at([x, y, 28.8])), { size: 0.6, tone: "blue" }),
  ...cycleTiles.map(({ plan, selected }) => {
    const y = plan.y + plan.d - 2;
    const d = k.pathOf([at([plan.x + 1.4, y, 5.1]), at([plan.x + plan.w - 1.4, y, 5.1])]);
    return L(d, { tone: selected ? "lit" : "lo", free: true });
  }),
];

const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  standalone: true,
  label:
    "Isometric RUL workbench showing fifteen retained sensor channels, a thirty-cycle window, Ridge feature channels, two LSTM layers, and a validation selection gauge. The figure is an explanatory schematic, not a physical C-MAPSS engine cutaway.",
  body: [
    `<path class="iso-halo" d="${k.haloOf(PLATE, -10, 8, P)}"/>`,
    ...solids.sort((a, b) => a.depth - b.depth).map(({ markup }) => markup),
    ...wires,
    ...details,
  ],
});

if (partCount < 40 || partCount > 80) {
  throw new Error(`The anatomy figure must contain 40 to 80 physical parts, got ${partCount}`);
}

const palette = `<style>.iso{--anatomy-paper:#141a24;--anatomy-face:#1c2632;--anatomy-top:#263240;--anatomy-shade-0:#10151e;--anatomy-shade-1:#151e29;--anatomy-shade-2:#1c2734;--anatomy-shade-3:#202c3a;--anatomy-lit-top:#3bc9de;--anatomy-lit-shade:#1c8193;--anatomy-hi:#eaecef;--anatomy-mid:#8a93a6;--anatomy-lo:#384252;--anatomy-faint:#262e3d;--anatomy-lit:#3bc9de;--anatomy-halo:#05070b;--anatomy-dot:#cbd5e1;--anatomy-blue:#3bc9de;--anatomy-green:#4fd1a5}</style>`;
const outputSvg = svg.replace("</svg>", `${palette}</svg>`);
await mkdir(dirname(OUTPUT), { recursive: true });
await writeFile(OUTPUT, outputSvg, "utf8");
console.log(`wrote ${OUTPUT} with ${partCount} physical parts`);
