// Voronoi pond. (See ANIMATION.md for the full picture.)
//
// Every word, lily pad and duck on the page is built out of a few tens of
// thousands of nodes. Each frame, all on the GPU:
//   1. a damped wave simulation (two layers: fast and slow ripples) is pushed
//      by the cursor and the ducks' paddling feet, and a mouse wake field is
//      splatted along the cursor's path,
//   2. each node is pulled toward its home by a spring, wanders on Perlin noise,
//      gets kicked by turbulence wherever the wake is strong, and (pads, ducks,
//      letters) sways with the waves; background nodes also live, die and respawn,
//   3. the nodes are scattered into a css-pixel grid (up to SLOTS per cell, so
//      dense small text keeps every seed) and a jump-flood pass finds each
//      cell's nearest node,
//   4. a full-screen shader takes the two closest nodes per pixel and draws the
//      exact cells, lily pads as overlapping round leaves, and the ripples as
//      a mosaic of small cells that light up along wave crests.
//
// Distances are power distances (|p - node|^2 - weight), so the diagram is a
// weighted Voronoi with straight edges. A node's weight is 0 while alive and
// strongly negative while being born or dying, which shrinks its cell to
// nothing so births and deaths animate smoothly instead of popping.
//
// Node kinds: 0 = background (water or lily pad), 1 = halo around plain text,
// 2 = plate inside a link box (can turn orange), 3 = glyph ink, 4 = duck cell.

import * as THREE from "three/webgpu";
import {
  Fn,
  If,
  uniform,
  instanceIndex,
  instancedArray,
  storage,
  hash,
  atomicAdd,
  atomicStore,
  atomicMin,
  uint,
  float,
  int,
  vec2,
  vec3,
  vec4,
  mx_noise_vec3,
  mx_noise_float,
  screenCoordinate,
  select,
  max,
  min,
  exp,
  smoothstep,
  mix,
  clamp,
  length,
  dot,
  floor,
  fract,
  abs,
  normalize,
  ivec4,
  Loop,
  Break,
  round,
  sin,
  cos,
  atan,
  sign,
  mod,
  uniformArray,
  varying,
  positionGeometry,
} from "three/tsl";

const MAX_NODES = 1 << 17;
const MAX_GROUPS = 64;
const FIELD_CELL = 8; // css px per wake-field cell
const WAKE_RADIUS = 60; // css px
// In 2x2-cell blocks (reach ~60px), with an extra final step-1 pass ("JFA+1")
// to fix the jump flood's occasional wrong picks.
const JFA_STEPS = [16, 8, 4, 2, 1, 1];
const SLOTS = 4; // seeds remembered per grid cell

const KIND_BG = 0;
const KIND_HALO = 1;
const KIND_PLATE = 2;
const KIND_INK = 3;
const KIND_DUCK = 4;

const PAD_CLUSTER = 0.12; // pads grow where a slow noise field exceeds this (higher = fewer pads)
const FLOWER_FRACTION = 0.2; // share of lily pads that have a flower
const MERGE_FRACTION = 0.22; // share of 2x2 background patches whose pads merge into one big pad
// Ripples are a real (damped) wave simulation on a grid over the viewport,
// with two layers: fast waves (quick cursor strokes) and slow waves (ducks and
// slow strokes), since real ripples aren't all one speed. Anything moving
// faster than its layer's waves trails a V wake; the water is drawn as a fine
// mosaic of small cells that swell and brighten on crests (see shade()).
const WAVE_CELL = 4; // css px per wave-grid cell
const WAVE_SPEED = [220, 30]; // css px/s: fast, slow layer
const WAVE_DAMP = [1.0, 0.45]; // 1/s (fast ripples halve in ~1.4s, slow ones ~3s, plus spreading)
const WAVE_FREQ = [4, 0.9];
const WAVE_VISC = [25, 8];
const WAVE_LEVEL = [7.5, 2]; // rad/s, how fast an offset water level bobs back // css px^2/s, kills grid-scale noise // Hz the movers' pushes pulse at (sets the wavelength)
const CURSOR_PUSH = 7; // wave push of a cursor stroke (scaled by speed)
// A click (or tap) makes a big splash: a wide push that sends out a few rings.
const MAX_DROPS = 4; // splashes in flight at once
const DROP_PUSH = [110, 8]; // fast, slow layer
const DROP_SIGMA = 16; // css px, width of the splash
const DROP_FREQ = 3; // Hz the fast push swings at (a train of a few rings)
const DROP_TIME = 0.8; // s, how long the fast push lasts (the slow one: a quarter)
// A tiny drop (e.g. a dragonfly touching down on a lily pad): one quick,
// soft push into the slow layer only, so a gentle ring rises and fades
// nearby instead of racing across the pond. It pushes along a circle (the
// pad's rim), so the ring shows at once rather than surfacing from under
// the pad half a second later. Passed as strength -(radius + amount), with
// the radius in whole px and the amount in [0, 1).
const TINY_DROP = { sigma: 6, push: 8, time: 0.2 };
const PAD_SWAY = 60; // how far pads sway with passing waves (px per unit slope)
const TEXT_SWAY = 18; // how far letters' cells sway with passing waves (px per unit slope)
const TEXT_SWAY_MAX = 2; // css px, cap on a letter cell's sway (keeps letters legible)
const DUCK_PADDLE = 6; // how hard its feet paddle
const DUCKLING_PADDLE = 1.2; // (per unit of slow-layer stiffness, which is ~5.7x a duck's for its smaller feet)
const FACET = 9; // css px, size of the water mosaic's cells
// Ducks arrive one at a time, spaced so that a quiet pond has about
// DUCKS_IDLE of them at once (2-4, rarely 5, on any screen size: the spacing
// scales with how long a duck takes to cross the view). Ducks that someone's
// waves keep in the pond stay longer while more keep arriving, up to
// MAX_DUCKS.
const MAX_DUCKS = 12; // big ducks
const DUCKS_IDLE = 2.5;
// Ducklings: about half the ducks bring a brood of 3-4 that swim in a line
// behind them. They live in the last duck slots (slot >= FIRST_CHICK) and
// are drawn, shoved and pushed by waves just like ducks.
// (Plenty of slots, so one duck can gather a long line of them; the shaders
// only pay for slots in use.)
const MAX_DUCKLINGS = 64;
// Raccoons swim through now and then (only while there are ducklings to
// hunt): they stalk the nearest duckling, crouch, lunge, and swallow it
// whole, getting a little rounder for each; full after RACCOON_FULL they
// paddle off. Ducklings flee them and ducks steer clear. Their slots sit
// between the big ducks' and the ducklings'.
const MAX_RACCOONS = 2;
const FIRST_COON = MAX_DUCKS;
const FIRST_CHICK = MAX_DUCKS + MAX_RACCOONS;
const DUCK_SLOTS = FIRST_CHICK + MAX_DUCKLINGS;
const RACCOON_SCALE = 1.8; // (the raccoon is ~70 units long, ~125 css px)
const RACCOON_EVERY = [50, 100]; // s between raccoons (the first comes after 25-45 s)
const RACCOON_SPEED = 0.75; // cruising pace (x DUCK_SPEED): slower than a duck
const RACCOON_STALK = 1.6; // swimming after a duckling (x DUCK_SPEED): gains on a duck, not on a fleeing duckling
const RACCOON_NOTICE = 230; // css px: it goes after a duckling this close
const RACCOON_POUNCE = 80; // css px from mouth to duckling when it crouches to lunge
const RACCOON_LUNGE = 6; // lunge speed (x DUCK_SPEED)
const RACCOON_CATCH = 19; // css px from its mouth: caught
const RACCOON_FULL = 3; // ducklings until it's full and leaves
const RACCOON_GROW = 0.07; // how much rounder it gets per duckling
const RACCOON_PADDLE = 7;
// Dragonflies dart about over the pond (quick straight darts, dead stops to
// hover and pivot), now and then settle on a lily pad, sit a while with
// their wings spread flat, and lift off again; a ripple marks each touch
// down and take-off, and a pad that moves under one (a ripple, a duck, the
// cursor's wake) sends it off at once. They fly over everything.
const MAX_FLIES = 3;
const FLY_EVERY = [14, 32]; // s between arrivals (the first after 5-10 s)
const FLY_PERCH = [4, 14]; // s it sits on a pad, if left alone
const FLY_SPLASH = 0.1; // the gentle ripple it makes landing and taking off (a tiny drop around its pad's rim, see TINY_DROP)
const FLY_SPOOK = 16; // css px/s: a pad moving this fast sends it off
const FLY_SHIFT = 5; // css px: ...or moved this far since it landed
const FLY_ALT = 24; // css px it flies above the water (sets its shadow)
const FLY_SIZE = 1.3; // (its shapes are drawn in units of this many css px)
const DUCKLING_FEAR = 115; // css px: ducklings flee a raccoon this close
const DUCK_WARY = 170; // css px: ducks turn away from a raccoon this close
const BROOD_CHANCE = 0.5;
const CHICK = 0.46; // a duckling's size relative to a duck
const DUCKLING_WAVE_PUSH = 1.5; // waves push a duckling this many times harder than a duck
const DUCKLING_LOST = 70; // css px from its place in line before a duckling loses the line
const DUCKLING_SCATTER = 45; // css px/s of wave drift that scatters a duckling at once
const DUCKLING_RUSH = 4.2; // how fast a lost duckling hurries back (x DUCK_SPEED)
const DUCKLING_JOIN = 24; // css px from its new place when it joins a line
const DUCK_SCALE = 2; // the duck is ~38 css px long at scale 1
const DUCK_R = 24 * DUCK_SCALE; // rough radius of a duck, css px
const DUCK_SPEED = 38; // average swimming speed, css px/s
const DUCK_WAVE_PUSH = 600; // how hard waves push a duck (per unit of wave energy flux)
// Duck cell parts (attr.w); +10 marks cells that ride on the bobbing head,
// +20 ones on a wagging tail. (Surface code = part + 3, see nodeUpdate.)
const PART_BODY = 1;
const PART_BILL = 2;
const PART_CHICK = 3; // a duckling's (yellow) body
const PART_FUR = 4; // a raccoon's grey fur
const PART_MASK = 5; // its dark mask, tail rings and nose
const PART_PALE = 6; // its pale muzzle and ears

// A duck, top-down, is a handful of big weighted cells drawn like the lily
// pads (see padShader): a tail, two pairs of body cells (the seam down the
// middle reads as folded wings), a head over the front of the body, and a
// bill poking out ahead of it. In its own frame: x forward, y sideways, css
// px; r is the cell's radius (its power weight is r^2).
const duckCells = (() => {
  const S = DUCK_SCALE;
  return [
    { x: -17.5, y: 0, r: 4.4, part: PART_BODY },
    { x: -10.5, y: -4.6, r: 6, part: PART_BODY },
    { x: -10.5, y: 4.6, r: 6, part: PART_BODY },
    { x: -2, y: -5.4, r: 6.4, part: PART_BODY },
    { x: -2, y: 5.4, r: 6.4, part: PART_BODY },
    { x: 6, y: 0, r: 7, part: PART_BODY + 10 },
    { x: 15.2, y: 0, r: 4.4, part: PART_BILL + 10 },
  ].map((c) => ({ x: c.x * S, y: c.y * S, r: c.r * S, part: c.part }));
})();
// A duckling: a round fluffy body, a big head and a little bill. (Split
// into more cells, like the duck, it read as a scatter of petals at this
// size.)
const chickCells = (() => {
  const S = DUCK_SCALE * CHICK;
  return [
    { x: -4, y: 0, r: 8.8, part: PART_CHICK },
    { x: 8.5, y: 0, r: 7.6, part: PART_CHICK + 10 },
    { x: 16.5, y: 0, r: 3.7, part: PART_BILL + 10 },
  ].map((c) => ({ x: c.x * S, y: c.y * S, r: c.r * S, part: c.part }));
})();
// A raccoon, swimming: a chubby round back, a big head and a bushy tail of
// alternating fur and dark rings that floats out behind and wags. The face
// is built from the front: two plain dark mask patches and a short pale
// muzzle with a button nose, in front of a smaller crown with a big round
// pale ear at each back corner. (Cells inside a bigger one lose the power
// diagram and shrink to slivers, so the patches can't sit on top of the
// crown; patches poking out past it looked like horns; pupils with glints
// in the patches didn't match the simple blob style. Tail rings narrower
// than their radius, so it reads as one striped tail, not a row of beads; a
// smaller head read as a caterpillar.)
const coonCells = (() => {
  const S = RACCOON_SCALE;
  const T = 20; // (tail)
  const H = 10; // (head)
  return [
    { x: -39.2, y: 0, r: 5.4, part: PART_MASK + T },
    { x: -33.6, y: 0, r: 6.8, part: PART_FUR + T },
    { x: -27.6, y: 0, r: 7.4, part: PART_MASK + T },
    { x: -21.5, y: 0, r: 7.2, part: PART_FUR + T },
    { x: -15.5, y: 0, r: 6.2, part: PART_MASK + T },
    { x: -5, y: 0, r: 8.8, part: PART_FUR },
    { x: 6.8, y: 0, r: 7.9, part: PART_FUR + H },
    { x: 2.6, y: -8.4, r: 4.3, part: PART_PALE + H },
    { x: 2.6, y: 8.4, r: 4.3, part: PART_PALE + H },
    { x: 13.4, y: -4.5, r: 4.3, part: PART_MASK + H },
    { x: 13.4, y: 4.5, r: 4.3, part: PART_MASK + H },
    { x: 17.6, y: 0, r: 3.7, part: PART_PALE + H },
    { x: 20.9, y: 0, r: 2.1, part: PART_MASK + H },
  ].map((c) => ({ x: c.x * S, y: c.y * S, r: c.r * S, part: c.part }));
})();
const COON_MOUTH = 20 * RACCOON_SCALE; // css px ahead of its centre
const COON_TAIL = -12 * RACCOON_SCALE; // where its tail starts to wag
const isChickSlot = (i) => i >= FIRST_CHICK;
const isCoonSlot = (i) => i >= FIRST_COON && i < FIRST_CHICK;
const slotCells = (i) => (isChickSlot(i) ? chickCells : isCoonSlot(i) ? coonCells : duckCells);
// How far a duck's cells reach ahead / behind (rx) and to the side (ry),
// and the box around them along its length (mid: its centre, half: half
// its length; a raccoon's long tail puts it well behind the raccoon).
const slotExtent = (i) => {
  const cells = slotCells(i);
  const back = Math.max(...cells.map((c) => c.r - c.x));
  const front = Math.max(...cells.map((c) => c.x + c.r));
  return {
    rx: Math.max(...cells.map((c) => Math.abs(c.x) + c.r)),
    ry: Math.max(...cells.map((c) => Math.abs(c.y) + c.r)),
    back,
    front,
    mid: (front - back) / 2,
    half: (front + back) / 2,
  };
};

const PALETTES = {
  light: {
    water: "#2a6f97",
    waterLight: "#4f97bd",
    waterDeep: "#123f5c",
    ripple: "#5d9fc4",
    rippleRim: "#9ccbe2",
    pad: "#5e9f45",
    padLight: "#79b45a",
    padRim: "#3f7c2f",
    petal: "#fdf0f5",
    petalDeep: "#f2a2c6",
    text: "#f6f1e3",
    deep: "#10283a",
    accent: "#ffb82b",
    duckling: "#ffd94a",
    raccoon: "#aaa49b",
    raccoonDark: "#45413d",
    raccoonPale: "#efe9de",
    fly: "#ec5b2c",
    flyWing: "#f1f7ff",
  },
  dark: {
    water: "#0f3149",
    waterLight: "#20577a",
    waterDeep: "#061825",
    ripple: "#1d4f6f",
    rippleRim: "#3f7ea3",
    pad: "#3d7734",
    padLight: "#4f8c3f",
    padRim: "#285622",
    petal: "#efdce5",
    petalDeep: "#d77fa8",
    text: "#efe9da",
    deep: "#081a26",
    accent: "#ffb82b",
    duckling: "#f2cf4e",
    raccoon: "#8f8a83",
    raccoonDark: "#302d2b",
    raccoonPale: "#ddd6c9",
    fly: "#d9552d",
    flyWing: "#d6e4f0",
  },
};

const root = document.documentElement;
const params = new URLSearchParams(location.search);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const darkScheme = matchMedia("(prefers-color-scheme: dark)");
const narrow = matchMedia("(max-width: 600px)");
// Each visit gets its own pond: this picks where the lily pad colonies grow,
// which patches merge into big pads and every node's jitter. Fixed for the
// whole visit, so a resize keeps the same pond. (?seed=N to repeat one; 0 is
// the pond every visit used to get.)
const POND_SEED = params.has("seed") ? (parseInt(params.get("seed"), 10) >>> 0) % 100000 : Math.floor(Math.random() * 100000);
// (The colony noise is read on its own slice per seed.)
const PAD_NOISE_Z = 3.7 + POND_SEED * 1.618;

setupDevMenu();

main().catch((err) => {
  console.warn("Voronoi effect disabled:", err);
  root.classList.remove("fx-on");
});

async function main() {
  if (params.has("nofx") || !navigator.gpu) return;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return;

  const canvas = document.getElementById("fx");
  const gpuTime = params.has("gputime");
  // The per-pixel grid outgrows WebGPU's default 128MB buffer limit on big
  // screens (4K at 1x), so ask for whatever the adapter allows.
  const requiredLimits = {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  };
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: false, depth: false, trackTimestamp: gpuTime, requiredLimits });
  await renderer.init();
  if (!renderer.backend.isWebGPUBackend) return;

  // GPU-only storage buffers (see gpuArray) are created here at full size,
  // empty (WebGPU zero-fills them), instead of from three's CPU array.
  const device = renderer.backend.device;
  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = (desc) => {
    const gpuOnly = /^gpu-only:(\d+)$/.exec(desc.label || "");
    if (!gpuOnly) return createBuffer(desc);
    const buffer = createBuffer({ label: desc.label, size: +gpuOnly[1], usage: desc.usage });
    // (three fills its 1-item array in through a mapping; nothing to upload.)
    const scratch = new ArrayBuffer(desc.size);
    buffer.getMappedRange = () => scratch;
    buffer.unmap = () => {};
    return buffer;
  };
  // A storage buffer only the GPU writes. three keeps a CPU copy of every
  // storage buffer and uploads it mapped-at-creation, and Chrome keeps that
  // mapping's shared memory for the buffer's whole life, in both the page and
  // the GPU process: ~4 copies of each buffer, which made the tab take ~750MB.
  // So three gets a 1-item array, and the real buffer is made above. (WGSL
  // storage arrays are runtime-sized, so the shaders see all of it.)
  const gpuArray = (count, type) => {
    const node = instancedArray(1, type);
    node.value.name = `gpu-only:${count * node.value.array.byteLength}`;
    return node;
  };

  // (The full-screen pass covers everything; the lily pad pass draws over it.)
  renderer.autoClear = false;
  const padCamera = new THREE.OrthographicCamera();

  // Colors are authored and blended in sRGB so the canvas matches the CSS.
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;

  await Promise.all([
    document.fonts.load('400 64px "Fjalla One"'),
    document.fonts.load('700 16px "Lato"'),
  ]).catch(() => {});
  await document.fonts.ready;

  // ---------------------------------------------------------------- uniforms

  const uTime = uniform(0);
  const uDt = uniform(1 / 60);
  const uFrame = uniform(0, "int");
  const uScroll = uniform(new THREE.Vector2());
  const uMouse = uniform(new THREE.Vector2(-1e5, -1e5));
  const uMousePrev = uniform(new THREE.Vector2(-1e5, -1e5));
  const uMouseVel = uniform(new THREE.Vector2());
  const uIntensity = uniform(0);
  const uAmbient = uniform(reduceMotion.matches ? 0.2 : 1);
  const uDpr = uniform(1);
  const uW = uniform(1, "int"); // jump-flood grid size (css px)
  const uH = uniform(1, "int");
  const uFW = uniform(1, "int");
  const uFH = uniform(1, "int");
  const uNW = uniform(1, "int"); // block grid (jump flood, neighbour lists): one entry per 2x2 cells
  const uNH = uniform(1, "int");
  // Ducks (viewport px): xy position, zw unit heading. Parked far away when idle.
  // (All duck slots: the big ducks, then the ducklings.)
  const duckData = Array.from({ length: DUCK_SLOTS }, () => new THREE.Vector4(-1e5, -1e5, 1, 0));
  const uDucks = uniformArray(duckData, "vec4");
  const duckVel = Array.from({ length: DUCK_SLOTS }, () => new THREE.Vector4());
  const uDuckVel = uniformArray(duckVel, "vec4"); // xy velocity, z body bend, w size (1 = normal)
  // The slots of the ducks in the water, packed (x slot, y kind: 0 duck, 1
  // duckling, 2 raccoon): the shaders loop over just these, so a big pool of
  // slots costs nothing until ducks fill it.
  const duckList = Array.from({ length: DUCK_SLOTS }, () => new THREE.Vector4());
  const uDuckList = uniformArray(duckList, "vec4");
  const uDuckCount = uniform(0, "int");
  // Calls body once per duck in the water with its slot, its size and
  // pick(duck value, duckling value, raccoon value).
  const forEachDuck = (body) =>
    Loop({ start: int(0), end: uDuckCount, type: "int", condition: "<", name: "di" }, ({ di }) => {
      const info = uDuckList.element(di);
      const chick = info.y.greaterThan(0.5).and(info.y.lessThan(1.5));
      const coon = info.y.greaterThan(1.5);
      const slot = int(info.x);
      body({ slot, size: uDuckVel.element(slot).w, pick: (a, b, c) => select(coon, float(c), select(chick, float(b), float(a))) });
    });
  // Dragonflies (one entry per slot). A: viewport xy, heading, height above
  // the water (css px). B: x the lily pad it's sitting on (node id, -1 =
  // flying), yz its spot on that pad (from the pad's centre; the pass draws
  // it on the pad wherever the pad is), w how hard its wings beat (0 still,
  // 1 flying; -1 = slot unused).
  const flyA = Array.from({ length: MAX_FLIES }, () => new THREE.Vector4());
  const flyB = Array.from({ length: MAX_FLIES }, () => new THREE.Vector4(-1, 0, 0, -1));
  const uFlyA = uniformArray(flyA, "vec4");
  const uFlyB = uniformArray(flyB, "vec4");
  // What each dragonfly asks the GPU about pads (see flyProbe): xy a
  // viewport point to look for a pad under (off screen = none), z a pad to
  // watch (node id, -1 = none).
  const flyAsk = Array.from({ length: MAX_FLIES }, () => new THREE.Vector4(-1e5, -1e5, -1, 0));
  const uFlyAsk = uniformArray(flyAsk, "vec4");
  // ...and its answers, read back to the CPU: [2i] the pad found under the
  // point (id or -1, document xy, radius), [2i+1] the watched pad (document
  // xy, speed, its id, or -2 if it's no longer a pad, -1 if none asked).
  const flyOut = instancedArray(MAX_FLIES * 2, "vec4");
  // Wave grid (viewport-sized, anchored to the document so waves scroll
  // with the page): size, the document position of cell (0,0)'s corner, how
  // many cells the anchor moved since last frame, the per-substep timestep
  // and (c*dt/cell)^2 per layer, and the cursor's segment and push strengths.
  const uWW = uniform(1, "int");
  const uWH = uniform(1, "int");
  const uWaveOrigin = uniform(new THREE.Vector2());
  const uWaveShiftX = uniform(0, "int");
  const uWaveShiftY = uniform(0, "int");
  const uWaveDt = uniform(1 / 240);
  const uWaveK = uniform(new THREE.Vector2());
  const uWaveVisc = uniform(new THREE.Vector2());
  const uWaveLevel = uniform(new THREE.Vector2());
  const uWaveReset = uniform(1);
  const uCurA = uniform(new THREE.Vector2(-1e5, -1e5));
  const uCurB = uniform(new THREE.Vector2(-1e5, -1e5));
  const uCurPush = uniform(new THREE.Vector2()); // fast layer, slow layer
  const uCurSteady = uniform(0);
  // Click splashes: document xy, start time (uTime), strength (0 = unused).
  const dropData = Array.from({ length: MAX_DROPS }, () => new THREE.Vector4(-1e5, -1e5, -1e5, 0));
  const uDrops = uniformArray(dropData, "vec4");
  // Ripple mosaic: a grid of FACET-px cells (document-anchored) over the view.
  const uFacetX0 = uniform(0, "int"); // document cell of the grid's corner
  const uFacetY0 = uniform(0, "int");
  const uFacetW = uniform(1, "int");
  const uFacetH = uniform(1, "int");
  const palette =Object.fromEntries(Object.keys(PALETTES.light).map((k) => [k, uniform(new THREE.Vector3())]));
  const {
    water: uWater,
    waterLight: uWaterLight,
    waterDeep: uWaterDeep,
    pad: uPad,
    padLight: uPadLight,
    padRim: uPadRim,
    ripple: uRipple,
    rippleRim: uRippleRim,
    petal: uPetal,
    petalDeep: uPetalDeep,
    text: uText,
    deep: uDeep,
    accent: uAccent,
    duckling: uDuckling,
    raccoon: uRaccoon,
    raccoonDark: uRaccoonDark,
    raccoonPale: uRaccoonPale,
    fly: uFly,
    flyWing: uFlyWing,
  } = palette;

  const applyPalette = () => {
    const pal = darkScheme.matches ? PALETTES.dark : PALETTES.light;
    for (const [key, u] of Object.entries(palette)) {
      const n = parseInt(pal[key].slice(1), 16);
      u.value.set(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
    }
  };
  applyPalette();
  darkScheme.addEventListener("change", applyPalette);

  // ----------------------------------------------------------------- buffers

  // pos: xy position (document px), zw velocity
  // home: xy rest position, z group, w kind
  // attr: x bar coordinate (0 bottom..1 top of link box), y spacing, z/w random
  // life (GPU-written): x power weight, y lifecycle index, z paper tint, w life phase
  // site (GPU-written): xy position, z power weight -- everything the diagram
  //       needs per candidate, packed so each lookup is a single read
  const posBuf = instancedArray(MAX_NODES, "vec4");
  const homeBuf = instancedArray(MAX_NODES, "vec4");
  const attrBuf = instancedArray(MAX_NODES, "vec4");
  const lifeBuf = gpuArray(MAX_NODES, "vec4");
  const siteBuf = gpuArray(MAX_NODES, "vec4");
  // look (GPU-written): everything the pixel shader needs about a node, so it
  // reads one vec4 instead of home/attr/life/group.
  //   x surface: 0 water, 1 lily pad, 2..2.9 glyph (fraction = hover darkening), 3 orange bar, 4 duck body, 5 duck bill, 6 ripple
  //   y spacing, z random, w water tint (water/pad) | tone (glyph) | strength (ripple)
  const lookBuf = gpuArray(MAX_NODES, "vec4");
  // group: [2g] = (hover, darkens-on-hover, restTone, hoverTone), [2g+1] = bar (restLo, restHi, hoverLo, hoverHi)
  const groupBuf = instancedArray(MAX_GROUPS * 2, "vec4");

  const fieldCap =
    Math.ceil(Math.max(screen.width, innerWidth, 2560) / FIELD_CELL + 2) *
    Math.ceil(Math.max(screen.height, innerHeight, 1600) / FIELD_CELL + 2);
  const fieldBuf = gpuArray(fieldCap, "vec4"); // x energy, yz flow velocity, w water height

  const ro = (node, type, count) => storage(node.value, type, count).toReadOnly();
  const siteRO = ro(siteBuf, "vec4", MAX_NODES);
  const lookRO = ro(lookBuf, "vec4", MAX_NODES);

  // Wake-field lookup for a viewport-space point; zero outside the viewport.
  const sampleField = (buf, vp) => {
    const fx = int(floor(vp.x.div(FIELD_CELL)));
    const fy = int(floor(vp.y.div(FIELD_CELL)));
    const inside = fx
      .greaterThanEqual(0)
      .and(fx.lessThan(uFW))
      .and(fy.greaterThanEqual(0))
      .and(fy.lessThan(uFH));
    const cell = max(min(fy, uFH.sub(1)), int(0)).mul(uFW).add(max(min(fx, uFW.sub(1)), int(0)));
    return select(inside, buf.element(cell), vec4(0));
  };
  // The water's height (fieldBuf.w), bilinear between field cell centres, so
  // neighbouring small cells (letters) get smoothly varying slopes instead of
  // 8px steps.
  const fieldHeight = (vp) => {
    const g = vp.div(FIELD_CELL).sub(0.5);
    const g0 = floor(g);
    const t = g.sub(g0);
    const at = (dx, dy) => sampleField(fieldBuf, g0.add(vec2(dx + 0.5, dy + 0.5)).mul(FIELD_CELL)).w;
    return mix(mix(at(0, 0), at(1, 0), t.x), mix(at(0, 1), at(1, 1), t.x), t.y);
  };

  // ------------------------------------------------- compute: water waves

  // Height field per cell, ping-ponged between two buffers: x/y fast layer
  // (height now, height last step), z/w slow layer.
  const waveCap =
    Math.ceil(Math.max(screen.width, innerWidth, 2560) / WAVE_CELL + 12) *
    Math.ceil(Math.max(screen.height, innerHeight, 1600) / WAVE_CELL + 12);
  const waveA = gpuArray(waveCap, "vec4");
  const waveB = gpuArray(waveCap, "vec4");
  // Just the two heights (written by every step, whichever buffer it
  // lands in), for everything that only looks at the water.
  const waveView = gpuArray(waveCap, "vec2");
  // Lily pad colonies soak waves up: extra damping per cell. Pads never move,
  // so this is only recomputed when the grid is re-anchored (scroll/resize).
  const wavePads = gpuArray(waveCap, "float");
  const wavePadPass = Fn(() => {
    const idx = int(instanceIndex);
    const c = uWaveOrigin.add(vec2(float(idx.mod(uWW)), float(idx.div(uWW))).add(0.5).mul(WAVE_CELL));
    wavePads.element(idx).assign(smoothstep(PAD_CLUSTER - 0.04, PAD_CLUSTER + 0.16, mx_noise_float(vec3(c.mul(1 / 230), PAD_NOISE_Z))).mul(0.8));
  })().compute(waveCap);
  const WAVE_K = WAVE_FREQ.map((f, i) => (Math.PI * 2 * f) / WAVE_SPEED[i]);

  // One leapfrog step of the damped wave equation (isotropic 9-point
  // Laplacian), for both layers at once. `shifted` re-indexes the grid when
  // the page has scrolled so the waves stay put on the water.
  const waveStep = (src, dst, shifted, view = waveView) =>
    Fn(() => {
      const idx = int(instanceIndex);
      const x = idx.mod(uWW);
      const y = idx.div(uWW);
      const sx = shifted ? x.add(uWaveShiftX) : x;
      const sy = shifted ? y.add(uWaveShiftY) : y;
      const at = (dx, dy) => {
        const qx = sx.add(dx);
        const qy = sy.add(dy);
        const inside = qx.greaterThanEqual(0).and(qx.lessThan(uWW)).and(qy.greaterThanEqual(0)).and(qy.lessThan(uWH));
        return select(inside, src.element(max(min(qy, uWH.sub(1)), int(0)).mul(uWW).add(max(min(qx, uWW.sub(1)), int(0)))), vec4(0));
      };
      const C = at(0, 0).toVar();
      const lap = at(1, 0)
        .add(at(-1, 0))
        .add(at(0, 1))
        .add(at(0, -1))
        .mul(4)
        .add(at(1, 1).add(at(-1, 1)).add(at(1, -1)).add(at(-1, -1)))
        .sub(C.mul(20))
        .div(6)
        .toVar();
      const c = uWaveOrigin.add(vec2(float(x), float(y)).add(0.5).mul(WAVE_CELL));

      // Extra damping: a sponge at the grid's edges (so waves leave the screen
      // instead of bouncing), and lily pad colonies.
      const edge = float(min(min(x, y), min(uWW.sub(1).sub(x), uWH.sub(1).sub(y))));
      const sponge = float(1).sub(smoothstep(0, 6, edge)).mul(8);
      const damp = vec2(WAVE_DAMP[0], WAVE_DAMP[1]).add(sponge).add(wavePads.element(idx));

      // The cursor pushes the water along its path, pulsing, so a stroke
      // sends out trains of ripples (into the fast or slow layer by speed).
      const ab = uCurB.sub(uCurA);
      const t = clamp(dot(c.sub(uCurA), ab).div(max(dot(ab, ab), 1e-4)), 0, 1);
      const dc = c.sub(uCurA.add(ab.mul(t)));
      // (Forces are scaled by each layer's stiffness, c^2/r^2, so a push of 1
      // makes waves of roughly the same height in either layer.)
      const stiff = (r) => vec2(WAVE_SPEED[0] ** 2 / r ** 2, WAVE_SPEED[1] ** 2 / r ** 2);
      // The push has a "Mexican hat" profile across the path (zero on
      // average, sized to each layer's wavelength), so it only makes ripples
      // and never broad swells.
      const sig = WAVE_K.map((k) => Math.SQRT2 / k);
      const d2 = dot(dc, dc);
      const hat = (sg) => float(1).sub(d2.div(2 * sg * sg)).mul(exp(d2.div(-2 * sg * sg)));
      // Pulsing makes rings around a slow stroke; a fast one pushes steadily,
      // which leaves a clean V wake (uCurSteady blends between the two).
      const pulse = mix(vec2(sin(uTime.mul(Math.PI * 2 * WAVE_FREQ[0])), sin(uTime.mul(Math.PI * 2 * WAVE_FREQ[1]))), vec2(1), uCurSteady);
      const force = uCurPush.mul(pulse).mul(vec2(hat(sig[0]), hat(sig[1]))).mul(vec2(stiff(sig[0]).x, stiff(sig[1]).y)).toVar();
      // Click splashes: a wide push (the same zero-mean profile) that sends a
      // train of big rings out across the pond, with slow rings lingering
      // where it landed.
      const dropForce = vec2(...WAVE_SPEED.map((cs, l) => (cs * cs * DROP_PUSH[l]) / DROP_SIGMA ** 2));
      for (let i = 0; i < MAX_DROPS; i++) {
        const drop = uDrops.element(i);
        // Fast layer: a swing that dies away (a big lead crest, then smaller
        // rings); slow layer: one soft push.
        const age = uTime.sub(drop.z).toVar();
        const fade = float(1).sub(clamp(age.div(DROP_TIME), 0, 1));
        const swing = sin(age.mul(Math.PI * 2 * DROP_FREQ)).mul(fade.mul(fade));
        const soft = sin(clamp(age.div(DROP_TIME / 4), 0, 1).mul(Math.PI));
        const tiny = drop.w.lessThan(0);
        const sigma = select(tiny, float(TINY_DROP.sigma), float(DROP_SIGMA));
        // (A tiny drop's profile runs across its circle, not out from the
        // middle.)
        const across = length(c.sub(drop.xy)).sub(select(tiny, floor(abs(drop.w)), float(0)));
        const q = across.mul(across).div(sigma.mul(sigma).mul(2));
        const shape = float(1).sub(q).mul(exp(q.negate())).mul(select(tiny, fract(abs(drop.w)), drop.w));
        const tinyPush = sin(clamp(age.div(TINY_DROP.time), 0, 1).mul(Math.PI)).mul((WAVE_SPEED[1] ** 2 * TINY_DROP.push) / TINY_DROP.sigma ** 2);
        force.addAssign(select(tiny, vec2(0, tinyPush), dropForce.mul(vec2(swing, soft))).mul(shape));
      }
      // Ducks: the body presses a dent that travels with it (a moving dent
      // makes a wake), and the paddling feet behind it pulse.
      // (Ducklings: smaller feet, a quicker, gentler paddle.)
      const slowStiff = (r) => WAVE_SPEED[1] ** 2 / r ** 2;
      forEachDuck(({ slot, size, pick }) => {
        // (Feet: how far behind its centre, how big; a raccoon paddles all
        // four under its middle.)
        const DS = DUCK_SCALE;
        const CS = DUCK_SCALE * CHICK;
        const RS = RACCOON_SCALE;
        const duck = uDucks.element(slot);
        const at2 = duck.xy.add(uScroll);
        const feet = c.sub(at2.sub(duck.zw.mul(pick(11 * DS, 7 * CS, 5 * RS))));
        const gFeet = exp(dot(feet, feet).div(pick(-((6 * DS) ** 2), -((6 * CS) ** 2), -((8 * RS) ** 2))));
        const paddle = sin(uTime.mul(pick(1, 1.6, 0.75).mul(Math.PI * 2 * WAVE_FREQ[1])).add(float(slot).mul(2.1)));
        const push = gFeet.mul(paddle).mul(pick(DUCK_PADDLE * slowStiff(6 * DS), DUCKLING_PADDLE * slowStiff(6 * CS), RACCOON_PADDLE * slowStiff(8 * RS))).mul(min(size, 1));
        force.addAssign(vec2(0, push));
      });

      const h = vec2(C.x, C.z);
      const hPrev = vec2(C.y, C.w);
      const next = h
        .add(h.sub(hPrev).mul(max(float(1).sub(damp.mul(uWaveDt)), 0)))
        .add(vec2(lap.x, lap.z).mul(uWaveK))
        // A gentle pull back to level: water can't hold a dent or bulge, so
        // any leftover offset just bobs once and fades instead of lingering
        // as a broad blotch.
        .sub(h.mul(uWaveLevel))
        // Viscosity: damps grid-scale chop (which reads as static) while
        // barely touching real ripples.
        .add(vec2(lap.x.sub(lap.y), lap.z.sub(lap.w)).mul(uWaveVisc))
        .add(force.mul(uWaveDt.mul(uWaveDt)))
        .toVar();
      dst.element(idx).assign(select(uWaveReset.greaterThan(0.5), vec4(0), vec4(next.x, h.x, next.y, h.y)));
      view.element(idx).assign(select(uWaveReset.greaterThan(0.5), vec2(0), next));
    })().compute(waveCap);
  // [from A, from B] x [first step of the frame (re-anchors), later steps]
  const waveSteps = [
    [waveStep(waveA, waveB, true), waveStep(waveA, waveB, false)],
    [waveStep(waveB, waveA, true), waveStep(waveB, waveA, false)],
  ];
  let waveCur = 0; // which buffer holds the latest heights

  // Waves push the ducks. A floating thing is carried the way waves travel,
  // and for the wave equation that's the energy flux, -dh/dt * grad(h). Per
  // duck: that flux (fast layer only: the cursor's waves; the slow layer is
  // mostly the duck's own paddling) averaged over a disc around it, read
  // back to the CPU, which steers the duck (see updateDucks).
  const duckWave = instancedArray(DUCK_SLOTS, "vec4"); // xy flux, z energy
  const duckWavePass = (buf) =>
    Fn(() => {
      const duck = uDucks.element(instanceIndex);
      const centre = duck.xy.add(uScroll);
      // (A duckling feels the water over a smaller disc.)
      const R = select(int(instanceIndex).lessThan(FIRST_COON), float(DUCK_R), select(int(instanceIndex).lessThan(FIRST_CHICK), float(DUCK_R * 1.15), float(DUCK_R * CHICK))).toVar();
      const sum = vec4(0).toVar();
      const cell = (qx, qy) => buf.element(max(min(qy, uWH.sub(1)), int(0)).mul(uWW).add(max(min(qx, uWW.sub(1)), int(0))));
      Loop(81, ({ i }) => {
        const off = vec2(float(i.mod(9)).sub(4), float(i.div(9)).sub(4)).mul(R.div(4.5));
        const w = exp(dot(off, off).div(R.mul(R).mul(-0.49)));
        const u = centre.add(off).sub(uWaveOrigin).div(WAVE_CELL);
        const qx = int(floor(u.x));
        const qy = int(floor(u.y));
        If(qx.greaterThan(0).and(qx.lessThan(uWW.sub(1))).and(qy.greaterThan(0)).and(qy.lessThan(uWH.sub(1))), () => {
          const c = cell(qx, qy).toVar();
          const grad = vec2(cell(qx.add(1), qy).x.sub(cell(qx.sub(1), qy).x), cell(qx, qy.add(1)).x.sub(cell(qx, qy.sub(1)).x)).div(2 * WAVE_CELL);
          const ht = c.x.sub(c.y).div(max(uWaveDt, 1e-4));
          sum.addAssign(vec4(grad.mul(ht.negate()), c.x.mul(c.x), 1).mul(w));
        });
      });
      duckWave.element(instanceIndex).assign(sum.div(max(sum.w, 1e-3)));
    })().compute(DUCK_SLOTS);
  const duckWavePasses = [duckWavePass(waveA), duckWavePass(waveB)];
  const duckPush = Array.from({ length: DUCK_SLOTS }, () => ({ x: 0, y: 0, e: 0 }));
  let duckWaveBusy = false;
  const readDuckWaves = () => {
    if (duckWaveBusy) return;
    duckWaveBusy = true;
    renderer
      .getArrayBufferAsync(duckWave.value)
      .then((ab) => {
        const a = new Float32Array(ab);
        for (let i = 0; i < DUCK_SLOTS; i++) Object.assign(duckPush[i], { x: a[i * 4] || 0, y: a[i * 4 + 1] || 0, e: a[i * 4 + 2] || 0 });
        if (params.has("duckwave")) window.__duckWave = duckPush.map((d) => ({ ...d }));
      })
      .finally(() => (duckWaveBusy = false));
  };

  // Water height per layer (x fast, y slow) at a document point, bilinear,
  // plus each layer's slope.
  const waveAt = (buf, q) => {
    const u = q.sub(uWaveOrigin).div(WAVE_CELL).sub(0.5);
    const i0 = floor(u);
    const f = u.sub(i0);
    const read = (dx, dy) => {
      const qx = int(i0.x).add(dx);
      const qy = int(i0.y).add(dy);
      const inside = qx.greaterThanEqual(0).and(qx.lessThan(uWW)).and(qy.greaterThanEqual(0)).and(qy.lessThan(uWH));
      const w = buf.element(max(min(qy, uWH.sub(1)), int(0)).mul(uWW).add(max(min(qx, uWW.sub(1)), int(0))));
      return select(inside, w.xy, vec2(0));
    };
    const h00 = read(0, 0).toVar();
    const h10 = read(1, 0).toVar();
    const h01 = read(0, 1).toVar();
    const h11 = read(1, 1).toVar();
    const top = mix(h00, h10, f.x);
    const bottom = mix(h01, h11, f.x);
    const height = mix(top, bottom, f.y);
    const dX = mix(h10.sub(h00), h11.sub(h01), f.y).div(WAVE_CELL);
    const dY = bottom.sub(top).div(WAVE_CELL);
    return { height, fastSlope: vec2(dX.x, dY.x), slowSlope: vec2(dX.y, dY.y) };
  };

  // ----------------------------------------------- compute: ripple mosaic

  // The water is drawn as a fine mosaic of small cells (a jittered-grid
  // Voronoi, fixed to the page) that is invisible while calm. Per mosaic cell,
  // once per frame: its centre, swayed by the water's slope (so the mosaic
  // moves with passing waves), and how much it lights up as a crest or darkens
  // as a trough. The pixel shader then only has to find the nearest centre.
  const facetCap =
    (Math.ceil(Math.max(screen.width, innerWidth, 2560) / FACET) + 6) *
    (Math.ceil(Math.max(screen.height, innerHeight, 1600) / FACET) + 6);
  const facetBuf = gpuArray(facetCap, "vec4"); // xy centre (document px), z crest, w trough
  const facetMax = gpuArray(facetCap, "float"); // brightest in its 3x3 (0 = calm, skip)
  const facetUpdate = Fn(() => {
    const idx = int(instanceIndex);
    const cx = uFacetX0.add(idx.mod(uFacetW));
    const cy = uFacetY0.add(idx.div(uFacetW));
    const seed = uint(cx.add(65536)).mul(uint(7919)).add(uint(cy.add(65536)).mul(uint(104729)));
    // (Jitter ±0.25 cell, so a pixel's two nearest centres are always in the
    // 2x2 cells it faces: the pixel shader reads 4 records, not 9.)
    const jit = vec2(hash(seed), hash(seed.add(uint(1)))).sub(0.5).mul(0.5);
    const home = vec2(float(cx), float(cy)).add(0.5).add(jit).mul(FACET);
    const w = waveAt(waveView, home);
    // Where on its wave each layer is here: for a wave h = A cos(phase),
    // h / sqrt(h^2 + (slope/k)^2) is cos(phase), 1 on a crest and -1 in a
    // trough, whatever the wave's height. So crests light up as clean lines
    // of cells, faded by the wave's size A.
    const crestOf = (h, slope, k) => {
      const amp = h.mul(h).add(dot(slope, slope).div(k * k)).sqrt().toVar();
      const cosPhase = h.div(max(amp, 1e-4)).toVar();
      const size = smoothstep(0.12, 0.45, amp);
      return { lit: smoothstep(0.62, 0.94, cosPhase).mul(size), dark: smoothstep(0.62, 0.94, cosPhase.negate()).mul(size) };
    };
    // Both layers together, so cursor ripples and duck wakes interfere
    // (crests that meet add up, a crest meeting a trough cancels out): the
    // combined height, and the sum of each layer's quarter-wave-shifted
    // partner (slope / k) for the phase.
    const quad = w.fastSlope.div(WAVE_K[0]).add(w.slowSlope.div(WAVE_K[1]));
    const both = crestOf(w.height.x.add(w.height.y), quad, 1);
    const sway = w.fastSlope.add(w.slowSlope).mul(-60);
    const swayLen = length(sway);
    const swayed = sway.mul(min(swayLen, FACET * 0.12).div(max(swayLen, 1e-4)));
    // Right under a moving cursor the water just churns (the push pulses
    // there), so that's left calm-looking: rings appear from under it.
    const toCursor = length(home.sub(uCurB));
    const churn = float(1).sub(smoothstep(12, 36, toCursor)).mul(min(uCurPush.x.add(uCurPush.y).mul(2), 1));
    const seen = float(1).sub(churn);
    facetBuf.element(idx).assign(vec4(home.add(swayed), both.lit.mul(seen), both.dark.mul(seen)));
  })().compute(facetCap);
  const facetSpread = Fn(() => {
    const idx = int(instanceIndex);
    const x = idx.mod(uFacetW);
    const y = idx.div(uFacetW);
    const m = float(0).toVar();
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const qx = min(max(x.add(dx), int(0)), uFacetW.sub(1));
        const qy = min(max(y.add(dy), int(0)), uFacetH.sub(1));
        const f = facetBuf.element(qy.mul(uFacetW).add(qx));
        m.assign(max(m, max(f.z, f.w)));
      }
    }
    facetMax.element(idx).assign(m);
  })().compute(facetCap);
  const facetRO = ro(facetBuf, "vec4", facetCap);
  const facetMaxRO = ro(facetMax, "float", facetCap);

  // --------------------------------------------------- compute: wake field

  const fieldUpdate = Fn(() => {
    const idx = int(instanceIndex);
    const c = vec2(float(idx.mod(uFW)).add(0.5), float(idx.div(uFW)).add(0.5)).mul(FIELD_CELL);
    const ab = uMouse.sub(uMousePrev);
    const t = clamp(dot(c.sub(uMousePrev), ab).div(max(dot(ab, ab), 1e-4)), 0, 1);
    const d = c.sub(uMousePrev.add(ab.mul(t)));
    const splat = exp(dot(d, d).div(-WAKE_RADIUS * WAKE_RADIUS)).mul(uIntensity);
    const f = fieldBuf.element(idx);
    const energy = max(f.x.mul(exp(uDt.mul(-1.4))), splat).toVar();
    const flow = mix(f.yz.mul(exp(uDt.mul(-2.2))), uMouseVel, clamp(splat, 0, 1)).toVar();
    // Ducks stir the water too, leaving a gentler wake behind them.
    forEachDuck(({ slot, size, pick }) => {
      const duck = uDucks.element(slot);
      const R = pick(DUCK_R, DUCK_R * CHICK, DUCK_R * 1.15).mul(max(size, 0.05));
      // Centred behind the duck so it doesn't shake its own cells apart.
      const dd = c.sub(duck.xy.sub(duck.zw.mul(R.mul(1.3))));
      const duckSplat = exp(dot(dd, dd).div(R.mul(R).negate())).mul(pick(0.4, 0.25, 0.45));
      energy.assign(max(energy, duckSplat));
      flow.assign(mix(flow, duck.zw.mul(DUCK_SPEED * 6), clamp(duckSplat, 0, 1)));
    });
    // (Nodes read the water's height from here: it saves them a buffer.)
    const wave = waveAt(waveView, c.add(uScroll)).height;
    f.assign(vec4(energy, flow, wave.x.add(wave.y)));
  })().compute(fieldCap);

  // -------------------------------------------------------- compute: nodes

  const nodeUpdate = Fn(() => {
    const P = posBuf.element(instanceIndex);
    const Hm = homeBuf.element(instanceIndex);
    const A = attrBuf.element(instanceIndex);
    const L = lifeBuf.element(instanceIndex);
    const p = P.xy.toVar();
    const v = P.zw.toVar();
    const isBg = Hm.w.lessThan(0.5);
    const isDuck = Hm.w.greaterThan(3.5).and(Hm.w.lessThan(4.5));
    // (A duck cell's slot, and that slot's size: a duckling being swallowed
    // shrinks away, a raccoon gets rounder with each one.)
    const duckIdx = max(int(A.x), int(0));
    const slotSize = select(isDuck, uDuckVel.element(duckIdx).w, float(1));
    const spacing = A.y.mul(max(slotSize, 0.02)); // (never 0: some maths below divides by it)
    const rnd = A.z;

    // Intro: nodes start scattered and lock on in a staggered wave.
    const start = rnd.mul(1.3).add(0.2);
    const ramp = smoothstep(start, start.add(0.9), uTime);

    // Lily pads grow in colonies: a slow noise field over the pond (by base
    // home, so it never moves), with a little randomness at cluster edges.
    const padField = mx_noise_float(vec3(Hm.xy.mul(1 / 230), PAD_NOISE_Z)).add(rnd.sub(0.5).mul(0.25));
    // Some patches of 2x2 background-grid cells merge their small pads into
    // one big one: the patch's chosen node becomes a big pad and the rest
    // stay water.
    const quad = floor(Hm.xy.add(spacing).div(spacing));
    const patch = floor(quad.div(2));
    const patchSeed = uint(int(patch.x).add(4096)).mul(uint(7919)).add(uint(int(patch.y).add(4096)).mul(uint(104729))).add(uint(POND_SEED * 31337));
    const merged = hash(patchSeed).lessThan(MERGE_FRACTION);
    const quadIdx = quad.x.sub(patch.x.mul(2)).add(quad.y.sub(patch.y.mul(2)).mul(2));
    const isLead = quadIdx.equal(floor(hash(patchSeed.add(uint(3))).mul(4)));
    const patchCentre = patch.add(0.5).mul(spacing.mul(2)).sub(spacing);
    const patchPad = mx_noise_float(vec3(patchCentre.div(230), PAD_NOISE_Z)).greaterThan(PAD_CLUSTER);
    const isPadNode = isBg.and(select(merged, isLead.and(patchPad), padField.greaterThan(PAD_CLUSTER)));
    const isBigPad = isPadNode.and(merged);
    const isWater = isBg.and(isPadNode.not());
    // A pad is a disc of radius padR that its power weight padR^2 keeps whole
    // against anything unweighted (water, ducks, text): every bisector with a
    // zero-weight node is at least padR away. Between two pads the edge is a
    // straight power bisector, so colonies pack into rounded Voronoi cells.
    // Sizes vary; big (merged) pads are bigger; a pad near text is limited to
    // the room it has (attr.x, from layout()) so it can't swallow letters.
    // Pads rest near their grid cell's centre (or the patch centre), so
    // neighbours never crowd into split shapes.
    const cellCentre = quad.add(0.5).mul(spacing).sub(spacing);
    const padHome = select(isBigPad, patchCentre, mix(Hm.xy, cellCentre, 0.6));
    const room = A.x.sub(length(padHome.sub(Hm.xy))).sub(4);
    const padR = max(min(spacing.mul(select(isBigPad, float(1.15), mix(float(0.52), float(0.78), fract(rnd.mul(31.7))))), room), 4).toVar();
    // Pads open up from nothing once the nodes have settled.
    const grow = smoothstep(start.add(0.7), start.add(2.2), uTime);

    // Lifecycle (water only; pads are permanent): each life lasts 14-28s and
    // starts with a slow birth (~2-3s) and ends with a slow death. Every new
    // life respawns the node at a random spot near its base home.
    const period = mix(float(14), float(28), A.w);
    const age = uTime.div(period).add(rnd.mul(7));
    const life = floor(age);
    const phase = select(isWater, age.sub(life), float(0.5));
    const alive = smoothstep(0, 0.14, phase).mul(float(1).sub(smoothstep(0.86, 1, phase)));
    const seed = uint(instanceIndex).mul(uint(7919)).add(uint(life).mul(uint(104729)));
    const jitter = vec2(hash(seed), hash(seed.add(uint(1)))).sub(0.5).mul(spacing.mul(1.4));
    const home = select(isPadNode, padHome, Hm.xy.add(select(isBg, jitter, vec2(0))));
    If(isWater.and(life.notEqual(L.y)), () => {
      // The node is invisible at this point (weight at minimum), so teleport.
      p.assign(home);
      v.assign(vec2(0));
    });
    // At most (0.6 * spacing)^2, so the node stays inside its own (shrinking)
    // cell; the shader fades the last sliver out using `alive`.
    const padSize = padR.mul(grow);
    // (Duck cells weigh their radius^2 too: attr.y is a duck cell's radius.)
    const weight = select(
      isWater,
      float(1).sub(alive).mul(spacing.mul(0.6).pow(2)).negate(),
      select(isPadNode, padSize.mul(padSize), select(isDuck, spacing.mul(spacing), float(0))),
    );
    const tint = mx_noise_float(vec3(Hm.xy.mul(0.0035), uTime.mul(0.05)));
    L.assign(vec4(weight, life, tint, phase));

    // Ambient drift: the rest position wanders through a slow Perlin field.
    const drift = mx_noise_vec3(vec3(home.mul(1 / 170), uTime.mul(0.07))).xy;
    const amp = select(isWater, float(15), select(isPadNode, float(6), spacing.mul(0.16))).mul(uAmbient);

    // Duck cells: home is a spot in the duck's moving, rotating frame (head
    // cells bob side to side). They chase it with the duck's own velocity fed
    // forward, so the duck keeps its shape while swimming but can still be
    // knocked apart and reassemble.
    const duck = uDucks.element(duckIdx);
    const duckV = select(isDuck, uDuckVel.element(duckIdx).xy, vec2(0));
    const onHead = A.w.greaterThan(9.5).and(A.w.lessThan(19.5));
    const onTail = A.w.greaterThan(19.5);
    const isChick = duckIdx.greaterThanEqual(FIRST_CHICK);
    const isCoon = duckIdx.greaterThanEqual(FIRST_COON).and(isChick.not());
    const kindVal = (a, b, c) => select(isCoon, float(c), select(isChick, float(b), float(a)));
    // (Ducklings bob their heads quicker; a raccoon sniffs from side to side.)
    const bob = sin(uTime.mul(kindVal(1.4, 2.6, 1.1)).add(A.x.mul(40))).mul(kindVal(1.2 * DUCK_SCALE, 0.9 * DUCK_SCALE * CHICK, 0.8 * RACCOON_SCALE));
    // A raccoon's tail floats out behind and wags in a slow wave that runs
    // down it, swinging more toward the tip.
    const wag = sin(uTime.mul(2.3).add(A.x.mul(1.7)).add(Hm.x.mul(0.05))).mul(Hm.x.negate().add(COON_TAIL).mul(0.11));
    const straight = Hm.xy.add(vec2(0, select(onHead, bob, select(onTail, wag, float(0))))).mul(slotSize);
    // Bend the body along its turn: rotating each cell by an angle
    // proportional to how far forward it is curls head and tail into an arc.
    const bend = straight.x.mul(uDuckVel.element(duckIdx).z);
    const local = vec2(
      straight.x.mul(cos(bend)).sub(straight.y.mul(sin(bend))),
      straight.x.mul(sin(bend)).add(straight.y.mul(cos(bend))),
    );
    const duckHome = duck.xy.add(uScroll).add(duck.zw.mul(local.x)).add(vec2(duck.w.negate(), duck.z).mul(local.y));
    const target = select(isDuck, duckHome, home.add(drift.mul(amp)));

    // The mouse wake feeds the noise map: where the field is hot, nodes get
    // turbulent Perlin kicks plus a push along the cursor's direction.
    const F = sampleField(fieldBuf, p.sub(uScroll));
    const energy = F.x;
    const turb = mx_noise_vec3(vec3(p.mul(0.028), uTime.mul(0.9).add(energy.mul(2)))).xy;

    const k = select(isBg, float(5), select(isDuck, float(30), float(18))).mul(select(isDuck, float(1), ramp));
    const damping = select(isBg, float(3.2), select(isDuck, float(6.5), float(5.2))).mul(max(ramp, 0.3));

    const acc = target
      .sub(p)
      .mul(k)
      .sub(v.sub(duckV).mul(damping))
      .add(turb.mul(energy.mul(select(isDuck, float(1200), float(2000)))))
      .add(F.yz.mul(energy.mul(1.6)))
      .toVar();

    // Ducks shove nearby nodes (including other ducks' cells) out of a
    // duck-shaped ellipse just a few px bigger than the duck itself; pads out
    // of one grown by their radius, so a duck parts them like real ones
    // instead of overlapping them.
    const padReach = select(isPadNode, padSize.mul(0.9), float(0));
    const ext = [slotExtent(0), slotExtent(FIRST_CHICK), slotExtent(FIRST_COON)];
    // (A duckling being swallowed neither shoves nor is shoved: it slips
    // into the raccoon's mouth.)
    const gulped = isDuck.and(slotSize.lessThan(0.98));
    forEachDuck(({ slot, size, pick }) => {
      // (Tail to bill tip and side to side, plus a margin.)
      // (Sizes kept above 0: a NaN here would poison every node for good.)
      const sz = max(size, 0.05);
      const rx = padReach.add(pick(...ext.map((e) => e.half + 4)).mul(sz));
      const ry = padReach.add(pick(...ext.map((e) => e.ry + 4)).mul(sz));
      const other = uDucks.element(slot);
      const away = p.sub(uScroll).sub(other.xy.add(other.zw.mul(pick(...ext.map((e) => e.mid)).mul(sz))));
      const side = vec2(other.w.negate(), other.z);
      const lx = dot(away, other.zw);
      const ly = dot(away, side);
      const inside = float(1).sub(length(vec2(lx.div(rx), ly.div(ry))));
      const normal = normalize(other.zw.mul(lx.div(rx.mul(rx))).add(side.mul(ly.div(ry.mul(ry)))).add(vec2(1e-6, 0)));
      const own = isDuck.and(duckIdx.equal(slot));
      // (Ducklings don't shove a raccoon's cells: ones ahead of a lunge tore
      // its face apart.)
      const otherKind = pick(0, 1, 2);
      const predPrey = isDuck.and(isCoon).and(otherKind.greaterThan(0.5)).and(otherKind.lessThan(1.5));
      const off = own.or(gulped).or(size.lessThan(0.98)).or(predPrey);
      acc.addAssign(select(off, vec2(0), normal.mul(max(inside, 0).mul(select(isPadNode, float(4000), float(2600))))));
    });

    // Pads ride the waves: they sway with the water (drawn offset down its
    // slope, like the water's own back-and-forth under a passing ripple),
    // get a gentle push, and catch the light on crests (see lookBuf below).
    // Ducks bob along a little too. Letters (and their halo / plate cells,
    // so whole letters move together) waver as a ripple passes under them,
    // like text seen through moving water: a capped sway, no push, so they
    // stay legible and settle back as soon as the water calms.
    const isText = isBg.or(isDuck).not();
    const vp = p.sub(uScroll);
    const e = fieldHeight(vp.add(vec2(FIELD_CELL, 0))).toVar();
    const w = fieldHeight(vp.sub(vec2(FIELD_CELL, 0))).toVar();
    const s = fieldHeight(vp.add(vec2(0, FIELD_CELL))).toVar();
    const n = fieldHeight(vp.sub(vec2(0, FIELD_CELL))).toVar();
    const slope = vec2(e.sub(w), s.sub(n)).div(2 * FIELD_CELL).toVar();
    const lift = e.add(w).add(s).add(n).mul(0.25).toVar();
    const swayRaw = slope.mul(select(isDuck, float(-0.35 * PAD_SWAY), select(isBg, float(-PAD_SWAY), float(-TEXT_SWAY)))).toVar();
    const swayLen = length(swayRaw);
    const swell = select(isText, swayRaw.mul(min(swayLen, TEXT_SWAY_MAX).div(max(swayLen, 1e-4))), swayRaw);
    acc.addAssign(slope.mul(select(isBg, float(-900), float(0))));

    v.addAssign(acc.mul(uDt));
    p.addAssign(v.mul(uDt));
    If(isDuck.and(length(target.sub(p)).greaterThan(DUCK_R * 3)), () => {
      // Spawning (or parked): jump straight into formation.
      p.assign(target);
      v.assign(duckV);
    });
    P.assign(vec4(p, v));
    siteBuf.element(instanceIndex).assign(vec4(p.add(swell), weight, select(isWater, alive, select(isPadNode, grow, float(1)))));

    // What this node's cell looks like, resolved once per node per frame.
    const kind = Hm.w;
    const grp = int(Hm.z);
    const G = groupBuf.element(grp.mul(2));
    const GB = groupBuf.element(grp.mul(2).add(1));
    const hover = G.x;
    const inBar = kind
      .greaterThan(1.5)
      .and(kind.lessThan(2.5))
      .and(A.x.greaterThanEqual(mix(GB.x, GB.z, hover)))
      .and(A.x.lessThanEqual(mix(GB.y, GB.w, hover)));
    const isInk = kind.greaterThan(2.5).and(kind.lessThan(3.5));
    const part = A.w.mod(10);
    // (A pad's or duck cell's fraction carries how high the water lifts it,
    // 0.45 = level.)
    const lifted = clamp(lift.mul(0.35).add(0.45), 0.02, 0.9);
    const surface = select(
      isDuck,
      part.add(3).add(lifted), // 4 body / 5 bill / 6 duckling body / 7 raccoon fur / 8 its mask / 9 its muzzle
      select(
        isInk,
        float(2).add(hover.mul(G.y).mul(0.9)),
        select(
          inBar,
          float(3),
          // (A pad's fraction carries how high the water lifts it, 0.45 = level.)
          select(isPadNode, float(1).add(lifted), float(0)),
        ),
      ),
    );
    const extra = select(isInk, mix(G.z, G.w, hover), tint);
    // (For a pad or duck cell, y is its radius.)
    // (For glyph and bar cells, z packs the water's lift into its fraction:
    // floor(rnd * 32) + lifted.)
    const lookRnd = select(isInk.or(inBar), floor(rnd.mul(32)).add(lifted), rnd);
    lookBuf.element(instanceIndex).assign(vec4(surface, select(isPadNode, padR, spacing), lookRnd, extra));
  })().compute(MAX_NODES);

  // --------------------------------------- compute: jump-flood Voronoi grid

  let pixelStage = null;

  // capacity: grid cells (css px); the jump flood, neighbour lists and pad
  // blocks are per 2x2 block, so they get a quarter of that (plus the odd
  // edge row and column).
  const buildPixelStage = (capacity) => {
    const blockCap = Math.ceil(capacity / 4) + 8192;
    const jfaA = gpuArray(blockCap, "int");
    const jfaB = gpuArray(blockCap, "int");
    const slotCount = gpuArray(capacity, "uint").toAtomic();
    const slots = gpuArray(capacity * SLOTS, "int");
    // Per 2x2 block, the lily pad nearest its centre by power distance among
    // those whose disc covers it, packed as (quantized distance << 17 | id)
    // so atomicMin picks it (0xffffffff = none). The pixel shader adds it as
    // a candidate, so a pad is never missed (the block lists can miss a heavy
    // pad whose centre is far off).
    const padBlock = gpuArray(blockCap, "uint").toAtomic();

    const clear = Fn(() => {
      If(int(instanceIndex).lessThan(uNW.mul(uNH)), () => {
        jfaA.element(instanceIndex).assign(int(-1));
        atomicStore(padBlock.element(instanceIndex), uint(0xffffffff));
      });
      atomicStore(slotCount.element(instanceIndex), uint(0));
      for (let j = 0; j < SLOTS; j++) slots.element(int(instanceIndex).mul(SLOTS).add(j)).assign(int(-1));
    })().compute(capacity);

    const seed = Fn(() => {
      const p = posBuf.element(instanceIndex).xy.sub(uScroll);
      const x = int(floor(p.x));
      const y = int(floor(p.y));
      If(x.greaterThanEqual(0).and(x.lessThan(uW)).and(y.greaterThanEqual(0)).and(y.lessThan(uH)), () => {
        const cell = y.mul(uW).add(x);
        const k = atomicAdd(slotCount.element(cell), uint(1)).toVar();
        // Any seed will do for the block's jump-flood start.
        jfaA.element(y.div(2).mul(uNW).add(x.div(2))).assign(int(instanceIndex));
        If(k.lessThan(uint(SLOTS)), () => {
          slots.element(cell.mul(SLOTS).add(int(k))).assign(int(instanceIndex));
        });
      });
    })().compute(MAX_NODES);

    const padScatter = Fn(() => {
      If(isLeafCode(lookBuf.element(instanceIndex).x), () => {
        const site = siteBuf.element(instanceIndex).toVar();
        const c = site.xy.sub(uScroll).toVar();
        const reach = max(site.z, 0).sqrt().add(2).toVar();
        const x0 = max(int(floor(c.x.sub(reach).div(2))), int(0));
        const y0 = max(int(floor(c.y.sub(reach).div(2))), int(0));
        const x1 = min(int(floor(c.x.add(reach).div(2))), uNW.sub(1));
        const y1 = min(int(floor(c.y.add(reach).div(2))), uNH.sub(1));
        Loop({ start: y0, end: y1.add(1), type: "int", condition: "<", name: "py" }, ({ py }) => {
          Loop({ start: x0, end: x1.add(1), type: "int", condition: "<", name: "px" }, ({ px }) => {
            const off = vec2(float(px).mul(2).add(1), float(py).mul(2).add(1)).sub(c);
            const pw = dot(off, off).sub(site.z).toVar();
            If(length(off).lessThan(reach), () => {
              const key = uint(clamp(pw.add(4096), 0, 32767)).shiftLeft(uint(17)).bitOr(uint(instanceIndex));
              atomicMin(padBlock.element(py.mul(uNW).add(px)), key);
            });
          });
        });
      });
    })().compute(MAX_NODES);

    const jfaPass = (src, dst, step) =>
      Fn(() => {
        const idx = int(instanceIndex);
        const x = idx.mod(uNW);
        const y = idx.div(uNW);
        const pc = vec2(float(x).mul(2).add(1), float(y).mul(2).add(1)); // block centre, css px
        const best = int(-1).toVar();
        const bestD = float(1e20).toVar();
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const qx = x.add(dx * step);
            const qy = y.add(dy * step);
            If(qx.greaterThanEqual(0).and(qx.lessThan(uNW)).and(qy.greaterThanEqual(0)).and(qy.lessThan(uNH)), () => {
              const s = src.element(qy.mul(uNW).add(qx)).toVar();
              If(s.greaterThanEqual(0), () => {
                const site = siteBuf.element(s);
                const d = site.xy.sub(uScroll).sub(pc);
                const dd = dot(d, d).sub(site.z).toVar();
                If(dd.lessThan(bestD), () => {
                  bestD.assign(dd);
                  best.assign(s);
                });
              });
            });
          }
        }
        dst.element(idx).assign(best);
      })().compute(blockCap);

    const passes = [];
    let src = jfaA;
    let dst = jfaB;
    for (const step of JFA_STEPS) {
      passes.push(jfaPass(src, dst, step));
      [src, dst] = [dst, src];
    }

    // Per 2x2 block of grid cells, the 4 nearest nodes to its centre: the
    // block's (and its border's) own seeds or jump-flood results, plus two
    // wider rings so neighbours a corner-radius away are found too. The
    // pixel shader then only has to rank these exactly.
    const jfa = src;
    const nbr = gpuArray(blockCap, "ivec4");
    const neighbours = Fn(() => {
      const idx = int(instanceIndex);
      const bx = idx.mod(uNW).mul(2); // the block's top-left cell
      const by = idx.div(uNW).mul(2);
      const pc = vec2(float(bx).add(1), float(by).add(1));
      const ids = [0, 1, 2, 3].map(() => int(-1).toVar());
      const pws = [0, 1, 2, 3].map(() => float(1e12).toVar());
      const consider = (node) => {
        const sIdx = node.toVar();
        let fresh = sIdx.greaterThanEqual(0);
        for (const id of ids) fresh = fresh.and(sIdx.notEqual(id));
        If(fresh, () => {
          const site = siteBuf.element(sIdx);
          const off = site.xy.sub(uScroll).sub(pc);
          const pw = dot(off, off).sub(site.z).toVar();
          let chain = null;
          for (let k = 0; k < 4; k++) {
            const insert = () => {
              for (let m = 3; m > k; m--) {
                ids[m].assign(ids[m - 1]);
                pws[m].assign(pws[m - 1]);
              }
              ids[k].assign(sIdx);
              pws[k].assign(pw);
            };
            chain = chain ? chain.ElseIf(pw.lessThan(pws[k]), insert) : If(pw.lessThan(pws[k]), insert);
          }
        });
      };
      const inGrid = (qx, qy) => qx.greaterThanEqual(0).and(qx.lessThan(uW)).and(qy.greaterThanEqual(0)).and(qy.lessThan(uH));
      // Real loops rather than unrolled taps keep this shader small.
      Loop(16, ({ i: n }) => {
        const qx = bx.add(n.mod(4)).sub(1);
        const qy = by.add(n.div(4)).sub(1);
        If(inGrid(qx, qy), () => {
          // A cell's own seeds, or (if it has none) its jump-flood result.
          // Slots fill in order, so stop at the first empty one.
          const cell = qy.mul(uW).add(qx).toVar();
          Loop(SLOTS, ({ i: j }) => {
            const cand = slots.element(cell.mul(SLOTS).add(j)).toVar();
            If(cand.lessThan(0), () => {
              If(j.equal(0), () => consider(jfa.element(qy.div(2).mul(uNW).add(qx.div(2)))));
              Break();
            });
            consider(cand);
          });
        });
      });
      // Two rings of 8 taps (radius 4 and 9), the second rotated a little,
      // plus (t = 16) the block's own jump-flood result: its nearest node by
      // power distance, so a heavy lily pad is found even where the block's
      // cells hold only water nodes it swallowed. (A third ring at 15px used
      // to cover that, at ~2ms a frame.) Kept inside this loop on purpose:
      // calling consider() outside the loops corrupted the lists.
      Loop(17, ({ i: t }) => {
        const ring = t.div(8);
        const r = select(ring.equal(0), float(4), select(ring.equal(1), float(9), float(0)));
        const ang = float(t.mod(8)).add(float(ring).div(3)).mul(Math.PI / 4);
        const qx = int(floor(pc.x.add(cos(ang).mul(r))));
        const qy = int(floor(pc.y.add(sin(ang).mul(r))));
        If(inGrid(qx, qy), () => consider(jfa.element(qy.div(2).mul(uNW).add(qx.div(2)))));
      });
      nbr.element(idx).assign(ivec4(ids[0], ids[1], ids[2], ids[3]));
    })().compute(blockCap);


    // Lily pads show every edge of their cell (water cells are invisible), so
    // each pad finds its true neighbours once per frame (the 8 nodes with the
    // nearest bisectors, from the block lists around it) and stores their
    // edges as lines, so a pad pixel needs no neighbour lookups. (Kept out of
    // the neighbours pass on purpose: extra work in there corrupted its lists.)
    // Per pad node, 5 vec4s: its 6 nearest edge lines as (normal x, normal y,
    // offset) triples packed back to back (sorted nearest first; farther ones
    // never form a visible edge), then how much room its flower has.
    const padData = gpuArray(MAX_NODES * 5, "vec4");
    // (Half the pads per frame, alternating: pads drift slowly, so edge lines
    // one frame old are fine, and this pass isn't cheap.)
    const padEdgePass = Fn(() => {
      If(isLeafCode(lookBuf.element(instanceIndex).x).and(int(instanceIndex).add(uFrame).bitAnd(int(1)).equal(0)), () => {
        const me = int(instanceIndex);
        const self = siteBuf.element(me).toVar();
        const vp = self.xy.sub(uScroll).toVar();
        const ids = Array.from({ length: 8 }, () => int(-1).toVar());
        const ts = Array.from({ length: 8 }, () => float(1e6).toVar());
        // A 5x5 grid of blocks 12px apart, out to 24px: every neighbour's
        // node is in the 4-nearest list of blocks along its shared edge.
        Loop(25, ({ i }) => {
          const qx = int(floor(vp.x.div(2))).add(i.mod(5).sub(2).mul(6));
          const qy = int(floor(vp.y.div(2))).add(i.div(5).sub(2).mul(6));
          If(qx.greaterThanEqual(0).and(qx.lessThan(uNW)).and(qy.greaterThanEqual(0)).and(qy.lessThan(uNH)), () => {
            const n = nbr.element(qy.mul(uNW).add(qx)).toVar();
            for (const c of [n.x, n.y, n.z, n.w]) {
              const cand = c.toVar();
              let fresh = cand.greaterThanEqual(0).and(cand.notEqual(me));
              for (const id of ids) fresh = fresh.and(cand.notEqual(id));
              If(fresh, () => {
                const other = siteBuf.element(cand);
                const D = max(length(other.xy.sub(self.xy)), 1e-3);
                // Distance from this node to the power bisector.
                const t = D.mul(D).add(self.z).sub(other.z).div(D.mul(2)).toVar();
                let chain = null;
                for (let k = 0; k < 8; k++) {
                  const insert = () => {
                    for (let m = 7; m > k; m--) {
                      ids[m].assign(ids[m - 1]);
                      ts[m].assign(ts[m - 1]);
                    }
                    ids[k].assign(cand);
                    ts[k].assign(t);
                  };
                  chain = chain ? chain.ElseIf(t.lessThan(ts[k]), insert) : If(t.lessThan(ts[k]), insert);
                }
              });
            }
          });
        });
        const base = me.mul(5);
        // The flower's spot (see the pad shader) and its distance to the cell edge.
        const rnd = attrBuf.element(me).z;
        const offset = vec2(fract(rnd.mul(3.7)), fract(rnd.mul(5.3))).sub(0.5).mul(lookBuf.element(me).y.mul(0.2)).toVar();
        const flowerRoom = float(1e6).toVar();
        const flat = [];
        for (let j = 0; j < 8; j++) {
          const other = siteBuf.element(max(ids[j], int(0)));
          const nrm = normalize(other.xy.sub(self.xy).add(vec2(1e-6, 0))).toVar();
          const t = select(ids[j].greaterThanEqual(0), ts[j], float(1e6));
          flowerRoom.assign(min(flowerRoom, t.sub(dot(offset, nrm))));
          if (j < 6) flat.push(nrm.x, nrm.y, t);
        }
        flat.push(flowerRoom, float(0));
        for (let k = 0; k < 5; k++) padData.element(base.add(k)).assign(vec4(...flat.slice(k * 4, k * 4 + 4)));
      });
    })().compute(MAX_NODES);

    // Dragonflies' questions about pads (see uFlyAsk): the lily pad under a
    // point, if any (one with no flower, fully grown: its id, centre and
    // radius), and where a watched pad is now and how fast it's moving.
    const padBlockRO = ro(padBlock, "uint", blockCap);
    const flyProbe = Fn(() => {
      const i = int(instanceIndex);
      const ask = uFlyAsk.element(i);
      const bx = int(floor(ask.x));
      const by = int(floor(ask.y));
      const found = vec4(-1, 0, 0, 0).toVar();
      If(bx.greaterThanEqual(0).and(bx.lessThan(uW)).and(by.greaterThanEqual(0)).and(by.lessThan(uH)), () => {
        const key = padBlockRO.element(by.div(2).mul(uNW).add(bx.div(2)));
        If(key.notEqual(uint(0xffffffff)), () => {
          const id = int(key.bitAnd(uint(0x1ffff)));
          const look = lookBuf.element(id);
          const site = siteBuf.element(id);
          const flower = fract(look.z.mul(7.31)).lessThan(FLOWER_FRACTION);
          If(floor(look.x).equal(1).and(flower.not()).and(site.w.greaterThan(0.95)), () => {
            found.assign(vec4(float(id), site.xy, look.y.mul(site.w)));
          });
        });
      });
      flyOut.element(i.mul(2)).assign(found);
      const watched = vec4(0, 0, 0, -1).toVar();
      const w = int(ask.z);
      If(w.greaterThanEqual(0), () => {
        const site = siteBuf.element(w);
        const ok = floor(lookBuf.element(w).x).equal(1);
        watched.assign(vec4(site.xy, length(posBuf.element(w).zw), select(ok, float(w), float(-2))));
      });
      flyOut.element(i.mul(2).add(1)).assign(watched);
    })().compute(MAX_FLIES);

    const material = new THREE.MeshBasicNodeMaterial();
    material.colorNode = shade(ro(nbr, "ivec4", blockCap), ro(slots, "int", capacity * SLOTS), ro(padBlock, "uint", blockCap));

    const pads = padShader(ro(padData, "vec4", MAX_NODES * 5));

    return { capacity, clear, seed, padScatter, passes, neighbours, padEdgePass, flyProbe, pads, quad: new THREE.QuadMesh(material) };
  };


  // Lily pads and duck cells ("leaves") are weighted cells drawn by the pad
  // pass; surface codes 1 (pad), 4 (duck body), 5 (bill), 6 (duckling body),
  // 7-9 (raccoon fur, mask, muzzle).
  const isLeafCode = (x) => {
    const code = floor(x);
    return code.equal(1).or(code.greaterThan(3.5));
  };

  // ------------------------------------------------------ render: lily pads

  // Lily pads are drawn by a pass of their own, one small quad per pad over
  // its leaf, composited on top of the main shader. (Inside the full-screen
  // shader their code made every pixel slower, through register pressure.)
  // A pad's leaf always lies inside its own power cell, so it never covers
  // text or ducks. The leaf: its cell (6 nearest edge lines from padEdgePass)
  // inset by a gap and rounded, softly intersected with a disc. A pad's
  // weight keeps the whole disc against water and ducks, so in open water
  // it's a round leaf; where pads meet, straight power-diagram edges, so a
  // colony packs into rounded Voronoi cells. Plus a notch, faint veins, an
  // upturned rim lit from the top left, and the waves' light. They open up
  // from nothing at the start.
  const padShader = (padRO) => {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide }); // (y is flipped into screen space)
    const look = lookRO.element(instanceIndex);
    const site = siteRO.element(instanceIndex);
    const isPad = isLeafCode(look.x);
    const reach = look.y.mul(site.w).add(3);
    const corner = site.xy.sub(uScroll).add(positionGeometry.xy.mul(reach));
    material.vertexNode = select(
      isPad,
      vec4(corner.x.div(float(uW)).mul(2).sub(1), float(1).sub(corner.y.div(float(uH)).mul(2)), 0, 1),
      vec4(2, 2, 2, 1), // (not a pad: off screen)
    );
    const vRel = varying(positionGeometry.xy.mul(reach));
    const vLook = varying(vec4(look.x, look.y, look.z, site.w));
    const vId = varying(float(instanceIndex));
    material.colorNode = Fn(() => {
      const sCol = vec3(0).toVar();
      const sFill = float(0).toVar();
      const id = int(round(vId));
      const look = vLook;
      const spacing = look.y;
      const rnd = look.z;
      const a1 = look.w;
      const rel = vRel.toVar();
      const d1 = length(rel).toVar();
      const aa = float(0.7).div(uDpr);
      // (Kept above zero: smoothstep with equal edges is undefined.)
      const size = max(spacing.mul(a1), 0.5);
          const r = spacing; // (a pad's radius)
          const isDuck = floor(look.x).greaterThan(3.5);
          const isBill = floor(look.x).equal(5);
          const isChick = floor(look.x).equal(6);
          const isFur = floor(look.x).equal(7);
          const isMask = floor(look.x).equal(8);
          const isPale = floor(look.x).equal(9);
          const base = id.mul(5);
          const pe2 = float(1e6).toVar();
          const pe3 = float(1e6).toVar();
          const flat = [];
          for (let k = 0; k < 5; k++) {
            const L = padRO.element(base.add(k)).toVar();
            flat.push(L.x, L.y, L.z, L.w);
          }
          for (let j = 0; j < 6; j++) {
            const [nx, ny, t] = flat.slice(j * 3, j * 3 + 3);
            const e = t.sub(rel.x.mul(nx).add(rel.y.mul(ny))).toVar();
            If(e.lessThan(pe2), () => {
              pe3.assign(pe2);
              pe2.assign(e);
            }).ElseIf(e.lessThan(pe3), () => {
              pe3.assign(e);
            });
          }
          const flowerRoom = flat[18];
          const offset = vec2(fract(rnd.mul(3.7)), fract(rnd.mul(5.3))).sub(0.5).mul(r.mul(0.2));
          // (Duck cells are narrow wedges: much tighter corners and a thinner
          // gap, so the duck reads as one bird and the cells don't shrink to
          // petals.)
          const halfGap = select(isDuck, float(0.55), float(1.4));
          const corner = r.mul(select(isDuck, float(0.16), float(0.45)));
          const q = vec2(corner.sub(pe2.sub(halfGap)), corner.sub(pe3.sub(halfGap)));
          const cellSdf = min(max(q.x, q.y), 0).add(length(max(q, vec2(0)))).sub(corner);
          const disc = d1.sub(size);
          const blend = select(isDuck, float(3), float(8));
          const hm = clamp(cellSdf.sub(disc).div(blend).mul(0.5).add(0.5), 0, 1);
          const leaf = mix(disc, cellSdf, hm).add(hm.mul(float(1).sub(hm)).mul(blend));
          // The notch: a slim wedge from the centre out.
          const notchDir = normalize(vec2(fract(rnd.mul(97.13)), fract(rnd.mul(41.71))).sub(0.5));
          const along = dot(rel, notchDir);
          const across = abs(rel.x.mul(notchDir.y).sub(rel.y.mul(notchDir.x)));
          const notch = select(along.greaterThan(0).and(isDuck.not()), across.sub(along.mul(0.16)).sub(0.3), float(1e6)); // < 0 inside the wedge (pads only)
          const shape = max(leaf, notch.negate()).toVar();
          const fill = float(1).sub(smoothstep(aa.negate(), aa, shape)).toVar();
          const leafFill = float(1).sub(smoothstep(aa.negate(), aa, leaf)); // (without the notch, for the flower)

          // Pads: two greens per pad. Ducks: the page's colours, text-cream
          // body and accent-orange bill (ducklings: a downy yellow body).
          // Each with a lighter and darker tone for the light; lighter riding
          // a crest, darker in a trough.
          // (Raccoons: grey fur, a dark mask, tail rings and nose, a pale
          // muzzle and ears.)
          const duckBase = select(
            isBill,
            vec3(uAccent),
            select(isChick, vec3(uDuckling), select(isFur, vec3(uRaccoon), select(isMask, vec3(uRaccoonDark), select(isPale, vec3(uRaccoonPale), vec3(uText))))),
          );
          const baseCol = select(isDuck, duckBase, mix(uPad, uPadLight, fract(rnd.mul(13.7))));
          const light = select(isDuck, mix(duckBase, vec3(1), select(isMask, float(0.22), float(0.45))), vec3(uPadLight).mul(1.18));
          const shade = select(isDuck, duckBase.mul(select(isBill, float(0.8), float(0.82))), vec3(uPadRim));
          const edgeCol = select(isDuck, duckBase.mul(select(isBill, float(0.72), float(0.74))), vec3(uPadRim));
          const padLift = fract(look.x).sub(0.45);
          const leafCol = mix(baseCol, select(padLift.greaterThan(0), light, shade), min(abs(padLift).mul(2.2), 0.6)).toVar();
          // Veins: faint lines radiating from the centre, between the rim and
          // the middle (measured from the notch so they're symmetric to it).
          const theta = fastAtan2(across.mul(sign(rel.x.mul(notchDir.y).sub(rel.y.mul(notchDir.x)))), along);
          const veinPhase = abs(fract(theta.mul(11 / (Math.PI * 2))).sub(0.5));
          const veinDist = veinPhase.mul((Math.PI * 2) / 11).mul(d1);
          const vein = float(1).sub(smoothstep(0.25, 0.9, veinDist)).mul(smoothstep(size.mul(0.12), size.mul(0.35), d1)).mul(float(1).sub(smoothstep(size.mul(0.7), size.mul(0.92), d1)));
          leafCol.assign(mix(leafCol, uPadRim, vein.mul(select(isDuck, float(0), float(0.22)))));
          // The rim turns up a little: lit on the top-left, shaded bottom-right.
          const rimBand = smoothstep(-6, -1.5, shape);
          const facing = dot(rel.div(max(d1, 1e-3)), vec2(-0.6, -0.8));
          leafCol.assign(mix(leafCol, select(facing.greaterThan(0), light, shade), rimBand.mul(abs(facing)).mul(0.45)));
          // ...and a thin darker edge.
          leafCol.assign(mix(leafCol, edgeCol, smoothstep(-1.6, 0, shape)));
          sCol.assign(leafCol);
          sFill.assign(fill);

          // Some pads carry a water lily: two layers of pointed petals around
          // a yellow centre, swaying a little; it opens after its pad.
          // (Only near the flower: the petal shapes aren't cheap.)
          const fq = rel.sub(offset).toVar();
          If(isDuck.not().and(fract(rnd.mul(7.31)).lessThan(FLOWER_FRACTION)).and(a1.greaterThan(0.6)).and(flowerRoom.greaterThan(r.mul(0.45).add(3))).and(length(fq).lessThan(r.mul(0.66).add(2))), () => {
            // Sized to fit its pad's cell, so petals never get cut off.
            const R = min(r.mul(0.6), flowerRoom.sub(3)).mul(smoothstep(0.6, 1, a1)).max(0.01);
            const rho = length(fq);
            const theta = atan(fq.y, fq.x).add(rnd.mul(Math.PI * 2)).add(sin(uTime.mul(0.4).add(rnd.mul(20))).mul(0.08));
            // Fold into one petal: x along the petal, y across it.
            const petal = (n, phase, len, width) => {
              const sector = (Math.PI * 2) / n;
              const a = mod(theta.add(phase), sector).sub(sector / 2);
              const x = rho.mul(cos(a));
              const y = abs(rho.mul(sin(a)));
              const w = sin(clamp(x.div(len), 0, 1).mul(Math.PI)).mul(len.mul(width));
              return max(y.sub(w), x.sub(len));
            };
            const outer = petal(8, 0, R, 0.3);
            const inner = petal(8, Math.PI / 8, R.mul(0.68), 0.34);
            const centre = rho.sub(R.mul(0.22));
            const cover = (sdf) => float(1).sub(smoothstep(aa.negate(), aa, sdf)).mul(leafFill);
            // A soft shadow on the pad first, then outer, inner, centre.
            const layer = (c, a) => {
              sCol.assign(mix(sCol, c, a));
              sFill.assign(max(sFill, a));
            };
            sCol.assign(mix(sCol, uPadRim, cover(petal(8, 0, R.mul(1.05), 0.34).sub(1.5)).mul(0.35).mul(fill)));
            layer(mix(uPetalDeep, uPetal, smoothstep(0, 1, rho.div(R)).mul(0.85).add(0.15)), cover(outer));
            layer(mix(uPetalDeep, uPetal, rho.div(R).mul(0.8)), cover(inner));
            layer(uAccent, cover(centre));
          });
      return vec4(sCol, sFill);
    })();
    const geometry = new THREE.InstancedBufferGeometry().copy(new THREE.PlaneGeometry(2, 2));
    geometry.instanceCount = 0;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    const scene = new THREE.Scene();
    scene.add(mesh);
    return { scene, geometry };
  };

  // ---------------------------------------------------- render: dragonflies

  // A dragonfly, top-down, as simple and cute as the ducks: one chubby
  // rounded pill of a body in flat colour, with the same lit rim (top left)
  // and thin darker edge as the pads' and ducks' cells, and four round
  // clear wings spread out sideways (the classic dragonfly cross).
  // Flying, the wings beat into a shimmer (they foreshorten and fade, fore
  // and hind pairs out of step); perched, they lie flat and still. Its
  // shadow falls on the water down and to the right, further and softer the
  // higher it is. One quad per slot, drawn over everything, in the lily pad
  // pass (its own render pass cost ~2ms at Retina size). (A first take
  // with segment bands, wing veins, tip spots and eye glints was too
  // detailed next to the ducks; so was a second with two eye beads, a
  // thorax and a long thin abdomen; a third with a separate darker head;
  // and a fourth blending that head into a teardrop tail looked like a
  // matchstick.)
  const flyMesh = (() => {
    const material = new THREE.MeshBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide }); // (y is flipped into screen space)
    const A = uFlyA.element(instanceIndex);
    const B = uFlyB.element(instanceIndex);
    // (Perched, it rides its pad: drawn at the pad's spot, wherever the pad
    // has swayed to this frame.)
    const pad = siteRO.element(max(int(B.x), int(0)));
    const at = select(B.x.greaterThanEqual(0), pad.xy.sub(uScroll).add(B.yz), A.xy);
    const reach = A.w.mul(0.006).add(1).mul(25).add(A.w.mul(0.8)).add(4);
    const corner = at.add(positionGeometry.xy.mul(reach));
    material.vertexNode = select(
      B.w.greaterThanEqual(0),
      vec4(corner.x.div(float(uW)).mul(2).sub(1), float(1).sub(corner.y.div(float(uH)).mul(2)), 0, 1),
      vec4(2, 2, 2, 1), // (unused slot: off screen)
    );
    const vRel = varying(positionGeometry.xy.mul(reach));
    const vFly = varying(vec4(A.z, A.w, B.w, float(instanceIndex)));
    material.colorNode = Fn(() => {
      const alt = vFly.y;
      const flap = vFly.z;
      const hx = cos(vFly.x);
      const hy = sin(vFly.x);
      const aa = float(0.7).div(uDpr);
      // (A little bigger the higher it flies.)
      const g = alt.mul(0.006).add(1).mul(FLY_SIZE);
      // Screen offset -> its own frame: x forward, y to its right, px.
      const toLocal = (q) => vec2(dot(q, vec2(hx, hy)), dot(q, vec2(hy.negate(), hx))).div(g);
      // (The light, top left on screen, in its frame.)
      const light = vec2(dot(vec2(-0.6, -0.8), vec2(hx, hy)), dot(vec2(-0.6, -0.8), vec2(hy.negate(), hx)));
      const ellipse = (q, c, u, L, W) => {
        const d = q.sub(c);
        return length(vec2(dot(d, u).div(L), dot(d, vec2(u.y.negate(), u.x)).div(W))).sub(1).mul(min(L, W));
      };
      // The body: one chubby rounded pill, a little narrower at the tail (an
      // uneven capsule: head circle r1 at x=X0, tail circle r2 H behind it,
      // exact distance), with its distance and outward direction (for the
      // rim light). Widest where the wings meet it, so with the wings spread
      // sideways it reads as a dragonfly rather than a knob on a stick.
      const body = (q) => {
        const X0 = 5.5, R1 = 2.9, R2 = 2.2, H = 14;
        const B = (R1 - R2) / H;
        const A = Math.sqrt(1 - B * B);
        const across = abs(q.y);
        const along = float(X0).sub(q.x); // (towards the tail)
        const k = across.mul(-B).add(along.mul(A));
        const head = vec2(across, along);
        const tail = vec2(across, along.sub(H));
        const d = select(k.lessThan(0), length(head).sub(R1), select(k.greaterThan(A * H), length(tail).sub(R2), across.mul(A).add(along.mul(B)).sub(R1)));
        const m = select(k.lessThan(0), normalize(head.add(vec2(1e-4, 0))), select(k.greaterThan(A * H), normalize(tail.add(vec2(1e-4, 0))), vec2(A, B)));
        return { d, n: vec2(m.y.negate(), m.x.mul(select(q.y.lessThan(0), float(-1), float(1)))) };
      };
      const pieces = (q) => [{ ...body(q), col: vec3(uFly) }];
      // Wings: fore pair, then hind pair, spread out sideways (fore a little
      // forward, hind a little back); beating, each foreshortens (and fades)
      // through its stroke.
      const beat = (lag) => mix(float(1), abs(cos(uTime.mul(Math.PI * 2 * 26).add(vFly.w.mul(1.7)).add(lag))).mul(0.5).add(0.5), flap);
      const wings = [
        { root: vec2(1.6, 0), dx: 0.25, L: 7.2, W: 3.4, lag: 0 },
        { root: vec2(-0.6, 0), dx: -0.2, L: 6.8, W: 3.6, lag: Math.PI / 2 },
      ].flatMap((w) =>
        [-1, 1].map((side) => {
          const u = normalize(vec2(w.dx, side));
          const L = beat(w.lag).mul(w.L);
          return (q) => ellipse(q, w.root.add(u.mul(L)), u, L, float(w.W));
        }),
      );

      const pc = vec3(0).toVar(); // (premultiplied)
      const a = float(0).toVar();
      const over = (col, al) => {
        pc.assign(col.mul(al).add(pc.mul(float(1).sub(al))));
        a.assign(al.add(a.mul(float(1).sub(al))));
      };

      // Shadow on the water (or the pad it sits on).
      const blur = alt.mul(0.06).add(0.5);
      const sq = toLocal(vRel.sub(vec2(0.45, 0.75).mul(alt.mul(0.55).add(1.2))));
      const shadowBody = pieces(sq).map((p) => p.d).reduce((x, y) => min(x, y));
      const shadowWing = wings.map((w) => w(sq)).reduce((x, y) => min(x, y));
      over(vec3(0.02, 0.08, 0.14), max(float(1).sub(smoothstep(blur.negate(), blur, shadowBody)).mul(0.26), float(1).sub(smoothstep(blur.negate(), blur, shadowWing)).mul(0.1)));

      const q = toLocal(vRel).toVar();
      // Wings: plain and clear, a touch brighter at the edge; fainter (a
      // blur) while beating.
      for (const w of wings) {
        const d = w(q);
        const cover = float(1).sub(smoothstep(aa.negate(), aa, d));
        over(vec3(uFlyWing), mix(float(0.42), float(0.7), smoothstep(-1, -0.2, d)).mul(float(1).sub(flap.mul(0.55))).mul(cover));
      }
      // Body: each piece flat, its rim lit on the light's side and shaded on
      // the other, with a thin darker edge.
      for (const p of pieces(q)) {
        const facing = dot(p.n, light);
        const rim = smoothstep(-2.2, -0.5, p.d);
        const col = mix(p.col, select(facing.greaterThan(0), mix(p.col, vec3(1), 0.45), p.col.mul(0.8)), rim.mul(abs(facing)).mul(0.5)).toVar();
        col.assign(mix(col, p.col.mul(0.72), smoothstep(-0.8, 0, p.d)));
        over(col, float(1).sub(smoothstep(aa.negate(), aa, p.d)));
      }

      return vec4(pc.div(max(a, 1e-4)), a);
    })();
    const geometry = new THREE.InstancedBufferGeometry().copy(new THREE.PlaneGeometry(2, 2));
    geometry.instanceCount = MAX_FLIES;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    mesh.renderOrder = 1; // (after the pads)
    mesh.visible = false;
    return mesh;
  })();

  // ------------------------------------------------------- render: shading

  // atan2 to within ~0.01 rad, much cheaper than the real one (pad veins).
  const fastAtan2 = (y, x) => {
    const ax = abs(x);
    const ay = abs(y);
    const a = min(ax, ay).div(max(max(ax, ay), 1e-6));
    const s2 = a.mul(a);
    const r = s2.mul(-0.0464964749).add(0.15931422).mul(s2).sub(0.327622764).mul(s2).mul(a).add(a);
    const r1 = select(ay.greaterThan(ax), float(Math.PI / 2).sub(r), r);
    const r2 = select(x.lessThan(0), float(Math.PI).sub(r1), r1);
    return select(y.lessThan(0), r2.negate(), r2);
  };

  const shade = (nbrRO, slotsRO, padBlockRO) =>
    Fn(() => {
      const p = screenCoordinate.xy.div(uDpr).toVar();
      const bx = int(floor(p.x));
      const by = int(floor(p.y));

      // Candidates: the 2x2 block's list, plus this cell's own seeds (dense
      // small text can have nodes closer than the block centre's best 4) and,
      // in dense areas only, the 8 neighbouring cells' first seeds. They're
      // streamed (re-read in each pass) rather than stored, which keeps the
      // shader's register use, and so its cost for every pixel, low.
      const forEachCandidate = (visit) => {
        const each = (node) => {
          const id = node.toVar();
          If(id.greaterThanEqual(0), () => visit(id));
        };
        If(bx.greaterThanEqual(0).and(bx.lessThan(uW)).and(by.greaterThanEqual(0)).and(by.lessThan(uH)), () => {
          const n = nbrRO.element(by.div(2).mul(uNW).add(bx.div(2))).toVar();
          each(n.x);
          each(n.y);
          each(n.z);
          each(n.w);
          const base = by.mul(uW).add(bx).mul(SLOTS);
          const s0 = slotsRO.element(base).toVar();
          If(s0.greaterThanEqual(0), () => {
            visit(s0);
            Loop({ start: 1, end: SLOTS }, ({ i: j }) => {
              const sj = slotsRO.element(base.add(j)).toVar();
              If(sj.lessThan(0), () => {
                Break();
              });
              visit(sj);
            });
            Loop(9, ({ i: k }) => {
              const qx = bx.add(k.mod(3)).sub(1);
              const qy = by.add(k.div(3)).sub(1);
              If(k.notEqual(4).and(qx.greaterThanEqual(0)).and(qx.lessThan(uW)).and(qy.greaterThanEqual(0)).and(qy.lessThan(uH)), () => {
                each(slotsRO.element(qy.mul(uW).add(qx).mul(SLOTS)));
              });
            });
          });
        });
      };

      // Pass 1: the cell this pixel is in (lowest power distance).
      const i1 = int(-1).toVar();
      const pw1 = float(1e12).toVar();
      const p1 = vec2(0).toVar();
      const a1 = float(1).toVar(); // how alive the nearest node is
      forEachCandidate((id) => {
        const site = siteRO.element(id).toVar();
        const sp = site.xy.sub(uScroll).toVar();
        const off = sp.sub(p);
        const pw = dot(off, off).sub(site.z).toVar();
        If(pw.lessThan(pw1), () => {
          i1.assign(id);
          pw1.assign(pw);
          p1.assign(sp);
          a1.assign(site.w);
        });
      });
      // ...and the lily pad covering this block, if any.
      If(bx.greaterThanEqual(0).and(bx.lessThan(uW)).and(by.greaterThanEqual(0)).and(by.lessThan(uH)), () => {
        const key = padBlockRO.element(by.div(2).mul(uNW).add(bx.div(2))).toVar();
        If(key.notEqual(uint(0xffffffff)), () => {
          const id = int(key.bitAnd(uint(0x1ffff))).toVar();
          const site = siteRO.element(id).toVar();
          const sp = site.xy.sub(uScroll).toVar();
          const off = sp.sub(p);
          const pw = dot(off, off).sub(site.z).toVar();
          If(pw.lessThan(pw1), () => {
            i1.assign(id);
            pw1.assign(pw);
            p1.assign(sp);
            a1.assign(site.w);
          });
        });
      });

      const aa = float(0.7).div(uDpr);
      const col = vec3(uWater).toVar();
      // Whatever lies on the water here (pad, glyph, bar, duck): its colour and
      // coverage, composited over the (rippling) water at the end.
      const sCol = vec3(0).toVar();
      const sFill = float(0).toVar();

      If(i1.greaterThanEqual(0), () => {
        col.assign(mix(col, uWaterLight, 0.04));
        const K1 = lookRO.element(i1).toVar();
        const code = floor(K1.x);
        const spacing = K1.y;
        const rnd = K1.z;
        const extra = K1.w;
        const d1 = length(p1.sub(p));
        const rel = p.sub(p1);

        // Lily pads and ducks are drawn by their own pass (padShader) on top;
        // under them the water needs no ripples.
        If(isLeafCode(K1.x).and(d1.lessThan(spacing.mul(a1).sub(2))), () => {
          sFill.assign(1);
          sCol.assign(col);
        });

        // Glyphs and bars.
        If(code.greaterThan(1.5).and(code.lessThan(3.5)), () => {
          // Pass 2: the two closest *edges*. The nearest edge doesn't always
          // belong to the second-nearest node, so measure every candidate's
          // bisector (exact for power diagrams: they're straight lines).
          // (Only these cells need it: water is invisible and pads use their
          // own edge lines, so most pixels skip this whole loop.)
          const n2 = int(-1).toVar();
          const n3 = int(-1).toVar();
          const e2 = float(1e6).toVar();
          const e3 = float(1e6).toVar();
          const p2 = vec2(0).toVar();
          const p3 = vec2(0).toVar();
          forEachCandidate((id) => {
            If(id.notEqual(i1).and(id.notEqual(n2)), () => {
              const site = siteRO.element(id).toVar();
              const sp = site.xy.sub(uScroll).toVar();
              const off = sp.sub(p);
              const e = dot(off, off).sub(site.z).sub(pw1).div(max(length(sp.sub(p1)), 1e-3).mul(2)).toVar();
              If(e.lessThan(e2), () => {
                n3.assign(n2);
                e3.assign(e2);
                p3.assign(p2);
                n2.assign(id);
                e2.assign(e);
                p2.assign(sp);
              }).ElseIf(e.lessThan(e3).and(id.notEqual(n3)), () => {
                n3.assign(id);
                e3.assign(e);
                p3.assign(sp);
              });
            });
          });

          const isGlyph = code.equal(2); // small, solid cells
          // Lighting and chunks only where the cells are big enough to show it.
          const big = smoothstep(1.9, 3.6, spacing);

          // Rounded cell: inset both nearby edges by half the gap, then round
          // the corner where they meet. Power bisectors are straight lines, so
          // the edge distances are exact. Glyph and bar cells join their own
          // kind without a gap (letters and bars stay solid); only their
          // outer outline is rounded.
          const joins = (iB) => {
            const codeB = floor(lookRO.element(max(iB, int(0))).x);
            return iB.greaterThanEqual(0).and(codeB.equal(code)).toVar();
          };
          const join2 = joins(n2);
          const join3 = joins(n3);
          const edge = (e, joined) => select(joined, float(1e6), e);
          const halfGap = select(isGlyph, spacing.mul(0.05), spacing.mul(0.09));
          // (Generous rounding where outline cells meet smooths the letters'
          // edges, like the pads' rounded cells.)
          const radius = select(isGlyph, spacing.mul(mix(float(0.15), float(0.55), big)), spacing.mul(0.4));
          const q = vec2(radius.sub(edge(e2, join2).sub(halfGap)), radius.sub(edge(e3, join3).sub(halfGap)));
          const cellSdf = min(max(q.x, q.y), 0).add(length(max(q, vec2(0)))).sub(radius).toVar();

          // A flung glyph (or bar) node would own a huge cell, so clip it to a
          // disc. (Tight enough that edge cells of small text can't grow long
          // tails.)
          cellSdf.assign(max(cellSdf, d1.sub(spacing.mul(select(isGlyph, float(1.8), float(2.4))))));

          const fill = float(1).sub(smoothstep(aa.negate(), aa, cellSdf));

          // Glyphs: cream on water; dark when sitting on a hovered orange plate.
          const glyphCol = mix(mix(uWater, uText, extra), uDeep, fract(K1.x).div(0.9));
          const barCol = uAccent.mul(floor(rnd).div(32).mul(0.12).add(0.94));
          const surface = select(isGlyph, glyphCol, barCol).toVar();
          // Riding the waves like the pads: lighter on a crest, sinking a
          // little into the water in a trough (rnd's fraction is the lift,
          // 0.45 = level). (Darkening toward black turned cream letters a
          // muddy grey.)
          const textLift = fract(rnd).sub(0.45);
          surface.assign(
            select(
              textLift.greaterThan(0),
              mix(surface, vec3(1), min(textLift.mul(1.2), 0.3)),
              mix(surface, uWater, min(textLift.negate().mul(0.9), 0.28)),
            ),
          );
          sFill.assign(fill);

          // Lit like the lily pads and ducks: each letter's (and bar's) outer
          // outline turns up a little, lit on the top-left and shaded on the
          // bottom-right, with a thin darker edge. The outline's direction is
          // that of the nearest edge with a cell of another kind. (Scaled
          // with the cells, and faded out on small text, where it would just
          // muddy the letters.)
          const outN = select(
            join2.not(),
            normalize(p2.sub(p1).add(vec2(1e-6, 0))),
            normalize(p3.sub(p1).add(vec2(1e-6, 0))),
          );
          const outline = join2.not().or(join3.not());
          const depth = cellSdf.negate(); // distance in from the outline
          const rimW = spacing.mul(0.7);
          // (Softer along the letter's own outline, which is a little ragged.)
          const rimBand = float(1).sub(smoothstep(rimW.mul(0.35), rimW, depth)).mul(select(outline, big, float(0)));
          const facing = dot(outN, vec2(-0.6, -0.8));
          surface.assign(
            select(
              facing.greaterThan(0),
              mix(surface, vec3(1), rimBand.mul(facing).mul(0.55)),
              surface.mul(float(1).sub(rimBand.mul(facing.negate()).mul(0.3))),
            ),
          );
          const edgeW = spacing.mul(0.2).max(0.45);
          surface.assign(surface.mul(float(1).sub(float(1).sub(smoothstep(edgeW.mul(0.3), edgeW, depth)).mul(big).mul(0.24))));

          // A faint seam where a letter's cells meet keeps the Voronoi visible,
          // like the seams between a duck's cells. (Faded out on small text,
          // where seams just make letters look dotted.)
          const seamW = min(float(0.45), spacing.mul(0.09));
          const seamA = float(1).sub(smoothstep(seamW.sub(aa), seamW.add(aa), e2)).mul(0.3).mul(smoothstep(2.4, 4, spacing));
          sCol.assign(mix(surface, uWater, select(join2, seamA, float(0))));
        });
      });

      // Ripples: the mosaic (see facetUpdate), only where water shows. Calm
      // water skips it after one read.
      const g = floor(p.add(uScroll).div(FACET));
      const lx = int(g.x).sub(uFacetX0);
      const ly = int(g.y).sub(uFacetY0);
      const inFacets = lx.greaterThanEqual(1).and(lx.lessThan(uFacetW.sub(1))).and(ly.greaterThanEqual(1)).and(ly.lessThan(uFacetH.sub(1)));
      If(inFacets.and(sFill.lessThan(0.995)).and(facetMaxRO.element(ly.mul(uFacetW).add(lx)).greaterThan(0.03)), () => {
        const q = p.add(uScroll);
        const fd1 = float(1e9).toVar();
        const fd2 = float(1e9).toVar();
        const f1 = vec4(0).toVar();
        const fc2 = vec2(0).toVar();
        // The 2x2 cells on the side of its cell the pixel is on.
        const inCell = fract(q.div(FACET));
        const sx = select(inCell.x.lessThan(0.5), int(-1), int(0));
        const sy = select(inCell.y.lessThan(0.5), int(-1), int(0));
        for (let oy = 0; oy <= 1; oy++) {
          for (let ox = 0; ox <= 1; ox++) {
            const f = facetRO.element(ly.add(sy).add(oy).mul(uFacetW).add(lx.add(sx).add(ox))).toVar();
            const off = f.xy.sub(q);
            const dd = dot(off, off).toVar();
            If(dd.lessThan(fd1), () => {
              fd2.assign(fd1);
              fc2.assign(f1.xy);
              fd1.assign(dd);
              f1.assign(f);
            }).ElseIf(dd.lessThan(fd2), () => {
              fd2.assign(dd);
              fc2.assign(f.xy);
            });
          }
        }
        // Each mosaic cell swells into a rounded blob: light on a crest, a
        // faint shadow in a trough.
        const e = fd2.sub(fd1).div(max(length(fc2.sub(f1.xy)), 1e-3).mul(2));
        const a = max(f1.z, f1.w);
        const cellSdf = float(FACET * 0.07).sub(e);
        const blob = fd1.sqrt().sub(mix(float(FACET * 0.2), float(FACET * 0.6), a));
        const shape = max(cellSdf, blob).toVar();
        const fill = float(1).sub(smoothstep(aa.negate(), aa, shape)).toVar();
        const crestCol = mix(uRipple, uRippleRim, smoothstep(-2.5, 0, shape));
        col.assign(mix(col, crestCol, fill.mul(f1.z)));
        col.assign(mix(col, uWaterDeep, fill.mul(f1.w).mul(0.07)));
      });

      col.assign(mix(col, sCol, sFill));

      return vec4(col, 1);
    })();

  // ----------------------------------------------------------------- layout

  const groups = [];
  const groupOf = new Map();

  const addGroup = (el, type) => {
    const g = {
      el,
      type,
      hover: 0,
      target: 0,
      restTone: type === "nav" ? 0.3 : type === "current" ? 0.62 : 1,
      hoverTone: type === "nav" || type === "current" ? 0.62 : 1,
      bar: [-1, -1, -1, -1],
    };
    groups.push(g);
    groupOf.set(el, groups.length - 1);
    if (type === "nav" || type === "link") {
      const on = () => (g.target = 1);
      const off = () => (g.target = 0);
      el.addEventListener("pointerenter", on);
      el.addEventListener("pointerleave", off);
      el.addEventListener("focus", on);
      el.addEventListener("blur", off);
    }
  };

  addGroup(null, "bg");
  for (const el of document.querySelectorAll("[data-fx]")) {
    if (groups.length < MAX_GROUPS) addGroup(el, el.dataset.fx);
  }

  const maskCanvas = document.createElement("canvas");
  const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });

  let nodeCount = 0;
  let layoutGen = 0; // (bumped by every layout: node ids change)
  let firstLayout = true;

  const layout = () => {
    const rand = mulberry32(1337 + POND_SEED);
    const sx = scrollX;
    const sy = scrollY;
    const docW = Math.max(document.documentElement.scrollWidth, innerWidth);
    const docH = Math.max(document.documentElement.scrollHeight, innerHeight);
    const barPx = narrow.matches ? 4 : 8;

    const home = homeBuf.value.array;
    const attr = attrBuf.value.array;
    let n = 0;
    const push = (x, y, group, kind, barV, spacing, w = rand()) => {
      if (n >= MAX_NODES) return;
      home.set([x, y, group, kind], n * 4);
      attr.set([barV, spacing, rand(), w], n * 4);
      n++;
    };

    // Coarse occupancy grid so background nodes keep clear of the text.
    const OCC = 6;
    const occW = Math.ceil(docW / OCC) + 1;
    const occH = Math.ceil(docH / OCC) + 1;
    const occ = new Uint8Array(occW * occH);
    const mark = (x, y) => {
      const cx = Math.floor(x / OCC);
      const cy = Math.floor(y / OCC);
      if (cx >= 0 && cy >= 0 && cx < occW && cy < occH) occ[cy * occW + cx] = 1;
    };

    for (let gi = 1; gi < groups.length; gi++) {
      const g = groups[gi];
      const el = g.el;
      const cs = getComputedStyle(el);
      const fontSize = parseFloat(cs.fontSize);
      const spacing = Math.min(5, Math.max(1.8, fontSize * 0.065));
      const hasPlate = g.type === "link" || g.type === "current";

      const chars = glyphRects(el, sx, sy);
      if (!chars.length) continue;
      const plates = hasPlate
        ? [...el.getClientRects()].map((r) => ({ l: r.left + sx, t: r.top + sy, r: r.right + sx, b: r.bottom + sy }))
        : [];

      if (hasPlate && plates.length) {
        const h = plates[0].b - plates[0].t;
        g.bar =
          g.type === "current"
            ? [0, 2 / h, 0, 2 / h]
            : [3 / h, (3 + barPx) / h, 0, 1];
      }

      // Bounding box of everything we sample, padded for the halo.
      const pad = spacing * 2 + 2;
      let l = Infinity, t = Infinity, r = -Infinity, b = -Infinity;
      for (const c of [...chars, ...plates]) {
        l = Math.min(l, c.l);
        t = Math.min(t, c.t);
        r = Math.max(r, c.r);
        b = Math.max(b, c.b);
      }
      l = Math.floor(l - pad);
      t = Math.floor(t - pad);
      r = Math.ceil(r + pad);
      b = Math.ceil(b + pad);

      // Rasterize the glyphs exactly where the DOM placed them.
      const SS = fontSize < 30 ? 3 : 2;
      const mw = (r - l) * SS;
      const mh = (b - t) * SS;
      maskCanvas.width = mw;
      maskCanvas.height = mh;
      maskCtx.setTransform(SS, 0, 0, SS, -l * SS, -t * SS);
      maskCtx.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
      maskCtx.fillStyle = "#000";
      maskCtx.textBaseline = "alphabetic";
      const ascent = maskCtx.measureText("Hg").fontBoundingBoxAscent;
      for (const c of chars) maskCtx.fillText(c.ch, c.l, c.t + ascent);
      const pixels = maskCtx.getImageData(0, 0, mw, mh).data;
      const ink = (x, y) => {
        const px = Math.floor((x - l) * SS);
        const py = Math.floor((y - t) * SS);
        if (px < 0 || py < 0 || px >= mw || py >= mh) return 0;
        return pixels[(py * mw + px) * 4 + 3] / 255;
      };
      const nearInk = (x, y) => {
        const rr = spacing * 0.9;
        for (let a = 0; a < 8; a++) {
          const ang = (a / 8) * Math.PI * 2;
          if (ink(x + Math.cos(ang) * rr, y + Math.sin(ang) * rr) > 0.35) return true;
        }
        return ink(x, y) > 0.1;
      };

      // Jittered hex grid over the box.
      const rowH = spacing * 0.866;
      const rows = Math.ceil((b - t) / rowH);
      const cols = Math.ceil((r - l) / spacing) + 1;
      const barTop = g.bar[1] + 0.08;
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const x = l + (col + (row & 1) * 0.5) * spacing + (rand() - 0.5) * 0.35 * spacing;
          const y = t + (row + 0.5) * rowH + (rand() - 0.5) * 0.35 * spacing;
          if (ink(x, y) > 0.5) {
            push(x, y, gi, KIND_INK, -1, spacing);
            mark(x, y);
            continue;
          }
          const near = nearInk(x, y);
          if (hasPlate) {
            const pr = plates.find((q) => x >= q.l && x <= q.r && y >= q.t && y <= q.b);
            const v = pr ? (pr.b - y) / (pr.b - pr.t) : -1;
            const lattice = row % 2 === 0 && col % 2 === 0;
            // Keep the bar's lower edge crisp with a dense strip just below it.
            const under = !pr && plates.some((q) => x >= q.l && x <= q.r && y > q.b && y < q.b + spacing * 2);
            if (near || lattice || under || (pr && v <= barTop)) {
              push(x, y, gi, KIND_PLATE, v, spacing);
              mark(x, y);
            }
          } else if (near) {
            push(x, y, gi, KIND_HALO, -1, spacing);
            mark(x, y);
          }
        }
      }
    }

    // Ducks: a few hundred cells each (attr.x = which duck, attr.w = part),
    // home = position in the duck's frame; the GPU carries them along.
    for (let d = 0; d < DUCK_SLOTS; d++) {
      for (const c of slotCells(d)) push(c.x, c.y, 0, KIND_DUCK, d, c.r, c.part);
    }

    // Background: a jittered grid over the whole document (plus a margin so
    // edge cells look natural), skipping anything close to text.
    const B = narrow.matches ? 24 : 32;
    const clearance = Math.ceil((B * 0.5) / OCC);
    for (let y = -B; y < docH + B; y += B) {
      for (let x = -B; x < docW + B; x += B) {
        const px = x + rand() * B;
        const py = y + rand() * B;
        const cx = Math.floor(px / OCC);
        const cy = Math.floor(py / OCC);
        let blocked = false;
        for (let oy = -clearance; oy <= clearance && !blocked; oy++) {
          for (let ox = -clearance; ox <= clearance; ox++) {
            const qx = cx + ox;
            const qy = cy + oy;
            if (qx >= 0 && qy >= 0 && qx < occW && qy < occH && occ[qy * occW + qx]) {
              blocked = true;
              break;
            }
          }
        }
        if (blocked) continue;
        // attr.x = room: distance to the nearest text (capped), which limits
        // the size of a lily pad here.
        let room = 48;
        const reach = Math.ceil(room / OCC);
        for (let oy = -reach; oy <= reach; oy++) {
          for (let ox = -reach; ox <= reach; ox++) {
            const qx = cx + ox;
            const qy = cy + oy;
            if (qx >= 0 && qy >= 0 && qx < occW && qy < occH && occ[qy * occW + qx]) {
              room = Math.min(room, Math.hypot((qx + 0.5) * OCC - px, (qy + 0.5) * OCC - py));
            }
          }
        }
        push(px, py, 0, KIND_BG, room, B);
      }
    }

    if (n >= MAX_NODES) console.warn(`Voronoi: node budget exhausted (${MAX_NODES})`);

    // Nodes start scattered across the viewport and fly home; later layouts
    // just retarget them so they glide to the new positions.
    if (firstLayout) {
      const pos = posBuf.value.array;
      const still = reduceMotion.matches;
      for (let i = 0; i < MAX_NODES; i++) {
        pos[i * 4] = still && i < n ? home[i * 4] : sx + rand() * innerWidth;
        pos[i * 4 + 1] = still && i < n ? home[i * 4 + 1] : sy + rand() * innerHeight;
        pos[i * 4 + 2] = 0;
        pos[i * 4 + 3] = 0;
      }
      posBuf.value.needsUpdate = true;
      firstLayout = false;
    }

    homeBuf.value.needsUpdate = true;
    attrBuf.value.needsUpdate = true;
    nodeCount = n;
    layoutGen++;
    nodeUpdate.count = n;
    if (pixelStage) pixelStage.seed.count = pixelStage.padEdgePass.count = pixelStage.padScatter.count = n;
    writeGroups(true);
  };

  const groupData = groupBuf.value.array;
  const writeGroups = (force) => {
    let dirty = force;
    for (let gi = 0; gi < groups.length; gi++) {
      const g = groups[gi];
      const o = gi * 8;
      if (groupData[o] !== g.hover) dirty = true;
      groupData.set([g.hover, g.type === "link" ? 1 : 0, g.restTone, g.hoverTone, ...g.bar], o);
    }
    if (dirty) groupBuf.value.needsUpdate = true;
  };

  // ----------------------------------------------------------------- sizing

  let tooBig = false; // the view outgrew the GPU's buffers: effect off for good
  const resize = () => {
    if (tooBig) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.ceil(innerWidth);
    const h = Math.ceil(innerHeight);
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    uDpr.value = dpr;
    uW.value = w;
    uH.value = h;
    uFW.value = Math.ceil(w / FIELD_CELL);
    uFH.value = Math.ceil(h / FIELD_CELL);
    fieldUpdate.count = Math.min(fieldCap, uFW.value * uFH.value);
    const ww = Math.min(Math.ceil(w / WAVE_CELL) + 12, Math.floor(waveCap / (Math.ceil(h / WAVE_CELL) + 12)));
    const wh = Math.ceil(h / WAVE_CELL) + 12;
    if (ww !== uWW.value || wh !== uWH.value) uWaveReset.value = 1; // (the grid is re-indexed)
    uWW.value = ww;
    uWH.value = wh;
    for (const step of waveSteps.flat()) step.count = ww * wh;
    wavePadPass.count = ww * wh;
    uFacetW.value = Math.ceil(w / FACET) + 6;
    uFacetH.value = Math.min(Math.ceil(h / FACET) + 6, Math.floor(facetCap / uFacetW.value));
    facetUpdate.count = facetSpread.count = uFacetW.value * uFacetH.value;

    if (!pixelStage || w * h > pixelStage.capacity) {
      // Room for the whole screen (so maximizing doesn't rebuild), within the
      // device's buffer limit (the slot grid is the biggest buffer). A view
      // too big for the device turns the effect off rather than erroring.
      const { maxStorageBufferBindingSize, maxBufferSize } = renderer.backend.device.limits;
      const maxCells = Math.floor(Math.min(maxStorageBufferBindingSize, maxBufferSize) / (4 * SLOTS));
      if (w * h > maxCells) {
        console.warn(`Voronoi: ${w}x${h} view is too big for this GPU's buffers; effect disabled`);
        tooBig = true;
        renderer.setAnimationLoop(null);
        canvas.style.display = "none";
        root.classList.remove("fx-on");
        return;
      }
      pixelStage = buildPixelStage(Math.min(Math.ceil(Math.max(w * h, screen.width * screen.height) * 1.05), maxCells));
      pixelStage.pads.scene.add(flyMesh); // (dragonflies ride in the pad pass)
      pixelStage.seed.count = pixelStage.padEdgePass.count = pixelStage.padScatter.count = nodeCount;
    }
    uNW.value = Math.ceil(w / 2);
    uNH.value = Math.ceil(h / 2);
    const blocks = uNW.value * uNH.value;
    pixelStage.clear.count = w * h;
    for (const pass of pixelStage.passes) pass.count = blocks;
    pixelStage.neighbours.count = blocks;
  };

  resize();
  if (tooBig) return;
  layout();

  let layoutTimer = 0;
  addEventListener("resize", () => {
    resize();
    clearTimeout(layoutTimer);
    layoutTimer = setTimeout(layout, 150);
  });
  narrow.addEventListener("change", () => setTimeout(layout, 0));

  // ------------------------------------------------------------------ input

  const mouse = { x: 0, y: 0, active: false, fresh: true };
  const prev = { x: 0, y: 0 };
  addEventListener("pointermove", (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    if (!mouse.active) mouse.fresh = true;
    mouse.active = true;
  });
  const deactivate = () => (mouse.active = false);
  document.addEventListener("pointerleave", deactivate);
  addEventListener("blur", deactivate);
  addEventListener("pointerup", (e) => e.pointerType === "touch" && deactivate());
  // A click or tap splashes the water (the oldest splash slot is reused).
  // (Document px; strength 1 is a click's; negative, a tiny drop, see
  // TINY_DROP.)
  let nextDrop = 0;
  const splash = (x, y, strength) => {
    dropData[nextDrop].set(x, y, uTime.value, strength);
    nextDrop = (nextDrop + 1) % MAX_DROPS;
  };
  addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || e.target.closest?.(".dev-menu")) return;
    splash(e.clientX + scrollX, e.clientY + scrollY, 1);
  });

  // ------------------------------------------------------------------ ducks

  // Each duck enters from a random edge, heads for a random point on the
  // opposite edge with a lazily swaying heading, and leaves (a new one
  // arrives every so often, see DUCKS_IDLE, while there's a free slot). Ducks turn to
  // keep clear of each other.
  //
  // About half of them bring a brood of 3-4 ducklings (the last slots).
  // Each duckling follows the duck in front of it on a short
  // rope, so they trail in a line. Waves push ducklings much harder than
  // ducks; one knocked too far from its place (or startled by a hard push)
  // loses its line, which closes up behind it. After a moment's daze it
  // hurries to the nearest duck and joins the end of that duck's line.
  //
  // Raccoons (see updateCoon) hunt the ducklings; ducklings flee them and
  // ducks steer well clear.
  const ducks = Array.from({ length: DUCK_SLOTS }, (_, i) => ({
    active: false,
    chick: isChickSlot(i),
    coon: isCoonSlot(i),
    leader: -1,
    size: 1,
  }));
  // (The first duck comes soon and the second not long after.)
  let nextDuck = 1.5 + Math.random() * 3;
  let arrivals = 0;
  const extents = Array.from({ length: DUCK_SLOTS }, (_, i) => slotExtent(i));
  // Centre-to-centre distance of duckling i swimming behind duck j (clear of
  // each other's shove ellipses, see nodeUpdate).
  const gapBehind = (j, i) => extents[j].back + extents[i].front + (isChickSlot(j) ? 6 : 10);
  // ?ducktest drops the first duck straight into open water, for tuning
  // (always with a brood).
  const duckTest = params.has("ducktest");
  if (duckTest) nextDuck = 0;
  const between = (a, b) => a + Math.random() * (b - a);
  const offScreen = (d, m) => {
    const vx = d.x - scrollX;
    const vy = d.y - scrollY;
    return vx < -m || vy < -m || vx > innerWidth + m || vy > innerHeight + m;
  };
  // A duck in a line: a big duck, or a duckling whose leaders lead to one.
  const motherOf = (i) => {
    for (let k = 0; k <= DUCK_SLOTS && i >= 0; k++) {
      const d = ducks[i];
      if (!d.active || d.coon || d.eaten) return -1;
      if (!d.chick) return i;
      i = d.leader;
    }
    return -1;
  };
  // The last duck of i's line.
  const tailOf = (i) => {
    for (let k = 0; k <= DUCK_SLOTS; k++) {
      const next = ducks.findIndex((c) => c.active && c.chick && c.leader === i);
      if (next < 0) return i;
      i = next;
    }
    return i;
  };
  // Duckling i leaves its line; the one behind it moves up.
  const loseLine = (c, i) => {
    for (const b of ducks) if (b.active && b.chick && b.leader === i) b.leader = c.leader;
    c.leader = -1;
    c.daze = 0.35;
  };
  const spawnChick = (c, x, y, heading, leader) => {
    Object.assign(c, {
      active: true,
      age: 0,
      x,
      y,
      heading,
      leader,
      phase: Math.random() * 100,
      sx: 0, // swimming velocity
      sy: 0,
      vx: 0,
      vy: 0,
      bend: 0,
      dx: 0,
      dy: 0,
      daze: 0,
      alarm: 0, // how scared of a raccoon (0-1)
      eaten: 0, // s since a raccoon caught it (0 = not caught)
      by: -1, // (which raccoon)
      size: 1,
    });
  };
  // A way across the view: in at a random edge (margin m outside it), out
  // at a random point on the opposite one. Ducks live in document
  // coordinates (they stay put in the pond when the page scrolls); they
  // enter at the edge of whatever is in view.
  const route = (m) => {
    const w = innerWidth;
    const h = innerHeight;
    const edge = (side) =>
      [
        [-m, between(0.1, 0.9) * h],
        [w + m, between(0.1, 0.9) * h],
        [between(0.1, 0.9) * w, -m],
        [between(0.1, 0.9) * w, h + m],
      ][side];
    const side = Math.floor(Math.random() * 4);
    const [sx, sy] = edge(side);
    const [tx, ty] = edge(side ^ 1);
    return { x: sx + scrollX, y: sy + scrollY, heading: Math.atan2(ty - sy, tx - sx) };
  };
  const spawnDuck = (d, i) => {
    const r = route(DUCK_R * 1.5);
    Object.assign(d, {
      active: true,
      age: 0,
      x: r.x,
      y: r.y,
      base: r.heading,
      heading: r.heading,
      speed: between(0.75, 1.25) * DUCK_SPEED,
      phase: Math.random() * 100,
      vx: 0,
      vy: 0,
      bend: 0,
      dx: 0, // drift from waves
      dy: 0,
      startle: 0,
    });
    if (duckTest && i === 0) {
      // ?ducktest=x,y,heading (fractions of the view, radians) to aim it.
      const [fx = 0.7, fy = 0.45, fh = 0] = (params.get("ducktest") || "").split(",").filter(Boolean).map(Number);
      Object.assign(d, { x: scrollX + innerWidth * fx, y: scrollY + innerHeight * fy, base: fh, heading: fh, age: 3 });
    }
    // The brood, lined up behind her.
    if (Math.random() < BROOD_CHANCE || (duckTest && i === 0)) {
      const free = [];
      ducks.forEach((c, ci) => c.chick && !c.active && free.push(ci));
      const n = Math.min(free.length, Math.random() < 0.5 ? 3 : 4);
      let lead = i;
      let x = d.x;
      let y = d.y;
      for (let k = 0; k < n; k++) {
        const ci = free[k];
        const g = gapBehind(lead, ci);
        x -= Math.cos(d.heading) * g;
        y -= Math.sin(d.heading) * g;
        spawnChick(ducks[ci], x, y, d.heading, lead);
        lead = ci;
      }
    }
  };
  // Waves push a duck: the water's push (wave energy flux around it, from
  // the GPU) drives a drift velocity that water drag slows down. Returns how
  // fast it's drifting.
  // Signed turn from angle b to angle a, in (-pi, pi].
  const angleTo = (a, b) => Math.atan2(Math.sin(a - b), Math.cos(a - b));
  const waveDrift = (d, i, gain, cap, dragRate, dt) => {
    const w = duckPush[i];
    let ax = w.x * gain;
    let ay = w.y * gain;
    const al = Math.hypot(ax, ay);
    if (al > cap) {
      ax *= cap / al;
      ay *= cap / al;
    }
    const drag = Math.exp(-dt * dragRate);
    d.dx = (d.dx + ax * dt) * drag;
    d.dy = (d.dy + ay * dt) * drag;
    return Math.hypot(d.dx, d.dy);
  };
  const updateDuck = (d, i, dt, t) => {
    d.age += dt;
    // A pushed duck turns to go with the waves, the more the harder it's
    // pushed, and paddles a little faster for a moment, as if startled.
    const drift = waveDrift(d, i, DUCK_WAVE_PUSH, 160, 2.2, dt);
    if (drift > 2) {
      const want = Math.atan2(d.dy, d.dx);
      const diff = Math.atan2(Math.sin(want - d.base), Math.cos(want - d.base));
      d.base += diff * Math.min(1, drift / 30) * (1 - Math.exp(-dt * 1.6));
    }
    d.startle = Math.max(d.startle * Math.exp(-dt * 0.8), Math.min(1, drift / 40));
    // Give other ducks room: steer aside from one near ahead, the harder the
    // nearer, and get nudged apart if they still touch.
    let px = 0;
    let py = 0;
    ducks.forEach((o, j) => {
      if (j === i || !o.active || o.chick) return;
      const ax = d.x - o.x;
      const ay = d.y - o.y;
      const al = Math.hypot(ax, ay) || 1;
      if (o.coon) {
        // A raccoon: turn away from it (not just aside), harder the nearer,
        // and hurry, so her ducklings are led away too.
        const near = 1 - al / DUCK_WARY;
        if (near > 0) {
          d.base += angleTo(Math.atan2(ay, ax), d.base) * Math.min(1, near * 1.6) * (1 - Math.exp(-dt * 2.5));
          d.startle = Math.max(d.startle, Math.min(1, near * 1.4));
        }
        return;
      }
      const minD = extents[i].rx + extents[j].ry + 8;
      const fx = Math.cos(d.heading);
      const fy = Math.sin(d.heading);
      if (al < minD * 2.5 && fx * ax + fy * ay < 0) {
        const side = fx * ay - fy * ax; // which side of us it's on
        d.base += Math.sign(side || 1) * (1 - al / (minD * 2.5)) * 1.2 * dt;
      }
      if (al < minD) {
        px += (ax / al) * (minD - al) * 3;
        py += (ay / al) * (minD - al) * 3;
      }
    });
    // Weave: a slow meander plus a quicker side-to-side.
    const sway = 0.7 * Math.sin(t * 0.45 + d.phase) + 0.35 * Math.sin(t * 1.15 + d.phase * 1.7);
    const before = d.heading;
    d.heading += (d.base + sway - d.heading) * (1 - Math.exp(-dt * 2));
    // Body bend follows the turn rate (curvature of the path, exaggerated).
    const turn = (d.heading - before) / Math.max(dt, 1e-3);
    d.bend += (Math.max(-0.012, Math.min(0.012, (turn / d.speed) * 0.9)) - d.bend) * (1 - Math.exp(-dt * 5));
    // Paddle-and-glide rhythm.
    const v = d.speed * (0.8 + 0.25 * Math.sin(t * 2.2 + d.phase)) * (1 + 0.6 * d.startle);
    d.vx = Math.cos(d.heading) * v + d.dx + px;
    d.vy = Math.sin(d.heading) * v + d.dy + py;
    d.x += d.vx * dt;
    d.y += d.vy * dt;
    // She leaves once she and her whole brood are out of view.
    const brood = ducks.some((c, ci) => c.active && c.chick && motherOf(ci) === i && !offScreen(c, DUCK_R));
    if (d.age > 2 && offScreen(d, DUCK_R * 2) && !brood) d.active = false;
  };
  const updateChick = (c, i, dt, t) => {
    c.age += dt;
    if (c.eaten > 0) {
      // Caught: drawn into the raccoon's mouth, shrinking away to nothing.
      const R = ducks[c.by];
      c.eaten += dt;
      const k = 1 - Math.exp(-dt * 18);
      c.x += (R.x + Math.cos(R.heading) * COON_MOUTH * R.size - c.x) * k;
      c.y += (R.y + Math.sin(R.heading) * COON_MOUTH * R.size - c.y) * k;
      c.vx = R.vx;
      c.vy = R.vy;
      c.bend = 0;
      c.size = 0.97 * Math.max(0, 1 - (c.eaten / 0.45) ** 2);
      if (c.eaten > 0.5) Object.assign(c, { active: false, eaten: 0, size: 1 });
      return;
    }
    // Light: waves shove ducklings about more than ducks (but they paddle
    // against it, so it dies down sooner).
    const drift = waveDrift(c, i, DUCK_WAVE_PUSH * DUCKLING_WAVE_PUSH, 260, 3.2, dt);
    if (c.leader >= 0 && !ducks[c.leader].active) c.leader = -1;
    let tx = c.x;
    let ty = c.y;
    let tvx = 0;
    let tvy = 0;
    let maxSpeed = DUCK_SPEED * 1.8;
    if (c.leader >= 0) {
      // Follow on a rope: a spot a gap away from the leader, on the line
      // toward this duckling but pulled round behind the leader.
      const L = ducks[c.leader];
      const ox = c.x - L.x;
      const oy = c.y - L.y;
      const ol = Math.hypot(ox, oy) || 1;
      let ux = ox / ol - Math.cos(L.heading) * 0.6;
      let uy = oy / ol - Math.sin(L.heading) * 0.6;
      const ul = Math.hypot(ux, uy) || 1;
      const g = gapBehind(c.leader, i);
      tx = L.x + (ux / ul) * g;
      ty = L.y + (uy / ul) * g;
      tvx = L.vx;
      tvy = L.vy;
      if (Math.hypot(tx - c.x, ty - c.y) > DUCKLING_LOST || drift > DUCKLING_SCATTER) loseLine(c, i);
    }
    if (c.leader < 0) {
      c.daze -= dt;
      // Lost: hurry to the nearest duck in a line, and join the end of it.
      let near = -1;
      let nd = Infinity;
      ducks.forEach((o, j) => {
        if (j === i || !o.active || motherOf(j) < 0) return;
        const dist = Math.hypot(o.x - c.x, o.y - c.y);
        if (dist < nd) {
          nd = dist;
          near = j;
        }
      });
      if (near >= 0) {
        const tail = tailOf(near);
        const T = ducks[tail];
        const g = gapBehind(tail, i);
        tx = T.x - Math.cos(T.heading) * g;
        ty = T.y - Math.sin(T.heading) * g;
        tvx = T.vx;
        tvy = T.vy;
        maxSpeed = DUCK_SPEED * DUCKLING_RUSH;
        if (Math.hypot(tx - c.x, ty - c.y) < DUCKLING_JOIN) c.leader = tail;
      } else {
        // Nobody left to follow: paddle on out of the pond.
        tvx = Math.cos(c.heading) * DUCK_SPEED;
        tvy = Math.sin(c.heading) * DUCK_SPEED;
      }
      if (c.daze > 0) {
        // (Dazed for a moment after being knocked away: just drifts.)
        tx = c.x;
        ty = c.y;
        tvx = 0;
        tvy = 0;
      }
    }
    let vx = tvx + (tx - c.x) * 2.4;
    let vy = tvy + (ty - c.y) * 2.4;
    // A raccoon! Once it sinks in (a moment's reaction), paddle away from
    // it as fast as a duckling can, off the line if need be. (One creeping
    // up is only noticed much closer; one lunging, at once.)
    let fx = 0;
    let fy = 0;
    let fear = 0;
    ducks.forEach((o) => {
      if (!o.active || !o.coon) return;
      const ax = c.x - o.x;
      const ay = c.y - o.y;
      const al = Math.hypot(ax, ay) || 1;
      const sneaky = o.mode === "stalk" || o.mode === "crouch" ? 0.55 : 1;
      const near = 1 - al / ((DUCKLING_FEAR + extents[FIRST_COON].half * 0.5) * sneaky);
      if (near > 0) {
        fx += (ax / al) * near;
        fy += (ay / al) * near;
        fear = Math.max(fear, Math.min(1, near * 2));
      }
    });
    c.alarm += (fear - c.alarm) * (1 - Math.exp(-dt * (fear > c.alarm ? 5 : 0.8)));
    if (c.alarm > 0.01) {
      const fl = Math.hypot(fx, fy) || 1;
      vx += (fx / fl) * c.alarm * DUCK_SPEED * 4;
      vy += (fy / fl) * c.alarm * DUCK_SPEED * 4;
      maxSpeed = Math.max(maxSpeed, DUCK_SPEED * (1.8 + 0.9 * c.alarm));
    }
    const vl = Math.hypot(vx, vy);
    if (vl > maxSpeed) {
      vx *= maxSpeed / vl;
      vy *= maxSpeed / vl;
    }
    // Keep out of other ducks' way (and off a raccoon's back and tail; its
    // mouth is another matter).
    ducks.forEach((o, j) => {
      if (j === i || !o.active || o.eaten) return;
      let ox = o.x;
      let oy = o.y;
      let minD = extents[i].rx + extents[j].ry + 4;
      if (o.coon) {
        const hx = Math.cos(o.heading);
        const hy = Math.sin(o.heading);
        // (Nearest point on its spine, from its middle back to near the
        // tail tip.)
        const along = Math.max(-(extents[j].back - extents[j].ry) * o.size, Math.min(0, (c.x - o.x) * hx + (c.y - o.y) * hy));
        ox = o.x + hx * along;
        oy = o.y + hy * along;
        minD = extents[i].ry + extents[j].ry * o.size + 2;
      }
      const ax = c.x - ox;
      const ay = c.y - oy;
      const al = Math.hypot(ax, ay) || 1;
      if (al < minD) {
        vx += (ax / al) * (minD - al) * 5;
        vy += (ay / al) * (minD - al) * 5;
      }
    });
    // Little paddling spurts.
    const spurt = 0.85 + 0.3 * Math.sin(t * 4.5 + c.phase);
    const k = 1 - Math.exp(-dt * 4);
    c.sx += (vx * spurt - c.sx) * k;
    c.sy += (vy * spurt - c.sy) * k;
    const before = c.heading;
    const sl = Math.hypot(c.sx, c.sy);
    if (sl > 4) {
      const want = Math.atan2(c.sy, c.sx);
      c.heading += Math.atan2(Math.sin(want - c.heading), Math.cos(want - c.heading)) * (1 - Math.exp(-dt * 6));
    }
    const turn = (c.heading - before) / Math.max(dt, 1e-3);
    c.bend += (Math.max(-0.03, Math.min(0.03, (turn / Math.max(sl, 20)) * 0.9)) - c.bend) * (1 - Math.exp(-dt * 5));
    c.vx = c.sx + c.dx;
    c.vy = c.sy + c.dy;
    c.x += c.vx * dt;
    c.y += c.vy * dt;
    // Gone once out of view with no line to swim in (its mother has left).
    if (motherOf(i) < 0 && offScreen(c, DUCK_R)) c.active = false;
  };
  // A raccoon comes in from an edge like a duck, but slower, sniffing from
  // side to side (real ones swim well but slowly, head up, tail floating out
  // behind). While there's room in its belly it hunts:
  //   cruise: swims on; spots the nearest duckling within RACCOON_NOTICE
  //   stalk: creeps toward where the duckling is going, until within
  //     RACCOON_POUNCE of its mouth
  //   crouch: stops short for a beat, fixed on it (the wind-up that sells
  //     the lunge)
  //   lunge: a quick burst at it, homing a little; any duckling within
  //     RACCOON_CATCH of its mouth is caught, else it overshoots
  //   recover: coasts, shakes it off, then stalks again (or gives up)
  //   munch: stops and wriggles happily while the duckling shrinks into its
  //     mouth with a little splash, and fills out a bit (RACCOON_GROW); then
  //     a rest before hunting again, or, full, it leaves.
  // A hard push from the waves spooks it off the hunt for a while, so people
  // can save ducklings.
  const coonTest = params.has("raccoontest");
  let nextCoon = coonTest ? 1.5 : between(25, 45);
  const spawnCoon = (r, i) => {
    const w = route(DUCK_R * 2.5);
    Object.assign(r, {
      active: true,
      age: 0,
      x: w.x,
      y: w.y,
      base: w.heading,
      heading: w.heading,
      phase: Math.random() * 100,
      vx: 0,
      vy: 0,
      dx: 0,
      dy: 0,
      bend: 0,
      wiggle: 0,
      spd: RACCOON_SPEED * DUCK_SPEED,
      mode: "cruise",
      modeT: 0,
      rest: 2, // s before it starts hunting
      prey: -1,
      hunt: 0, // s spent on this hunt
      meals: 0,
      size: 1,
    });
    if (coonTest && i === FIRST_COON) {
      // ?raccoontest=x,y,heading (fractions of the view, radians) to aim it.
      const [fx = 0.25, fy = 0.5, fh = 0] = (params.get("raccoontest") || "").split(",").filter(Boolean).map(Number);
      Object.assign(r, { x: scrollX + innerWidth * fx, y: scrollY + innerHeight * fy, base: fh, heading: fh, age: 3, rest: 0.5 });
    }
  };
  const updateCoon = (r, i, dt, t) => {
    r.age += dt;
    r.modeT += dt;
    r.rest -= dt;
    const set = (mode) => {
      r.mode = mode;
      r.modeT = 0;
    };
    // Heavier than a duck: waves push it less, but a hard push spooks it.
    const drift = waveDrift(r, i, DUCK_WAVE_PUSH * 0.6, 120, 2.6, dt);
    if (drift > 28 && r.mode !== "munch" && r.mode !== "leave") {
      if (r.mode !== "cruise") set("cruise");
      r.prey = -1;
      r.rest = Math.max(r.rest, 4);
      r.base += angleTo(Math.atan2(r.dy, r.dx), r.base) * (1 - Math.exp(-dt * 2));
    }
    const mx = r.x + Math.cos(r.heading) * COON_MOUTH * r.size;
    const my = r.y + Math.sin(r.heading) * COON_MOUTH * r.size;
    const P = r.prey >= 0 ? ducks[r.prey] : null;
    const preyOk = P && P.active && !P.eaten;
    const preyDist = preyOk ? Math.hypot(P.x - mx, P.y - my) : Infinity;
    let speed = RACCOON_SPEED; // (x DUCK_SPEED)
    let turn = 1.5; // how fast it turns to its heading (1/s)
    let sniff = 1; // how much it weaves
    if (r.mode === "cruise") {
      if (r.rest <= 0 && r.meals < RACCOON_FULL) {
        let best = -1;
        let bd = RACCOON_NOTICE;
        ducks.forEach((c, j) => {
          if (!c.active || !c.chick || c.eaten || offScreen(c, -10)) return;
          const dd = Math.hypot(c.x - mx, c.y - my);
          if (dd < bd) {
            bd = dd;
            best = j;
          }
        });
        if (best >= 0) {
          r.prey = best;
          r.hunt = 0;
          set("stalk");
        }
      }
    } else if (r.mode === "stalk") {
      r.hunt += dt;
      if (!preyOk || preyDist > RACCOON_NOTICE * 1.5 || r.hunt > 15) {
        // Gave up.
        r.prey = -1;
        r.rest = between(3, 6);
        set("cruise");
      } else {
        const lead = Math.min(preyDist / (RACCOON_STALK * DUCK_SPEED), 1);
        r.base = Math.atan2(P.y + P.vy * lead - my, P.x + P.vx * lead - mx);
        speed = RACCOON_STALK;
        turn = 3;
        sniff = 0.2;
        if (preyDist < RACCOON_POUNCE) set("crouch");
      }
    } else if (r.mode === "crouch") {
      speed = 0.1;
      turn = 6;
      sniff = 0;
      if (preyOk) r.base = Math.atan2(P.y - my, P.x - mx);
      if (!preyOk) set("cruise");
      else if (r.modeT > 0.28) set("lunge");
    } else if (r.mode === "lunge") {
      const k = Math.min(r.modeT / 0.5, 1);
      speed = RACCOON_LUNGE * (1 - 0.5 * k * k);
      turn = 2.5;
      sniff = 0;
      if (preyOk) r.base = Math.atan2(P.y - my, P.x - mx);
      let caught = -1;
      let cd = RACCOON_CATCH * r.size;
      ducks.forEach((c, j) => {
        if (!c.active || !c.chick || c.eaten) return;
        const dd = Math.hypot(c.x - mx, c.y - my);
        if (dd < cd) {
          cd = dd;
          caught = j;
        }
      });
      if (caught >= 0) {
        const c = ducks[caught];
        loseLine(c, caught);
        Object.assign(c, { eaten: 1e-3, by: i, size: 0.97 });
        splash(mx, my, 0.15);
        r.meals++;
        r.prey = -1;
        set("munch");
      } else if (r.modeT > 0.5) set("recover");
    } else if (r.mode === "recover") {
      speed = 0.35;
      sniff = 0.3;
      if (r.modeT > 0.9) {
        if (preyOk && r.hunt < 15) set("stalk");
        else {
          r.rest = between(2, 4);
          set("cruise");
        }
      }
    } else if (r.mode === "munch") {
      speed = 0.05;
      sniff = 0;
      if (r.modeT > 1.4) {
        r.rest = between(5, 9);
        if (r.meals < RACCOON_FULL) set("cruise");
        else {
          // Full: off to the nearest edge.
          const vx = r.x - scrollX;
          const vy = r.y - scrollY;
          const ways = [
            [vx, Math.PI],
            [innerWidth - vx, 0],
            [vy, -Math.PI / 2],
            [innerHeight - vy, Math.PI / 2],
          ].sort((a, b) => a[0] - b[0]);
          r.base = ways[0][1];
          set("leave");
        }
      }
    } else if (r.mode === "leave") {
      speed = 0.85;
    }
    // (Happy wriggle while munching.)
    r.wiggle = r.mode === "munch" ? 0.012 * Math.sin(r.modeT * 16) * Math.max(0, 1 - r.modeT / 1.4) : 0;
    r.size += (1 + RACCOON_GROW * r.meals - r.size) * (1 - Math.exp(-dt * 1.2));
    const sway = sniff * (0.45 * Math.sin(t * 0.5 + r.phase) + 0.2 * Math.sin(t * 1.3 + r.phase * 1.7));
    const before = r.heading;
    r.heading += angleTo(r.base + sway, r.heading) * (1 - Math.exp(-dt * turn));
    const turnRate = angleTo(r.heading, before) / Math.max(dt, 1e-3);
    r.spd += (speed * DUCK_SPEED - r.spd) * (1 - Math.exp(-dt * (r.mode === "lunge" ? 12 : 3)));
    r.bend += (Math.max(-0.008, Math.min(0.008, (turnRate / Math.max(r.spd, 20)) * 0.9)) - r.bend) * (1 - Math.exp(-dt * 5));
    // Steady dog-paddle, and room for the other raccoon.
    const v = r.spd * (0.9 + 0.12 * Math.sin(t * 1.8 + r.phase));
    let px = 0;
    let py = 0;
    ducks.forEach((o, j) => {
      if (j === i || !o.active || !o.coon) return;
      const ax = r.x - o.x;
      const ay = r.y - o.y;
      const al = Math.hypot(ax, ay) || 1;
      const minD = extents[i].ry * 2 + 10;
      if (al < minD) {
        px += (ax / al) * (minD - al) * 3;
        py += (ay / al) * (minD - al) * 3;
      }
    });
    r.vx = Math.cos(r.heading) * v + r.dx + px;
    r.vy = Math.sin(r.heading) * v + r.dy + py;
    r.x += r.vx * dt;
    r.y += r.vy * dt;
    if (r.age > 3 && offScreen(r, DUCK_R * 3) && (r.mode === "cruise" || r.mode === "leave")) r.active = false;
  };
  const updateDucks = (dt, t) => {
    ducks.forEach((d, i) => d.active && (d.chick ? updateChick : d.coon ? updateCoon : updateDuck)(d, i, dt, t));
    // (Ducks swim even with reduced motion: they're slow and gentle.)
    nextDuck -= dt;
    if (nextDuck <= 0) {
      const i = ducks.findIndex((d) => !d.chick && !d.coon && !d.active);
      if (i >= 0) spawnDuck(ducks[i], i);
      // (Crossing time: about the view's mean side plus the margins, at a
      // duck's average pace.)
      const cross = ((innerWidth + innerHeight) / 2 + 3 * DUCK_R) / (0.8 * DUCK_SPEED);
      nextDuck = arrivals++ === 0 ? between(4, 9) : (cross / DUCKS_IDLE) * between(0.7, 1.3);
    }
    // Raccoons only come while there are ducklings in view to hunt.
    nextCoon -= dt;
    if (nextCoon <= 0) {
      const i = ducks.findIndex((d) => d.coon && !d.active);
      const prey = ducks.some((c) => c.active && c.chick && !c.eaten && !offScreen(c, 0));
      if (i >= 0 && (prey || coonTest)) {
        spawnCoon(ducks[i], i);
        nextCoon = between(...RACCOON_EVERY);
      } else nextCoon = 3;
    }
    let n = 0;
    ducks.forEach((d, i) => {
      if (d.active) {
        // (The shaders take ducks in viewport coordinates.)
        duckData[i].set(d.x - scrollX, d.y - scrollY, Math.cos(d.heading), Math.sin(d.heading));
        duckVel[i].set(d.vx, d.vy, d.bend + (d.wiggle || 0), d.size);
        // (A swallowed duckling nearly gone no longer pushes anything.)
        if (d.size > 0.02) duckList[n++].set(i, d.chick ? 1 : d.coon ? 2 : 0, 0, 0);
      } else {
        duckData[i].set(-1e5, -1e5, 1, 0);
        duckVel[i].set(0, 0, 0, 1);
      }
    });
    uDuckCount.value = n;
    if (duckTest || coonTest) {
      const d = ducks[0];
      window.__duck = { x: d.x - scrollX, y: d.y - scrollY, h: d.heading, dx: d.dx, dy: d.dy };
      window.__ducks = ducks.map((c) => c.active && { x: c.x - scrollX, y: c.y - scrollY, leader: c.leader, chick: c.chick, coon: c.coon, mode: c.mode, size: c.size, eaten: c.eaten, meals: c.meals });
    }
  };

  // ------------------------------------------------------------ dragonflies

  // (See MAX_FLIES.) Each comes in from an edge and darts about: a quick
  // spring to a point, a dead stop, a hover (bobbing, glancing about), the
  // next dart. While hovering it looks for a lily pad, asking the GPU what's
  // under random points nearby (flyProbe; answers arrive a frame or two
  // later); given a free one it darts over it, settles onto it (wings
  // slowing, shadow drawing in) with a small ripple and sits FLY_PERCH s,
  // riding the pad. It watches the pad (flyProbe again) and is off at once,
  // with another ripple, if the pad moves faster than FLY_SPOOK or drifts
  // FLY_SHIFT from where it settled (a ripple, a duck shoving it, the
  // cursor's wake) or stops being a pad (relayout). After 1-3 pads it leaves.
  const flyTest = params.has("flytest");
  if (flyTest) window.__splash = (x, y, w) => splash(x + scrollX, y + scrollY, w);
  const flies = Array.from({ length: MAX_FLIES }, () => ({ active: false }));
  let nextFly = flyTest ? 0.5 : between(5, 10);
  let flyBusy = false;
  const readFlies = () => {
    if (flyBusy) return;
    flyBusy = true;
    renderer
      .getArrayBufferAsync(flyOut.value)
      .then((ab) => {
        const o = new Float32Array(ab);
        const now = performance.now();
        flies.forEach((f, i) => {
          if (!f.active) return;
          const k = i * 8;
          if (o[k] >= 0 && f.looking) f.found = { id: o[k], x: o[k + 1], y: o[k + 2], r: o[k + 3], t: now, gen: layoutGen };
          f.watch = { x: o[k + 4], y: o[k + 5], speed: o[k + 6], id: o[k + 7] };
        });
      })
      .finally(() => (flyBusy = false));
  };
  // A dart from where it is to (tx, ty) at about `speed` px/s. Real
  // dragonflies fly direct but not ruler-straight: each dart bows into a
  // gentle arc (to a random side), speeds up and slows down, and flutters a
  // little sideways on the way. A bolt (startled off its pad) starts at full
  // speed instead and only slows toward the end.
  const startDart = (f, tx, ty, speed, maxTime = 1.4, bolt = false) => {
    const d = Math.hypot(tx - f.x, ty - f.y);
    Object.assign(f, {
      sx: f.x,
      sy: f.y,
      tx,
      ty,
      bow: (Math.random() < 0.5 ? -1 : 1) * between(0.1, 0.3),
      dartT: 0,
      dartFor: Math.min(Math.max(d / speed, bolt ? 0.2 : 0.3), maxTime),
      bolt,
    });
  };
  // Somewhere lo-hi px away to dart to, in view.
  const pickDart = (f, lo, hi, speed = between(300, 420), bolt = false) => {
    const m = 50;
    for (let k = 0; k < 12; k++) {
      const a = Math.random() * Math.PI * 2;
      const d = between(lo, hi);
      const tx = f.x + Math.cos(a) * d;
      const ty = f.y + Math.sin(a) * d;
      if (tx > scrollX + m && tx < scrollX + innerWidth - m && ty > scrollY + m && ty < scrollY + innerHeight - m) {
        startDart(f, tx, ty, speed, 1.4, bolt);
        return;
      }
    }
    startDart(f, scrollX + innerWidth * between(0.2, 0.8), scrollY + innerHeight * between(0.2, 0.8), speed, 1.4, bolt);
  };
  const spawnFly = (f) => {
    const w = route(40);
    Object.assign(f, {
      active: true,
      age: 0,
      x: w.x,
      y: w.y,
      vx: 0,
      vy: 0,
      heading: w.heading,
      alt: FLY_ALT,
      mode: "dart",
      modeT: 0,
      pad: -1, // the pad it's sitting on
      target: -1, // the pad it's going for
      ox: 0, // its spot on the pad
      oy: 0,
      found: null,
      watch: null,
      looking: false,
      perches: 0,
      maxPerches: flyTest ? 9 : 1 + Math.floor(Math.random() * 3),
      phase: Math.random() * 100,
    });
    pickDart(f, 150, 300);
  };
  // Along the dart: a point that eases along the bowed path (a quadratic
  // curve), which the dragonfly follows on a stiff spring. Returns how far
  // it still is from the end.
  const flyAlong = (f, dt, k = 90, cap = 560) => {
    if (f.bolt) {
      // (A bolt: a stiffer spring and a higher top speed.)
      k = 160;
      cap = 1000;
    }
    f.dartT += dt;
    const k1 = Math.min(f.dartT / f.dartFor, 1);
    const u = f.bolt ? 1 - (1 - k1) ** 3 : k1 * k1 * (3 - 2 * k1);
    const dx = f.tx - f.sx;
    const dy = f.ty - f.sy;
    // (The curve's middle control point: off to one side of the straight
    // line, by `bow` of its length; plus a little flutter mid-dart.)
    const flutter = Math.sin(f.dartT * 13 + f.phase) * 1.5 * Math.sin(u * Math.PI);
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;
    const cx = f.sx + dx / 2 + nx * f.bow * len;
    const cy = f.sy + dy / 2 + ny * f.bow * len;
    const px = (1 - u) * (1 - u) * f.sx + 2 * u * (1 - u) * cx + u * u * f.tx + nx * flutter;
    const py = (1 - u) * (1 - u) * f.sy + 2 * u * (1 - u) * cy + u * u * f.ty + ny * flutter;
    const c = 2 * Math.sqrt(k) * 0.9;
    f.vx += ((px - f.x) * k - f.vx * c) * dt;
    f.vy += ((py - f.y) * k - f.vy * c) * dt;
    const sp = Math.hypot(f.vx, f.vy);
    if (sp > cap) {
      f.vx *= cap / sp;
      f.vy *= cap / sp;
    }
    f.x += f.vx * dt;
    f.y += f.vy * dt;
    return Math.hypot(f.tx - f.x, f.ty - f.y);
  };
  const faceMotion = (f, dt) => {
    if (Math.hypot(f.vx, f.vy) > 50) f.heading += angleTo(Math.atan2(f.vy, f.vx), f.heading) * (1 - Math.exp(-dt * 16));
  };
  const setFly = (f, mode) => {
    f.mode = mode;
    f.modeT = 0;
  };
  // (A gentle ring around the rim of the pad it's on.)
  const padRipple = (f) => splash(f.x - f.ox, f.y - f.oy, -(Math.max(1, Math.round(f.padR * 0.9)) + FLY_SPLASH));
  const takeOff = (f, startled) => {
    f.why = `${startled ? "spooked" : "rested"} after ${f.modeT.toFixed(1)}s`;
    if (f.pad >= 0) padRipple(f);
    f.pad = -1;
    f.target = -1;
    f.perches++;
    // Startled, it bolts: straight up off the pad and away, fast.
    if (startled) pickDart(f, 180, 320, between(650, 800), true);
    else pickDart(f, 90, 220);
    setFly(f, "takeoff");
  };
  // (Follow the pad it's going for or sitting on, from the GPU's answers.)
  const padNow = (f, id) => (f.watch && f.watch.id === id ? f.watch : null);
  const updateFly = (f, i, dt, t) => {
    f.age += dt;
    f.modeT += dt;
    f.looking = false;
    let ask = -1; // (the pad to watch)
    let probe = null; // (a viewport point to look for a pad under)
    if ((f.pad >= 0 || f.target >= 0) && f.gen !== layoutGen) {
      // The page was laid out again: pads have new ids.
      if (f.pad >= 0) takeOff(f, true);
      else {
        startDart(f, f.x, f.y, 300);
        setFly(f, "hover");
        f.hoverFor = 0.3;
      }
      f.target = -1;
    }
    if (f.mode === "dart" || f.mode === "takeoff") {
      const left = flyAlong(f, dt);
      faceMotion(f, dt);
      if (f.mode === "takeoff") {
        // (A bolt lifts off in a blink.)
        const lift = f.bolt ? 0.1 : 0.3;
        f.alt = FLY_ALT * Math.min(1, f.modeT / lift) ** 0.5;
        if (f.modeT > lift) setFly(f, "dart");
      } else if (f.dartT >= f.dartFor && left < 3 && Math.hypot(f.vx, f.vy) < 30) {
        setFly(f, "hover");
        f.hoverFor = between(0.3, 1.3);
      }
    } else if (f.mode === "hover") {
      flyAlong(f, dt);
      // (Glancing about.)
      f.heading += Math.sin(t * 2.3 + f.phase) * 0.9 * dt;
      const wants = f.perches < f.maxPerches && f.age < 75;
      if (wants) {
        f.looking = true;
        const a = Math.random() * Math.PI * 2;
        const d = between(20, 240);
        probe = [f.x + Math.cos(a) * d - scrollX, f.y + Math.sin(a) * d - scrollY];
      }
      if (f.modeT > f.hoverFor) {
        const p = f.found;
        const fresh = p && wants && performance.now() - p.t < 800 && p.gen === layoutGen && !flies.some((o) => o !== f && o.active && (o.pad === p.id || o.target === p.id));
        if (fresh) {
          // Settle a little off the pad's centre.
          const a = Math.random() * Math.PI * 2;
          const rr = p.r * 0.3 * Math.random();
          Object.assign(f, { target: p.id, gen: layoutGen, padR: p.r, ox: Math.cos(a) * rr, oy: Math.sin(a) * rr });
          startDart(f, p.x + f.ox, p.y + f.oy, between(260, 360));
          f.found = null;
          setFly(f, "approach");
        } else if (!wants) {
          // Off out of the pond, in one long sweep.
          const a = Math.random() * Math.PI * 2;
          startDart(f, f.x + Math.cos(a) * 1600, f.y + Math.sin(a) * 1600, 340, 30);
          setFly(f, "leave");
        } else {
          pickDart(f, 70, 240);
          setFly(f, "dart");
        }
      }
    } else if (f.mode === "approach" || f.mode === "land") {
      ask = f.target;
      const w = padNow(f, f.target);
      if (f.watch && f.watch.id === -2) {
        // (It stopped being a pad.)
        f.target = -1;
        f.alt = FLY_ALT;
        startDart(f, f.x, f.y, 300);
        setFly(f, "hover");
        f.hoverFor = 0.3;
      } else {
        if (w) {
          // (The pad drifts: the dart ends wherever it is now.)
          f.tx = w.x + f.ox;
          f.ty = w.y + f.oy;
        }
        const left = flyAlong(f, dt, f.mode === "land" ? 120 : 90);
        if (f.mode === "approach") {
          faceMotion(f, dt);
          if (f.dartT >= f.dartFor && left < 2.5 && Math.hypot(f.vx, f.vy) < 25) setFly(f, "land");
        } else {
          // Settling down: the shadow draws in under it, the wings slow.
          const k = Math.min(1, f.modeT / 0.45);
          f.alt = FLY_ALT * (1 - k * k * (3 - 2 * k));
          if (k >= 1) {
            f.alt = 0;
            f.pad = f.target;
            f.target = -1;
            f.base = null;
            f.perchFor = between(...FLY_PERCH);
            padRipple(f);
            setFly(f, "perched");
          }
        }
      }
    } else if (f.mode === "perched") {
      ask = f.pad;
      const w = padNow(f, f.pad);
      if (f.watch && f.watch.id === -2) takeOff(f, true);
      else if (w) {
        f.x = w.x + f.ox;
        f.y = w.y + f.oy;
        // (Its own landing ripple settles first; after that it measures
        // drift from where the pad has come to rest, which itself creeps
        // with the pads' slow ambient wander.)
        if (f.modeT < 1.2 || !f.base) f.base = { x: w.x, y: w.y };
        else {
          const k = 1 - Math.exp(-dt * 0.3);
          f.base.x += (w.x - f.base.x) * k;
          f.base.y += (w.y - f.base.y) * k;
          const spooked = w.speed > FLY_SPOOK || Math.hypot(w.x - f.base.x, w.y - f.base.y) > FLY_SHIFT;
          if (spooked || f.modeT > f.perchFor) takeOff(f, spooked);
        }
      }
    } else if (f.mode === "leave") {
      flyAlong(f, dt, 60);
      faceMotion(f, dt);
      if (f.age > 2 && offScreen(f, 80)) f.active = false;
    }
    flyAsk[i].set(probe ? probe[0] : -1e5, probe ? probe[1] : -1e5, ask, 0);
  };
  const updateFlies = (dt, t) => {
    nextFly -= dt;
    if (nextFly <= 0) {
      const f = flies.find((o) => !o.active);
      if (f) spawnFly(f);
      nextFly = between(...FLY_EVERY);
    }
    flies.forEach((f, i) => {
      if (f.active) updateFly(f, i, dt, t);
      if (!f.active) {
        flyB[i].set(-1, 0, 0, -1);
        flyAsk[i].set(-1e5, -1e5, -1, 0);
        return;
      }
      // (A hovering one bobs a little.)
      const bob = f.mode === "hover" ? 1 : 0;
      flyA[i].set(f.x - scrollX + Math.sin(t * 3.1 + f.phase) * 1.2 * bob, f.y - scrollY + Math.sin(t * 4.3 + f.phase * 2) * 1.2 * bob, f.heading, f.alt);
      const flap = f.mode === "perched" ? 0 : f.mode === "land" ? 1 - Math.min(1, f.modeT / 0.45) : 1;
      flyB[i].set(f.mode === "perched" ? f.pad : -1, f.ox, f.oy, flap);
    });
    if (flyTest) window.__flies = flies.map((f) => f.active && { x: f.x - scrollX, y: f.y - scrollY, mode: f.mode, pad: f.pad, target: f.target, perches: f.perches, why: f.why, watch: f.watch && { ...f.watch }, base: f.base && { ...f.base } });
  };

  // ------------------------------------------------------------------- loop

  let last = performance.now();
  let started = false;
  let gpuFrame = 0;
  const waterCursor = { x: 0, y: 0, speed: 0, fresh: true };

  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min((now - last) / 1000, 1 / 30);
    last = now;

    uTime.value += dt;
    uFrame.value = (uFrame.value + 1) & 1023;
    uDt.value = dt;
    uScroll.value.set(scrollX, scrollY);
    uAmbient.value = reduceMotion.matches ? 0.2 : 1;
    if (reduceMotion.matches && uTime.value < 3) uTime.value = 3;

    if (mouse.fresh) {
      prev.x = mouse.x;
      prev.y = mouse.y;
      mouse.fresh = false;
    }
    const vx = (mouse.x - prev.x) / Math.max(dt, 1e-3);
    const vy = (mouse.y - prev.y) / Math.max(dt, 1e-3);
    const speed = Math.hypot(vx, vy);
    const cap = Math.min(1, 4000 / Math.max(speed, 1));
    uMousePrev.value.set(prev.x, prev.y);
    uMouse.value.set(mouse.x, mouse.y);
    uMouseVel.value.set(vx * cap, vy * cap);
    uIntensity.value = mouse.active ? Math.min(speed / 1300, 1.25) : 0;
    // The cursor pushes on the water along its path. Pointer events don't
    // line up with frames (at slow speeds some frames see no movement and the
    // next sees double), so the water follows its own smoothed cursor, whose
    // path and speed are continuous.
    if (waterCursor.fresh || !mouse.active) {
      waterCursor.x = mouse.x;
      waterCursor.y = mouse.y;
      waterCursor.fresh = !mouse.active;
    }
    const wx0 = waterCursor.x;
    const wy0 = waterCursor.y;
    const follow = 1 - Math.exp(-dt / 0.035);
    waterCursor.x += (mouse.x - waterCursor.x) * follow;
    waterCursor.y += (mouse.y - waterCursor.y) * follow;
    const wSpeed = Math.hypot(waterCursor.x - wx0, waterCursor.y - wy0) / Math.max(dt, 1e-3);
    waterCursor.speed += (wSpeed - waterCursor.speed) * (1 - Math.exp(-dt / 0.06));
    const ws = waterCursor.speed;
    // Into the fast layer (the slow one only while barely moving: anything
    // quicker would outrun the slow waves and leave a trail, not ripples).
    const fastness = Math.min(1, Math.max(0, (ws - 20) / 60));
    // (Slow strokes make gentle ripples, quick ones bigger.)
    const push = mouse.active ? Math.min(ws / 500, 1.4) ** 0.6 * CURSOR_PUSH : 0;
    uCurA.value.set(wx0 + scrollX, wy0 + scrollY);
    uCurB.value.set(waterCursor.x + scrollX, waterCursor.y + scrollY);
    // (A steady push builds up far more than a pulsing one, so it's softer.)
    const steady = Math.min(1, Math.max(0, (ws - WAVE_SPEED[0]) / WAVE_SPEED[0]));
    uCurSteady.value = steady;
    uCurPush.value.set(push * fastness * (1 - 0.45 * steady), push * (1 - fastness));
    prev.x = mouse.x;
    prev.y = mouse.y;

    const ease = 1 - Math.exp(-dt * 12);
    for (const g of groups) g.hover += (g.target - g.hover) * ease;
    writeGroups(false);
    updateDucks(dt, uTime.value);
    updateFlies(dt, uTime.value);

    // Waves: as few steps per frame as the fast layer's stability limit
    // allows (one at 120fps, two at 60fps); a step is capped at that limit,
    // so a long frame just runs the water a hair slow. Re-anchored when the
    // page scrolls.
    const maxStep = (0.8 * WAVE_CELL) / WAVE_SPEED[0];
    const substeps = Math.min(3, Math.ceil(dt / maxStep - 0.25));
    const wdt = Math.min(dt / substeps, maxStep);
    const ox = Math.floor(scrollX / WAVE_CELL) * WAVE_CELL - 6 * WAVE_CELL;
    const oy = Math.floor(scrollY / WAVE_CELL) * WAVE_CELL - 6 * WAVE_CELL;
    uWaveShiftX.value = Math.round((ox - uWaveOrigin.value.x) / WAVE_CELL);
    uWaveShiftY.value = Math.round((oy - uWaveOrigin.value.y) / WAVE_CELL);
    const moved = ox !== uWaveOrigin.value.x || oy !== uWaveOrigin.value.y;
    uWaveOrigin.value.set(ox, oy);
    uWaveDt.value = wdt;
    uWaveK.value.set((WAVE_SPEED[0] * wdt / WAVE_CELL) ** 2, (WAVE_SPEED[1] * wdt / WAVE_CELL) ** 2);
    uWaveLevel.value.set((WAVE_LEVEL[0] * wdt) ** 2, (WAVE_LEVEL[1] * wdt) ** 2);
    uWaveVisc.value.set((WAVE_VISC[0] * wdt) / WAVE_CELL ** 2, (WAVE_VISC[1] * wdt) / WAVE_CELL ** 2);
    uFacetX0.value = Math.floor(scrollX / FACET) - 3;
    uFacetY0.value = Math.floor(scrollY / FACET) - 3;
    const waveWork = moved || uWaveReset.value ? [wavePadPass] : [];
    for (let k = 0; k < substeps; k++) {
      waveWork.push(waveSteps[waveCur][k === 0 ? 0 : 1]);
      waveCur ^= 1;
    }
    waveWork.push(duckWavePasses[waveCur]);

    if (gpuTime) {
      renderer.compute([...waveWork, facetUpdate, facetSpread]);
      renderer.compute([fieldUpdate, nodeUpdate]);
      renderer.compute([pixelStage.clear, pixelStage.seed, pixelStage.padScatter, ...pixelStage.passes, pixelStage.neighbours, pixelStage.padEdgePass, pixelStage.flyProbe]);
    } else {
      renderer.compute([...waveWork, facetUpdate, facetSpread, fieldUpdate, nodeUpdate, pixelStage.clear, pixelStage.seed, pixelStage.padScatter, ...pixelStage.passes, pixelStage.neighbours, pixelStage.padEdgePass, pixelStage.flyProbe]);
    }
    uWaveReset.value = 0;
    if (ducks.some((d) => d.active)) readDuckWaves();
    if (flies.some((f) => f.active)) readFlies();
    pixelStage.quad.render(renderer);
    pixelStage.pads.geometry.instanceCount = nodeCount;
    flyMesh.visible = flies.some((f) => f.active);
    renderer.render(pixelStage.pads.scene, padCamera);
    if (gpuTime && (gpuFrame = (gpuFrame || 0) + 1) % 30 === 0) {
      Promise.all([renderer.resolveTimestampsAsync(THREE.TimestampQuery.COMPUTE), renderer.resolveTimestampsAsync(THREE.TimestampQuery.RENDER)]).then(([c, r]) => {
        window.__gpu = { compute: renderer.info.compute.timestamp, render: renderer.info.render.timestamp, c, r };
      });
    }

    if (!started) {
      started = true;
      root.classList.add("fx-on");
    }
  });
}

// One rect per non-space character, in document coordinates.
function glyphRects(el, sx, sy) {
  const out = [];
  const range = document.createRange();
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node.textContent;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (/\s/.test(ch)) continue;
      range.setStart(node, i);
      range.setEnd(node, i + 1);
      const r = range.getClientRects()[0];
      if (!r || r.width === 0) continue;
      out.push({ ch, l: r.left + sx, t: r.top + sy, r: r.right + sx, b: r.bottom + sy });
    }
  }
  return out;
}

function mulberry32(a) {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t >>> 0) / 4294967296);
  };
}

// Bottom-right dev menu: fps meter, debug overlay toggle, effects on/off link.
// (/pond's menu is just a small fps meter that hides when clicked.)
function setupDevMenu() {
  const menu = document.querySelector(".dev-menu");
  if (!menu) return;
  const fpsEl = document.getElementById("dev-fps");
  const debugEl = document.getElementById("dev-debug");
  const fxEl = document.getElementById("dev-fx");
  const fxDisabled = params.has("nofx");

  if (menu.classList.contains("dev-mini")) {
    menu.addEventListener("click", () => menu.remove());
    startFpsMeter(fpsEl);
    return;
  }

  fxEl.href = fxDisabled ? location.pathname : "?nofx";
  fxEl.textContent = fxDisabled ? "Turn effects on" : "Turn effects off";

  debugEl.checked = params.has("debug");
  debugEl.disabled = fxDisabled;
  root.classList.toggle("fx-debug", debugEl.checked);
  debugEl.addEventListener("change", () => {
    root.classList.toggle("fx-debug", debugEl.checked);
    history.replaceState(null, "", debugEl.checked ? "?debug" : location.pathname);
  });

  startFpsMeter(fpsEl);
}

// Counts animation frames, which is the effect's frame rate when it's on.
function startFpsMeter(fpsEl) {
  let frames = 0;
  let since = performance.now();
  const tick = (now) => {
    if (!fpsEl.isConnected) return; // (hidden for good)
    frames++;
    if (now - since >= 500) {
      fpsEl.textContent = Math.round((frames * 1000) / (now - since));
      frames = 0;
      since = now;
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
