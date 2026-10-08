# Homepage pond animation (`js/voronoi.js`)

The homepage (`index.html`) is a WebGPU pond: every word, lily pad and duck
is made of cells of one big **power diagram** (weighted Voronoi) of ~30k
nodes, and the water's ripples come from a real (damped) wave simulation.
Everything is simulated and drawn on the GPU with three.js TSL compute
shaders. This doc is for future agents picking the work up.

## Files

- `index.html` — real DOM content (header, nav, links), a `<canvas id="fx">`,
  the dev menu (`.dev-menu`), and an importmap loading **three@0.186.1** from
  jsDelivr (`three/webgpu`, `three/tsl`). No build step (GitHub Pages).
- `css/home.css` — page styles. When the effect runs, `html.fx-on` makes DOM
  text transparent (it stays for layout, links, selection, a11y) and hides the
  CSS underline bars. `html.fx-debug` shows DOM text in red for alignment.
  Without WebGPU (or `?nofx`) the plain styled page shows.
- `js/voronoi.js` — everything else. `resume.html` / `css/style.css` /
  `css/index.css` are the untouched old pages.

## URL params / dev menu

`?nofx` disables the effect, `?debug` overlays DOM text, `?ducktest` spawns a
duck immediately at 70%/45% of the viewport heading right
(`?ducktest=x,y,heading` to aim it, fractions of the view / radians; it
also puts the duck's viewport position in `window.__duck`; e.g.
`?ducktest=0.5,0.38,0` sends it through a pad colony), `?gputime` turns
on GPU timestamp queries and puts `{c, r}` (compute / render ms, every 30
frames) in `window.__gpu` (splits compute into three submissions). The
bottom-right dev menu (`setupDevMenu()`) has an fps meter, debug toggle,
effects on/off link.

## Coordinates

Node positions are **document px**; the canvas is fixed to the viewport, so
shaders subtract `uScroll`. The jump-flood grid and wake field are in
**viewport css px**. The wave grid and the ripple mosaic are viewport-sized
but **anchored to the document** (re-indexed when the page scrolls, so waves
stay on the water). Ducks live in document px on the CPU (`d.x/d.y`) and are
handed to shaders in viewport px (`uDucks`). Fragment shader works in css px
(`screenCoordinate/uDpr`).

## Node kinds (`home.w`) and buffers

Kinds: 0 background (water or lily pad), 1 halo (around non-link text),
2 plate (inside link boxes; orange underline/hover bar), 3 glyph ink,
4 duck cell. (There are no ripple nodes any more.)

Node storage buffers (`instancedArray`, size `MAX_NODES = 2^17`):
- `posBuf` xy pos, zw vel (GPU-owned after first upload)
- `homeBuf` xy home, z group, w kind (CPU-written by `layout()`)
- `attrBuf` x bar coord / duck index / (background) room: distance to the
  nearest text, capped at 48 (limits pad size), y spacing, z rnd,
  w rnd | duck part (for ducks, attr.y is the cell's radius)
- `lifeBuf` x weight, y lifecycle index, z tint, w phase
- `siteBuf` xy drawn pos (incl. wave sway), z power weight, w alive (0..1;
  for pads, how far grown)
- `lookBuf` what the cell looks like, resolved once per node per frame:
  x surface code (0 water, 1.x pad (fraction = wave lift, 0.45 level),
  2..2.9 glyph (+hover darkening), 3 orange bar, 4.x duck body, 5.x duck bill (fraction = wave lift, like pads)),
  y spacing (**pad radius** for pads), z rnd, w tone (glyph) / tint
- `groupBuf` per DOM element: hover, darkens-on-hover, tones, bar ranges
- `fieldBuf` mouse/duck wake field, 8px cells: x energy, yz flow,
  **w water height** (copied from the wave sim; nodes read it from here
  because `nodeUpdate` is at the 8-storage-buffer limit)

Water buffers: `waveA`/`waveB` (ping-pong, per cell: fast h, fast h_prev,
slow h, slow h_prev), `waveView` (vec2: both heights, written by every step),
`wavePads` (extra damping in lily pad colonies, recomputed only when the grid
re-anchors), `facetBuf` (per mosaic cell: centre, crest, trough), `facetMax`
(3×3 max of crest/trough, so calm pixels skip after one read).

Uniform arrays: `uDucks` (viewport pos + heading), `uDuckVel` (vel + body
bend in z).

## Per-frame pipeline (`renderer.setAnimationLoop`)

CPU: dt, scroll, mouse → wake uniforms; smoothed **water cursor**; hover
easing → `groupBuf`; `updateDucks()`; wave grid anchor / substeps. Then one
`renderer.compute([...])`:

1. (`wavePadPass` if the wave grid moved) → `waveStep` × substeps (1 at
   120fps, 2 at 60fps: `ceil(dt / maxStep)`, step capped at the fast layer's
   CFL limit `0.8·cell/c`).
2. `facetUpdate` → `facetSpread` (ripple mosaic, see below).
3. `fieldUpdate` — splat mouse segment + ducks into the wake field, decay;
   copy water height into `.w`.
4. `nodeUpdate` — per node: lifecycle, target, forces, integrate, wave sway,
   write `siteBuf` + `lookBuf`.
5. `clear` / `seed` — scatter nodes into a css-px grid: up to `SLOTS=4`
   node ids per cell (atomic counter), plus a jump-flood seed per 2×2 block.
   `padScatter` — each pad `atomicMin`s `(quantized power distance << 17 |
   id)` into the 2×2 blocks under its disc (`padBlock`); the pixel shader adds
   that pad as a candidate in pass 1 (a heavy pad whose centre is far away
   could be missing from the block lists → flat-clipped pads).
6. Jump flood (`JFA_STEPS = [16,8,4,2,1,1]`, in 2×2 blocks, reach ~60px)
   using **power distance** `|p-node|² - weight`.
7. `neighbours` — per 2×2 block, the 4 nearest nodes to its centre, from its
   4×4 cells' slots, 2 rings of jump-flood taps (radius 4/9) and the block's
   own jump-flood result (as tap t=16). (A third ring at 15px cost ~2ms; it
   was only needed for pads.) Written with real `Loop`s (unrolling it blew
   up registers). **Fragile:** extra work in its `consider()`, or calling
   `consider()` outside the loops, silently corrupts the lists (text breaks
   into blocks) — do extra work in a separate pass instead.
8. `padEdgePass` — per lily pad node (half of them per frame, alternating):
   its 8 nearest power bisectors, gathered from the block lists of a 5×5
   grid of blocks 12px apart, stored in `padData` (5 vec4s/pad) as the 6
   nearest edge lines `(normal, offset)` plus the flower's room.

Then a full-screen `QuadMesh` with `shade()`:
- Candidates are **streamed** (`forEachCandidate`, read twice, never stored —
  storing 16 in arrays cost ~20fps): the block's 4, the pixel cell's own
  slots, and (only if the cell is dense, i.e. small text) the 8 neighbouring
  cells' first slot.
- Pass 1: nearest node by power distance. Pass 2: the two closest **edges**
  (bisector distance `(pwB - pw1) / (2|pB-p1|)`, exact for power diagrams).
- Surface first (pass 2 only runs for glyph/bar cells — most pixels are
  water/pad/duck and skip it): glyph/bar → rounded inset cell SDF from the
  two edges (joined same-kind neighbours with a faint seam, faded on small
  text; glyph cells clipped to `1.8×spacing`, bars `2.4×`). Then the ripple
  mosaic only where the surface doesn't fully cover (and not under pads),
  then composite.

Then the **lily pad pass** (`padShader`) draws all "leaves", i.e. lily pads
and duck cells (`isLeafCode`: surface 1, 4, 5): an instanced quad per node
(count = nodeCount; others emit an off-screen vertex), drawn over the main pass
(`renderer.autoClear = false`). **Why separate:** inside the full-screen
shader, the pad + flower code slowed *every* pixel (register pressure) by
~1.5ms; as its own pass only pad pixels pay. Safe to draw on top because a
leaf's shape always lies inside its own power cell, so it never covers text
or other leaves. `padScatter` / `padEdgePass` / the main shader's
ripple-hiding cover duck cells too. The material is `DoubleSide`: flipping y into screen space
reverses the winding (it was invisibly back-face culled).

## Lily pads

- Pads grow in colonies: `padField = noise(home/230) + jitter > PAD_CLUSTER`
  (0.12) (computed in `nodeUpdate`; `isPadNode`).
- **A pad is a weighted node**: power weight `padR²`. Against unweighted
  nodes (water, ducks, text) every bisector is ≥ padR away, so the whole disc
  of radius padR is its own: round leaves in open water. Between two pads the
  edge is a straight power bisector, so colonies pack into rounded Voronoi
  cells (owner wants them to read as part of the diagram; a little
  Voronoi irregularity is welcome; overlapping round leaves were rejected).
- padR = spacing × 0.52–0.78 (random), ×1.15 for big pads, limited by the
  room to the nearest text (`attr.x` from `layout()`, minus how far the pad's
  home moved) so a pad never swallows letters.
- Pads are **permanent** (no lifecycle): they rest near their grid cell's
  centre (60% of the way), so neighbours never crowd into split shapes.
- **Merging**: in `MERGE_FRACTION` (22%) of 2×2 patches of the background
  grid, one chosen node becomes a big pad at the patch centre; its
  patch-mates are water. (Joining cells into multi-cell leaves with seams
  looked shattered.)
- **Ducks part pads** instead of swallowing them: ducks shove pads out of an
  ellipse grown by the pad's radius (force 4000), so the leaf slides aside
  whole and the colony closes in behind. (Before, a duck's cells took over
  the pad cells and pads "disappeared" near ducks; owner disliked it.)
- **Intro**: pads grow from nothing (`grow`, ~0.7–2.2s after each node's
  start), weight `(padR·grow)²`; flowers open after their pad. (Full pads
  flying in during the intro looked bad.)
- Leaf shape (pad pass): cell from the 6 edge lines, inset by 1.4px, corners
  rounded 0.45·r, smooth-max with the disc (`r·grow`), notch wedge. Colour:
  two greens per pad, wave light (lookBuf.x fraction), faint radial veins
  (11, symmetric to the notch, `fastAtan2`), an upturned rim lit top-left /
  shaded bottom-right, a thin darker edge. Pixels of the cell outside the
  disc exit early.
- Flowers (20%): two 8-petal layers + accent centre, sized to the room at
  the flower's spot (`flowerRoom`, so petals are never cut by the cell edge)
  and hidden where there's too little room (no specks); not cut by the notch.
- Pads ride the waves: drawn offset by `-slope × PAD_SWAY`, lighter on crests
  / darker in troughs, small push; colonies damp waves.
- Water nodes live 14–28s, born/die over ~2–4s by easing a negative power
  weight.
- **NaN trap**: `smoothstep(a, a, x)` is undefined; a pad radius of 0 made
  NaN that rendered as black shards during the intro. Radii are clamped ≥0.5.

## Letter styling

Letters and underline bars get the pads' and ducks' lighting so the page is
cohesive: their outer outline (direction = the nearest edge to a cell of
another kind) has a rim lit top-left / shaded bottom-right and a thin darker
edge, applied as lighten/darken over the letter's own colour (works for
cream, dimmed nav, hovered dark text and orange bars). Outline corners are
rounded generously (0.55·spacing on big text) to smooth the letters' edges;
faint seams between a letter's cells keep the Voronoi visible. All of it
fades out on small text (`big = smoothstep(1.9, 3.6, spacing)`).
Tried and rejected: cracking letters into chunky pieces (a coarse procedural
Voronoi with gaps, or grouping cells per chunk): read as broken plaster /
scattered blobs, not as the pads' clean cells.

## Colours

Palettes in `PALETTES.light/dark` (water, pads, petals, ripple, text=cream,
accent=#ffb82b used for underlines, duck bill, flower centres). Colours are
sRGB values (`outputColorSpace = LinearSRGBColorSpace`, no conversion).

## Text

`layout()` rasterises each `[data-fx]` element's glyphs (per-character Range
rects + canvas `fillText` with the computed font) into a mask, samples a
jittered hex grid (spacing ≈ `fontSize*0.065`, clamped 1.8–5px): ink → kind
3, near-ink → halo/plate, link boxes get plate lattice (orange bar uses plate
`v` coordinate vs the group's animated bar range). Background nodes fill the
document on a 32px (24px narrow) jittered grid away from text. Relayout on
resize (debounced); nodes glide to new homes. Groups: title, nav, current,
link (`data-fx`), with hover/focus listeners.

## Mouse

Wake field: energy splat along the cursor segment, decays; nodes get Perlin
turbulence + flow push scaled by energy, then springs pull them home.
The water gets its own **smoothed cursor** (`waterCursor`, ~35ms follow, speed
smoothed over ~60ms): pointer events don't line up with frames, and using the
raw per-frame movement made slow strokes stutter.

## Ducks (`MAX_DUCKS=2`)

CPU steers each duck (enter from an edge of the current view, weave via two
sines, paddle-and-glide speed, exit, rest 3–10s), in **document coords** so
scrolling doesn't move them.

**Drawn like the lily pads** (owner asked for the same style): a duck is 7
big weighted cells (`duckCells`, radius r, power weight r²): a tail, two
pairs of body cells (the seam down the middle reads as folded wings), a head
over the front of the body and a bill ahead of it (`DUCK_SCALE=2`). They're
drawn by the pad pass with the pads' rendering language (rounded cell ∩
disc, lit rim top-left / shaded bottom-right, thin darker edge, wave light)
in the page's colours: text-cream body, accent-orange bill; no notch, veins
or flowers. Duck cells are narrow wedges, so they use much tighter rounding
than pads (corner 0.16·r vs 0.45·r, gap 0.55 vs 1.4px, smooth-max 3 vs 8):
with the pad values they shrank to scattered petals.

**Waves steer ducks.** `duckWavePass` (after the wave steps, on the latest
wave buffer) averages the fast layer's wave energy flux `-∂h/∂t·∇h` (the
direction waves carry floating things) over a Gaussian disc around each duck
into `duckWave`; the CPU reads it back asynchronously (`readDuckWaves`, one
read in flight at a time, a frame or two late, no fps cost) into `duckPush`.
`updateDucks` turns it into a drift velocity (`DUCK_WAVE_PUSH=600`, accel
capped 160px/s², drag 2.2/s) added to the swimming velocity, turns the duck's
base heading toward the drift (more the harder it's pushed), and adds a brief
"startled" speed-up. Net: a quick stroke passing ~45px away nudges the duck
~25px and it turns away from the disturbance over ~2s. Only the fast layer
counts, because the slow layer is mostly the duck's own paddling.
`?ducktest&duckwave` exposes the readback as `window.__duckWave`; `__duck`
also has heading and drift. (A first try at 3000 shoved the duck ~400px.)

Each cell springs to its spot in the duck's frame (head and bill bob; whole
body bends by rotating each cell by `x × bend`, bend from turn rate) with the
duck's velocity fed forward, so it holds shape but can be knocked apart.
Ducks shove other nodes out of a duck-shaped ellipse (pads out of one grown
by their radius, so they're parted, not swallowed), sway a little with
passing waves, and paddle: their feet (behind the body) pulse into the
**slow** wave layer at `WAVE_FREQ[1]`, which leaves a V wake of arcs (owner
likes this a lot). `DUCK_PRESS` (a moving body dent) is 0: it showed as a
dark dent around the duck.

## Ripples (wave simulation + mosaic)

**Simulation** (`waveStep`): leapfrog damped wave equation on a 4px grid
(`WAVE_CELL`), isotropic 9-point Laplacian, two independent layers packed in
one vec4: **fast** `c=220px/s` (cursor) and **slow** `c=30px/s` (ducks, and the
cursor only when barely moving). Two layers because real ripples aren't all
one speed and duck ripples must be slow (owner: duck ripples spreading faster
than the duck looked wrong); a single medium can't do both at resolvable
wavelengths. Terms per step:
- `(c·dt/cell)²·lap` (uWaveK), damping `WAVE_DAMP=[1.0, 0.45]`/s plus an edge
  sponge and pad-colony damping,
- viscosity `WAVE_VISC` on lap(velocity) — kills grid-scale chop (static),
- restoring `-(WAVE_LEVEL·dt)²·h` — **important**: without it the field
  accumulated broad offsets (dents/bulges), which lit up as blotches and hid
  crests,
- forces scaled by layer stiffness `c²/r²`.

**Cursor push**: along the water-cursor segment with a **2D zero-mean
"Mexican hat"** profile `(1 - d²/2σ²)e^{-d²/2σ²}`, `σ = √2/k` (peaks at the
layer's wavenumber `k = 2πf/c`), so it makes ripples, never swells. Pulsed at
`WAVE_FREQ=[4, 0.9]`Hz for slow strokes (rings), blending to a steady push
for fast strokes (`uCurSteady`, speed > c) which leaves a clean V wake (and is
softened ×0.55 since steady pushes build up more). Strength
`(speed/500)^0.6 × CURSOR_PUSH (7)`, capped.

**Mosaic** (`facetUpdate` / `facetSpread` / shade): the water is a jittered
grid Voronoi of `FACET=9px` cells, fixed to the page and invisible while calm.
Per mosaic cell (compute, ~16k cells): centre (+ sway down the slope, capped
at 0.25·FACET), and crest/trough strength from the **combined** surface:
`h = h_fast + h_slow`, quadrature `q = slope_fast/k_fast + slope_slow/k_slow`,
`cosPhase = h/√(h²+|q|²)`, lit when `cosPhase > 0.62..0.94` and amplitude
`> 0.12..0.45`. Combining both layers makes cursor ripples and duck wakes
interfere. Cells within ~12–36px of the moving water cursor are hidden (the
pulsing push churns there; rings appear to come from under the cursor). The
pixel shader finds the nearest centre in the 3×3 records and draws a rounded
blob (cell ∩ disc growing with strength): ripple colours on crests, a faint
`waterDeep` shadow in troughs.

## Performance notes (target: 120fps on a Retina MacBook)

- Measure with a real (headful) Chrome window; headless caps at 60 (and then
  runs 2 wave substeps). `?gputime` gives GPU ms (noisy: average many).
- Budget is tight (~8.3ms). GPU timestamps on this Mac are unreliable (clock
  scaling: turning passes off sometimes made frames *slower*); A/B with real
  fps instead, interleaving runs against a known-120fps backup, both calm
  (no mouse) and with constant mouse movement.
- Fragment shader size matters for every pixel (register pressure / occupancy):
  rarely-taken but big branches (pad + flower code) cost ~1.5ms for all
  pixels; move such work into its own pass.
- Expensive things that were removed: device-res grid, per-pixel Perlin,
  per-pixel neighbour search, candidate arrays, foam distance field,
  per-pixel wave sampling + hashing for the mosaic (moved to `facetUpdate`),
  per-cell Perlin for pad damping (moved to `wavePadPass`).
- TSL inlines a node expression at **every use**: `.toVar()` anything used
  more than once that reads buffers (e.g. the wave Laplacian) — missing this
  cost ~15fps.
- Keep the fragment shader's register use low; prefer per-node / per-facet
  work in compute passes over per-pixel work.
- Per-stage storage-buffer limit is 8 (nodeUpdate is at it).
- If the owner has the site open in their own Chrome, it shares the GPU and
  headful benchmarks read low (~90fps) for every version: compare against a
  backup in the same session before concluding something got slower.

## User preferences learned

Liked: blobby rounded ripple cells, V wakes (esp. duck wakes and fast cursor
strokes), clustered pads, Voronoi-cell pads (some bigger, a little Voronoi
irregularity is fine; veins, lit rim), flowers, ducks parting pads, ducks drawn in the pad style, letters with the same lit
rim (cohesive), chunky
simple ducks
(text-coloured body, accent bill), crisp (not blurry) edges, ripples that
dissipate fairly soon.
Disliked: pale foam/edge-lightening on water, thin ring-crest ripples, gaps
between ripple emissions, perfectly circular hand-placed rings, static/
speckle in the water, erratic slow-stroke ripples, ripples too strong for
slow strokes, duck ripples faster than the duck, jagged/stair-stepped lily
pads, overlapping non-Voronoi pads, pads disappearing near ducks, complete
pads flying in during the intro,
ducks jumping when scrolling, glitchy/specky edges, blurry particles.
