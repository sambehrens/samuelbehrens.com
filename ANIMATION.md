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
- `pond/index.html` (served at `/pond`) — the same animation with no words
  or links, so the pond fills the window, plus a small, faint fps meter in the
  bottom right (`.dev-menu.dev-mini`, a button: pale cream text on a light
  frosted tint, no border, so it blends into the water; clicking it removes it until
  reload). It loads `../css/home.css` and `../js/voronoi.js` unchanged; with
  no `[data-fx]` elements `layout()` only places water and pads, and
  `setupDevMenu()` only starts the fps meter (`startFpsMeter`) for a
  `.dev-mini` menu. Same URL params work there.
- `css/home.css` — page styles. When the effect runs, `html.fx-on` makes DOM
  text transparent (it stays for layout, links, selection, a11y) and hides the
  CSS underline bars. `html.fx-debug` shows DOM text in red for alignment.
  Without WebGPU (or `?nofx`) the plain styled page shows.
- `js/voronoi.js` — everything else. `resume.html` / `css/style.css` /
  `css/index.css` are the untouched old pages.

## URL params / dev menu

`?nofx` disables the effect, `?debug` overlays DOM text, `?ducktest` spawns a
duck (always with a brood of ducklings) immediately at 70%/45% of the viewport heading right
(`?ducktest=x,y,heading` to aim it, fractions of the view / radians; it
also puts the duck's viewport position in `window.__duck` and every duck
slot's position / leader in `window.__ducks`; e.g.
`?seed=0&ducktest=0.5,0.38,0` sends it through a pad colony), `?raccoontest`
sends a raccoon in at 25%/50% heading right after 1.5s
(`?raccoontest=x,y,heading,meals`, meals already eaten, for its belly;
use a heading like 0.001, since 0 is dropped as empty;
`?ducktest&raccoontest=0.42,0.5,0` starts it
just behind the test duck's brood, which makes a hunt within seconds; each
`__ducks` entry then also has coon, mode, size, eaten, meals), `?flytest`
sends a dragonfly in after 0.5s that keeps looking for pads (each perch's
state in `window.__flies`, incl. why it last took off; `window.__splash(x,
y, strength)` makes a splash at a viewport point), `?frogtest` brings a
frog up after 0.5s that keeps coming back (`?frogtest=N`: N pads per visit;
state in `window.__frogs`, incl. why it last fled; `__splash` too; with
`?frogtest=9&raccoontest=0.3,0.5,0` a raccoon hunts the frogs, and
`__ducks` entries show what each raccoon's `prey` is), `?gputime` turns
on GPU timestamp queries and puts `{c, r}` (compute / render ms, every 30
frames) in `window.__gpu` (splits compute into three submissions), `?seed=N`
repeats one pond layout (see Lily pads; random per visit otherwise). The
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
  2..2.9 glyph (+hover darkening), 3 orange bar, 4.x duck body, 5.x duck bill, 6.x duckling body, 7.x raccoon fur, 8.x its mask patches / tail rings / nose, 9.x its muzzle / ears (fraction = wave lift, like pads)),
  y spacing (**pad radius** for pads), z rnd (glyph / bar cells:
  `floor(rnd·32) + wave lift`, since their x fraction is taken), w tone
  (glyph) / tint
- `groupBuf` per DOM element: hover, darkens-on-hover, tones, bar ranges
- `fieldBuf` mouse/duck wake field, 8px cells: x energy, yz flow,
  **w water height** (copied from the wave sim; nodes read it from here
  because `nodeUpdate` is at the 8-storage-buffer limit, bilinearly via
  `fieldHeight()` so small letter cells get smooth slopes, not 8px steps)

Water buffers: `waveA`/`waveB` (ping-pong, per cell: fast h, fast h_prev,
slow h, slow h_prev), `waveView` (vec2: both heights, written by every step),
`wavePads` (extra damping in lily pad colonies, recomputed only when the grid
re-anchors), `facetBuf` (per mosaic cell: centre, crest, trough), `facetMax`
(3×3 max of crest/trough, so calm pixels skip after one read).

Uniform arrays (one entry per **duck slot**: `MAX_DUCKS` big ducks, then
`MAX_RACCOONS` raccoons (from `FIRST_COON`), then `MAX_DUCKLINGS` ducklings
(from `FIRST_CHICK`)): `uDucks` (viewport pos + heading), `uDuckVel` (vel +
body bend in z + **size** in w: 1 normal, a swallowed duckling shrinks to 0,
and it scales the slot's cell radii, cell offsets, shove ellipse, paddling
and wake. For a **raccoon** it's instead how full its belly is (≥1): only
its belly cell (`PART_FUR` with no head/tail mark) grows by it, its head
cells move forward and tail cells back by `coonShift` = 1.1 × the belly's
growth (keeps the power bisectors to head and tail where they were, so they
keep their size), and its shove ellipse gets longer by that and wider to
the belly); `uDuckList` + `uDuckCount` (the slots of
the ducks in the water, packed: x slot, y kind 0 duck / 1 duckling / 2
raccoon); `uDrops` (click splashes: document pos, start time, strength).

## Per-frame pipeline (`renderer.setAnimationLoop`)

CPU: dt, scroll, mouse → wake uniforms; smoothed **water cursor**; hover
easing → `groupBuf`; `updateDucks()`; `updateFlies()`; `updateFrogs()`;
wave grid anchor /
substeps. Then one
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
9. `padProbe` — one thread per dragonfly and frog (`PROBES`; dragonflies
   first): the lily pad under a point it asks about (`uProbeAsk.xy`, from
   `padBlock`; or `-3` and the point if no leaf covers it at all, i.e. open
   water or text) and where a watched pad (`uProbeAsk.z`) is and how fast it
   moves, into `probeOut`, read back to the CPU (`readProbes`; see
   Dragonflies, Frogs).

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
and duck cells (`isLeafCode`: surface 1, 4, 5, 6): an instanced quad per node
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
- **Every visit gets a different pond**: `POND_SEED` (random per page load,
  or `?seed=N` to repeat one) picks the colony noise's slice
  (`PAD_NOISE_Z = 3.7 + seed·1.618`, shared by `padField`, `patchPad` and
  `wavePadPass`), salts the merged-patch hash and seeds `layout()`'s RNG
  (`mulberry32(1337 + seed)`). It stays fixed for the visit, so resizes keep
  the same pond. `?seed=0` is the pond every visit used to get (owner asked
  why the pads were always arranged the same way).
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
**Letters ride the waves**: every text node (ink, halo and plate, so whole
letters move together) is drawn offset down the water's slope
(`TEXT_SWAY=18` px per unit slope, capped at `TEXT_SWAY_MAX=2`px; no push
force, so they settle back as the water calms), and glyph / bar cells get
the wave light: lighter on crests (toward white, ≤0.3), and in troughs mixed
toward the water colour (≤0.28) as if sinking a little. A first try (sway 40,
cap 4px, troughs darkened toward black) tore letters apart under a click
splash and turned cream letters muddy grey.
Tried and rejected: cracking letters into chunky pieces (a coarse procedural
Voronoi with gaps, or grouping cells per chunk): read as broken plaster /
scattered blobs, not as the pads' clean cells.

## Colours

Palettes in `PALETTES.light/dark` (water, pads, petals, ripple, text=cream,
accent=#ffb82b used for underlines, duck bill, flower centres; duckling
yellow; raccoon grey `raccoon`, `raccoonDark` and `raccoonPale`; dragonfly
orange-red `fly`, clear `flyWing`; frog lime `frog` and darker `frogLeg`,
lifted a little in dark mode so the legs show against dark pads). Colours are
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

## Ducks (`MAX_DUCKS=12`)

CPU steers each duck (enter from an edge of the current view, weave via two
sines, paddle-and-glide speed, exit), in **document coords** so
scrolling doesn't move them.

**How many.** One shared spawner (`nextDuck`) sends a duck in from a free
slot every `cross / DUCKS_IDLE × 0.7–1.3` s, where `cross` is a rough crossing
time (view's mean side + margins at 0.8×`DUCK_SPEED`). With `DUCKS_IDLE=2.5`
a quiet pond has ~2.7 ducks on average on any screen size (simulated:
2–4 about 95% of the time, 5 about 1–2%; the first duck comes at 1.5–4.5s,
the second 4–9s later). The owner wants it to feel unlimited with
interaction: ducks that someone's waves keep in the pond stay longer while
more keep arriving, up to 12. (Fixed spacing like 10–18s gave ~1.6 ducks on
a phone and ~4.4 on a 1080p screen, hence the scaling.) Ducks keep clear of
each other: one near ahead (within 2.5× the touching distance) makes a duck
steer its base heading aside, harder the nearer, and touching ducks get
nudged apart. Ducks swim even under
`prefers-reduced-motion` (they're slow and gentle; the owner's phone has iOS
Reduce Motion on and had no ducks when they were skipped). Reduced motion
still skips the fly-in intro and calms the ambient drift.

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

### Ducklings

About half the ducks (`BROOD_CHANCE=0.5`) bring a brood of 3–4 ducklings.
They use the last duck slots, after the big ducks' and raccoons' (`MAX_DUCKLINGS=64` shared slots,
`isChickSlot(i)`), so every GPU path (cells, shove ellipses, paddling, wake,
wave readback) handles them like ducks with per-slot sizes (`slotCells`,
`slotScale`, `slotExtent`). The many slots are so one duck can gather a long
line (the owner wants it to feel unlimited: people herd ducklings from duck
to duck with waves). The shader loops over ducks (paddling, wake, shove
ellipses) run only over the ducks in the water (`forEachDuck`: a dynamic
loop over `uDuckList`, sizes picked per entry), so empty slots cost nothing.
Unrolled loops over every slot cost ~0.3ms of compute at 76 slots even when
empty. A duckling is 3 cells at `CHICK=0.46` scale: a round body, a
big head and a little bill (`chickCells`), in the palette's `duckling`
yellow (surface code 6) with the accent bill. Splitting it into the duck's 7
cells read as a scatter of petals at this size; cream tinted toward the
accent read as tan, not yellow. Heads bob quicker, feet paddle at 1.6× the
duck's rate into the slow layer (`DUCKLING_PADDLE`), so each leaves a tiny
V wake.

CPU (`updateChick`): each duckling has a `leader` (a duck slot). It follows
a rope point `gapBehind(leader, i)` from the leader (on the line toward it,
pulled round behind the leader's heading), plus the leader's velocity, with
inertia and paddling spurts; heading follows its swimming velocity. Waves
push ducklings harder than ducks (`DUCKLING_WAVE_PUSH=1.5`× gain, more drag).
A duckling more than `DUCKLING_LOST=70`px from its spot, or drifting faster
than `DUCKLING_SCATTER=45`px/s, **loses its line**: the one behind it moves
up (`loseLine`). After a 0.35s daze it hurries (`DUCKLING_RUSH=4.2`×
`DUCK_SPEED`) to the **nearest duck that's in a line** (`motherOf` ≥ 0) and
joins the end of that line (`tailOf`) once within `DUCKLING_JOIN=24`px, so
it can end up with another mother. Ducklings keep out of other ducks' way
(a simple CPU repulsion). A mother leaves only once she and her whole brood
are out of view; ducklings with no line left (`motherOf` < 0) swim on and
disappear once out of view. Tested: a click splash beside a brood scatters
all of it and it regroups in ~3–4s; a quick stroke across the tail knocks
out just the last duckling or two.

### Duck cells

Each cell springs to its spot in the duck's frame (head and bill bob; whole
body bends by rotating each cell by `x × bend`, bend from turn rate) with the
duck's velocity fed forward, so it holds shape but can be knocked apart.
Ducks shove other nodes out of a duck-shaped ellipse (pads out of one grown
by their radius, so they're parted, not swallowed), sway a little with
passing waves, and paddle: their feet (behind the body) pulse into the
**slow** wave layer at `WAVE_FREQ[1]`, which leaves a V wake of arcs (owner
likes this a lot). A moving body dent (the old `DUCK_PRESS`) was tried and
removed: it showed as a dark dent around the duck.

Each shove ellipse is centred on the box around the slot's cells along its
length (`slotExtent().mid` / `half`), not on the slot's centre, so a
raccoon's long tail doesn't make it push things far ahead of its nose.
Ducklings never shove a raccoon's cells (ones ahead of a lunge tore its face
apart), and a duckling being swallowed neither shoves nor is shoved.
**Sizes never reach 0 in the shaders** (clamped; a slot nearly gone is also
dropped from `uDuckList`): a 0 size divided by 0 in the shove ellipse, the
NaN times the 0 weight was still NaN, and every node's position went NaN for
good (words, ducks and pads vanished; only water came back, since water
nodes respawn).

## Raccoons (`MAX_RACCOONS=2`)

Owner asked for raccoons as cute as the ducks that hunt ducklings and
"absorb" them, which ducks avoid, made to feel natural, fun and satisfying.
They hunt **frogs** sitting on pads too (`preyAround()`: ducklings, and
frogs that are `frogExposed`; `r.prey` holds the animal itself), and come
while either is in view; a caught frog is swallowed like a duckling.

**Look** (`coonCells`, `RACCOON_SCALE=1.8`, ~120 css px long, bigger than a
duck): a chubby round back; a big head; and a bushy tail of five
alternating dark / fur rings (dark tip) floating out behind. The face is
built from the front: two plain dark mask patches, then a short pale
muzzle and a button nose; behind them a smaller grey crown with a big round
pale ear at each back corner. Tail cells carry +20 in attr.w and **wag** in
a slow wave that runs down the tail, swinging more toward the tip; head
cells (+10) sniff side to side. Designed in a 2D power-diagram prototype
first (same cell / rounding rules, rendered per pixel on a 2D canvas):
thin, evenly spaced tail rings and a small head read as a caterpillar; tail
rings spaced closer than their radius read as one striped tail. Owner asked
for a cuter face than the first one (mask cells poking out past the crown
like horns, small ears, a long snout); patches placed inside a big crown
cell lose the power diagram and shrink to slivers, hence the front-built
face. Pupils with white glints drawn in the patches were then rejected:
they didn't match the simple blob style (keep every animal to flat blobs).

**Behaviour** (`updateCoon`, CPU; real raccoons swim well but slowly, head
up, tail floating, and do take ducklings): it comes in from an edge like a
duck, slower (`RACCOON_SPEED=0.75`× `DUCK_SPEED`), weaving and sniffing.
Modes:
- `cruise`: after its `rest`, spots the nearest duckling in view within
  `RACCOON_NOTICE=230`px of its mouth (`COON_MOUTH`) → `stalk`.
- `stalk`: swims at where the duckling is going at `RACCOON_STALK=1.6`×
  (gains on a duck, not on a fleeing duckling); gives up after 15s or if it
  gets away (> 1.5× notice). Within `RACCOON_POUNCE=80`px → `crouch`.
- `crouch`: stops short for 0.28s, fixed on it (the wind-up).
- `lunge`: 0.5s burst at `RACCOON_LUNGE=6`×, easing off, homing a little.
  Any duckling within `RACCOON_CATCH=19`px of its mouth is caught →
  `munch`; else → `recover` (0.9s coast), then stalks again.
- `munch`: stops and wriggles (body bend `wiggle`) for 1.4s while the
  duckling (`eaten` timer, `by`) is drawn into its mouth and shrinks to
  nothing in 0.45s, with a small splash (`splash(…, 0.15)`); its **belly**
  fills out (only the belly; head and tail stay the same size). Then a 5–9s
  rest. There's no "full". It **stays** `RACCOON_STAY` 150–240s (owner
  wanted them to stay longer): until then, cruising within 70px of the
  view's edge and heading out, it turns back toward somewhere in the middle
  of the view (tested: in view 99% of two minutes); then it heads for the
  nearest edge (`leave`).
**Belly and weight** (owner asked that raccoons get bigger the more they
eat, only the stomach blob, and slower the bigger, never so slow it can't
hunt, with no limit but growing less and less): belly =
`1 + RACCOON_GROW·ln(1 + meals)` (`RACCOON_GROW=0.32`: 1.22 after one meal,
1.44 after three, 1.77 after ten); its speeds (cruise, stalk, lunge) are
divided by `belly^RACCOON_HEAVY` (0.6: 89% / 80% / 71%), turning by the
square root of that. Tested (`?ducktest&raccoontest=0.42,0.5,0.001,M`):
empty, a catch in 8–12s after 2 lunges; 3 meals, 17–23s after 4–7 lunges;
10 meals, 2 of 3 runs caught one within 30s. (A first try slowing it by
`belly^1.2` with linear growth, full after 4, was too slow too soon.)
A hard wave push (drift > 28px/s) spooks it off the hunt for 4s and turns it
with the water, so people can save ducklings. Waves push it less than a
duck (0.6× gain). Two raccoons keep apart.

**Prey and ducks.** Ducklings flee a raccoon within `DUCKLING_FEAR=115`px (+
a bit for its size; only 0.55× that while it's stalking or crouching, so it
can creep up, full range once it lunges): a flee velocity scaled by `alarm`
(rises in ~0.2s, a moment's reaction), up to 2.7× `DUCK_SPEED`, which can
pull them off their line. They're also kept off its back and tail (CPU
repulsion from its spine), not its mouth. Big ducks within `DUCK_WARY=170`px
turn away (not just aside) and hurry (`startle`), leading their broods off.
`motherOf` never counts a raccoon (or a swallowed duckling) as a line.

**Spawning:** first after 25–45s, then every `RACCOON_EVERY` 50–100s, but
only while a duckling or a frog on a pad is in view (else it checks every 3s). Tested: with
`?ducktest&raccoontest=0.42,0.5,0` every run ended in a catch after 1–3
lunges with near misses (stalk 1.05× never got close; without the sneaky
stalk every lunge missed); naturally, the first raccoon arrived at 34s,
caught one duckling and left at ~95s. 120fps headful, ~+0.1ms compute.

## Dragonflies (`MAX_FLIES=3`)

Owner asked for dragonflies that fly in, land on lily pads, wait a random
time and fly off, with a small ripple when they land and take off, flying
off when their pad is disturbed; then for a very gentle ripple, a more
natural path than straight lines, and a simpler design.

**Look** (`flyMesh`, drawn in the lily pad pass's scene after the pads,
over everything): as simple and cute as the ducks. The body is one chubby
rounded pill in flat colour with the cells' lit rim (top left) and thin
darker edge: an exact uneven capsule, head end r=2.9 at x=5.5 tapering
slightly to r=2.2 14 units behind. It's widest where the wings meet it.
Four round clear wings spread out **sideways** (fore pair angled a little
forward, L 7.2 × W 3.4 from x=1.6; hind a little back, 6.8 × 3.6 from
x=−0.6) give the classic dragonfly cross silhouette (SDFs in the fragment
shader, one quad per slot, in units of `FLY_SIZE=1.3`px). Owner asked for
one body shape, not a separate head and body, and said a round head
blended into a tail "looks like a matchstick": a fat end on a thin tail
reads as a knob on a stick however it's blended (teardrops, eggs and
cones were all tried), so the fix was an even body plus sideways wings. Flying, the wings beat (foreshorten and fade, fore and
hind out of step, 26Hz); perched, they lie flat. A soft shadow down-right,
further and softer the higher it flies (`FLY_ALT=24`px); a little bigger
when high. Perched, the vertex stage places it at its pad's live `siteBuf`
position plus its spot, so it rides the pad's sway exactly. (Too detailed,
and rejected: a first take with segment bands, wing veins, tip spots and
eye glints; then two eye beads, a thorax and a long thin abdomen; then a
darker round head on a separate teardrop body with petal wings in an X;
then that head and body blended into one teardrop (the matchstick).
Designed
in a 2D SDF prototype. Its own render pass cost ~2ms of GPU at Retina size:
hence drawn in the pad pass.)

**Flight** (`updateFly`, CPU, document px; real dragonflies fly direct but
not ruler-straight): darts along a **bowed path** (`startDart`: a quadratic
curve bent 10–30% of its length to a random side, eased in and out over
`distance / 300–420px/s`, with a slight sideways flutter mid-dart) that it
follows on a stiff spring (`flyAlong`), facing its motion; then a dead stop
and a hover (0.3–1.3s, bobbing, glancing about). Modes: `dart`, `hover`,
`approach`, `land`, `perched`, `takeoff`, `leave`.

**Pads.** The CPU can't see pads, so while hovering a dragonfly asks
`padProbe` about a random point within ~240px each frame (`uProbeAsk.xy`);
answers (a frame or two late) give a pad's id, centre and radius (pads with
a flower, or not fully grown, are skipped; one already taken by another
dragonfly or a frog too, `padFree`). It darts over the pad (tracking it via `uProbeAsk.z`, the
watched pad), settles onto a spot within 0.3 of its radius (0.45s: shadow
draws in, wings slow) with a tiny-drop ripple round the rim, and sits
`FLY_PERCH` 4–14s. **Disturbed:** after 1.2s (its own ripple settles), it
measures the watched pad: moving faster than `FLY_SPOOK=16`px/s or drifting
`FLY_SHIFT=5`px from where it settled (that rest point creeps with the pads'
slow wander), or no longer a pad, sends it off with another ripple, and
it **bolts** (owner asked for a quicker escape when a ripple disturbs it):
a 0.1s lift instead of 0.3s, then a dart of 180–320px at 650–800px/s that
starts at full speed and only slows toward the end (ease-out, stiffer
spring, top speed 1000px/s), ~200px from the pad 0.4s after leaving; a
calm take-off eases into a 300–420px/s dart. A relayout (`layoutGen`) changes node ids, so it lets go.
Tested: a click 120px away sends it off ~0.55s later (as the ring reaches
it); one 300px away doesn't; left alone it sits its full time. After 1–3
pads (or 75s) it sweeps off out of view. Spawning: first after 5–10s, then
every `FLY_EVERY` 14–32s while a slot is free.

## Frogs (`MAX_FROGS=3`)

Owner asked for frogs that jump out of the water onto lily pads, jump
between pads and back into the water, jump in when disturbed like the
dragonflies, get eaten by raccoons, and make ripples in proportion to each
jump. Then: no eyes (a first take had two eye bumps, and floated with just
its eyes above water before leaping out; both removed), and a frog should
only flee a raccoon hunting it once it **notices** it, so the raccoon has a
chance.

**Look** (`frogMesh`, drawn in the pad pass's scene like the dragonflies,
before them; SDFs in units of `FROG_SIZE=1.5`px, ~21px long sitting): a
plump egg of a body (blunt nose a little longer than the rump, so it reads
as facing somewhere without eyes), two small front legs, and back legs
(thigh ellipse and foot disc smooth-unioned into one blob) folded at its
sides; in a leap (`legs` 0→1) they stretch straight out behind and the
front legs reach forward. Flat colours with the cells' lit rim and thin
darker edge, a soft shadow down-right that moves out with height, a little
bigger when high. Coming out of or going into the water (`sub`) the legs
vanish and the body shows faint and water-tinted. Sitting, the vertex stage
draws it at its pad's live `siteBuf` position plus its spot. Designed in a
2D SDF prototype.

**Behaviour** (`updateFrog`, CPU, document px). Modes: `under` (invisible),
`leap`, `sit`, `seek`, `aim`, `dive`.
- `under`: after 2–5s (none for a new frog) it looks for somewhere to come
  out: a free pad under a random point in view, then a spot 14–40px beyond
  its rim, checked to be open water (`padProbe` answers `-3` for it, matched
  by position) and not near the words (`textBoxes`, from `layout()`). Then
  it leaps straight out onto the pad. After 1–2 visits (or 100s) it's gone.
- `leap`: a straight hop over 0.3–0.6s (`distance / 280px/s`), height
  `4·H·u(1−u)` with `H = min(0.22·d, 30) + 8`, landing wherever the target
  pad has drifted to (watched); legs kick out at once and fold as it comes
  down on a pad.
- `sit`: rides the pad `FROG_SIT` 3–9s, glancing about, then leaps to a free
  pad `FROG_REACH` 35–150px away (`padSearch`, after an `aim` turn of
  ~0.25s); after `FROG_HOPS` 1–4 pads, `seek`.
- `seek`: finds open water 18–50px beyond its pad's rim (`waterSearch`),
  aims and leaps in; `dive`: plop, then it fades under in 0.2s.
**Ripples** (tiny drops, `FROG_RIPPLE`, sized between a dragonfly's 0.1 and
a click's): a ring of radius 7 where it leaves the water (0.5), round the
pad's rim when it lands (0.3) or kicks off (0.25), radius 8 where it plops
in (0.9).
**Disturbed** like a dragonfly (`ride`: its pad faster than `FROG_SPOOK=18`
px/s or `FROG_SHIFT=6`px from where it settled, after its own landing ring
settles in 1.2s, or no longer a pad) it's startled (`panic`): it checks the
water for at most 0.1s and leaps off at once, no turning first, away from
the danger.
**Raccoons**: a frog only flees one hunting it (`r.prey` is the frog), and
only once it notices: a chance per second `FROG_NOTICE` (stalking 1 × how
near its mouth is, 0–1 over `RACCOON_NOTICE`; crouched 1; lunging 2.5),
then a reaction of `FROG_REACT` 0.12–0.3s. It can be caught on its pad
until it's in the air. Tested with `?frogtest=9&raccoontest`: 5 of 15
hunts on frogs ended in a catch, mostly frogs that noticed too late, mid-
lunge (with stalk 0.3 from the raccoon's centre almost none noticed in
time; before the panic leap was made
instant, the 0.35s spent checking water and turning meant every hunt
did). Frogs and dragonflies never share a pad (`padFree`). 120fps headful
with frogs, dragonflies, ducks and a raccoon.

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

**Click splashes**: `pointerdown` (left button or a tap, not on the dev
menu) writes a splash into `uDrops` (`MAX_DROPS=4` slots, oldest reused).
In `waveStep` each splash pushes with the same zero-mean hat profile
(`DROP_SIGMA=16px`): the fast layer with a swing at `DROP_FREQ=3`Hz that
dies away over `DROP_TIME=0.8`s (fade²), so a big lead ring is followed by a
couple of smaller ones; the slow layer with one soft push (first quarter of
that), which leaves lazy rings lingering where it landed. `DROP_PUSH=[110,
8]`. The rings cross a 1280px view in ~3s, are soaked up by pad colonies,
and push ducks (wave flux) like any other wave. A single half-sine push gave
one thin, faint ring; 60 read too weak for a "large" wave. No fps cost.

**Tiny drops** (`TINY_DROP`; strength < 0, packed as `-(radius + amount)`,
radius in whole px, amount in [0, 1)): one quick soft push (0.2s,
`sigma=6`) into the **slow layer only**, along a circle of that radius, so
a gentle ring rises and fades nearby. Used by dragonflies around their
pad's rim (`FLY_SPLASH=0.1`). A scaled-down click splash (0.1) was far too
big (fast rings across the page); a centred tiny push looked delayed (the
slow ring took ~0.5s to surface from under the pad); pushing along the rim
shows at once (it carries ~4× the energy of a centred one, hence the small
amount).

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
- Ducks: with the packed duck loops, the 76-slot pool times the same as the
  old 10-slot one, and a pond packed full (12 ducks, ~48 ducklings, up to
  ~47 on screen) still holds 120fps at ~+0.2ms compute.
- Every extra `renderer.render()` is a full-screen render pass: the
  dragonflies' own pass cost ~2ms GPU at Retina size, inside the pad pass
  they cost ~nothing. Add new overlays to the pad pass's scene (the frogs
  are there too).
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
- **Buffer size limit**: the slot grid (`SLOTS` ints per css px) is the
  biggest buffer, ~139MB on a 4K screen at 1x, over WebGPU's default 128MB.
  `main()` requests the adapter's own `maxStorageBufferBindingSize` /
  `maxBufferSize`; `resize()` sizes the pixel stage for the whole screen (+5%)
  capped at what the device allows, and turns the effect off (plain page)
  if the view itself won't fit. The jump-flood, neighbour-list and pad-block
  buffers are per 2×2 block (`blockCap`, a quarter of the pixels).
- **Memory**: three.js keeps a CPU copy of every storage buffer and uploads
  it mapped-at-creation, and Chrome keeps that mapping's shared memory for
  the buffer's whole life, in the page *and* the GPU process (measured with a
  bare 200MB buffer: empty +192MB GPU / +0 page; mapped-at-creation +383MB
  GPU / +191MB page). With ~92MB of buffers that made the tab ~800MB
  (`footprint -p <pid>` on the GPU and renderer processes). So buffers only
  the GPU writes are made with `gpuArray()`: three gets a 1-item array named
  `gpu-only:<bytes>`, and a hook on `device.createBuffer` makes the real,
  empty (zero-filled) buffer at full size (WGSL storage arrays are
  runtime-sized, so shaders see all of it). Only CPU-written buffers (pos,
  home, attr, group, duckWave) use `instancedArray`. The renderer is also
  made with `depth: false` (nothing uses depth; a screen-size depth24
  texture). Result on a Retina MacBook: page 297→133MB, GPU process
  501→~400MB. What's left in the GPU process is mostly the canvas's
  swap-chain surfaces (~75MB), the buffers themselves (slot grid 32MB,
  padData 10.5MB, ...) and pipelines.
- If the owner has the site open in their own Chrome, it shares the GPU and
  headful benchmarks read low (~90fps) for every version: compare against a
  backup in the same session before concluding something got slower.

## User preferences learned

Asked for: a large ripple/wave when clicking (see Click splashes); waves
and ripples affecting the letters (see Letter styling); ducklings in a line
behind about half the ducks, separable, swimming back to the nearest duck
(see Ducklings); more ducks / ducklings that feel unlimited with
interaction (see Ducks); cute raccoons that hunt and swallow ducklings and
that ducks avoid, their bellies filling out (slower, never stopped) the
more they eat (see Raccoons); dragonflies that land on pads with very
gentle ripples, curved natural flight, simple style (see Dragonflies);
frogs that leap between the water and pads with proportional ripples, flee
when disturbed, can be eaten, no eyes (see Frogs).
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
ducks jumping when scrolling, glitchy/specky edges, blurry particles,
fine detail on the animals (pupils with glints, wing veins, segment bands):
keep them simple flat blobs.
