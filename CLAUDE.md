# samuelbehrens.com

Static personal site on GitHub Pages (no build step). The homepage is a
WebGPU / three.js (TSL) Voronoi pond animation in `js/voronoi.js`.

**Before working on the homepage animation, read `ANIMATION.md`.** It explains
the whole pipeline (node kinds, buffers, compute passes, shading, ducks,
ripples, lily pads), the performance constraints (keep 120fps on a Retina
MacBook; measure in a headful Chrome window), and the owner's style
preferences.

**Keep `ANIMATION.md` up to date.** Whenever you change how the animation
works (new features, buffers, passes, constants, tuning values, or things
the owner liked/disliked), update `ANIMATION.md` in the same change so it
always matches `js/voronoi.js`.

**Don't be afraid to start over.** The goal is a pleasing, polished animation,
not preserving the current code. If tuning an approach isn't getting there,
rethink how that part works from the ground up (new simulation, new
rendering, new data flow) whenever needed — then document the new design in
`ANIMATION.md`.

Local preview: `python3 -m http.server 8765`, then open
http://localhost:8765/ (`?debug`, `?nofx`, `?ducktest` are useful).
