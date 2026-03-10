/**
 * cpu.mjs — Prism vs K₃,₃ with DRESS (WASM/Node.js)
 *
 * Run:
 *   node cpu.mjs
 */
import { dressFit } from 'dress-graph';

// Prism (C₃ □ K₂): 6 vertices, 18 directed edges (0-based)
const prism_s = [0,1,1,2,2,0,0,3,1,4,2,5,3,4,4,5,5,3];
const prism_t = [1,0,2,1,0,2,3,0,4,1,5,2,4,3,5,4,3,5];

// K₃,₃: bipartite {0,1,2} ↔ {3,4,5} — 18 directed edges
const k33_s = [0,3,0,4,0,5,1,3,1,4,1,5,2,3,2,4,2,5];
const k33_t = [3,0,4,0,5,0,3,1,4,1,5,1,3,2,4,2,5,2];

const rp = await dressFit({ numVertices: 6, sources: prism_s, targets: prism_t });
const rk = await dressFit({ numVertices: 6, sources: k33_s,   targets: k33_t });

const fp = Array.from(rp.edgeDress).sort((a, b) => a - b);
const fk = Array.from(rk.edgeDress).sort((a, b) => a - b);

console.log('Prism:', fp.map(v => v.toFixed(6)));
console.log('K3,3: ', fk.map(v => v.toFixed(6)));

const same = fp.length === fk.length && fp.every((v, i) => v === fk[i]);
console.log('Distinguished:', !same);
