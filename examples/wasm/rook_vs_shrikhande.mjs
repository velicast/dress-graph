/**
 * rook_vs_shrikhande.mjs — Rook vs Shrikhande with Δ¹-DRESS (WASM/Node.js)
 *
 * Both are SRG(16,6,2,2) — indistinguishable by Δ⁰-DRESS.
 * Δ¹-DRESS (k=1) separates them via histograms and multisets.
 *
 * Run:
 *   node rook_vs_shrikhande.mjs
 */
import { deltaDressFit } from '../../wasm/dress.js';

// Rook L₂(4) = K₄ □ K₄ — 16 vertices, 96 directed edges
const rook_s = [0,1,0,4,0,2,0,8,0,3,0,12,1,5,1,2,1,9,1,3,1,13,2,6,2,10,2,3,2,14,3,7,3,11,3,15,4,5,4,6,4,8,4,7,4,12,5,6,5,9,5,7,5,13,6,10,6,7,6,14,7,11,7,15,8,9,8,10,8,11,8,12,9,10,9,11,9,13,10,11,10,14,11,15,12,13,12,14,12,15,13,14,13,15,14,15];
const rook_t = [1,0,4,0,2,0,8,0,3,0,12,0,5,1,2,1,9,1,3,1,13,1,6,2,10,2,3,2,14,2,7,3,11,3,15,3,5,4,6,4,8,4,7,4,12,4,6,5,9,5,7,5,13,5,10,6,7,6,14,6,11,7,15,7,9,8,10,8,11,8,12,8,10,9,11,9,13,9,11,10,14,10,15,11,13,12,14,12,15,12,14,13,15,13,15,14];

// Shrikhande — 16 vertices, 96 directed edges
const shri_s = [0,4,0,12,0,1,0,3,0,5,0,15,1,5,1,13,1,2,1,6,1,12,2,6,2,14,2,3,2,7,2,13,3,7,3,15,3,4,3,14,4,8,4,5,4,7,4,9,5,9,5,6,5,10,6,10,6,7,6,11,7,11,7,8,8,12,8,9,8,11,8,13,9,13,9,10,9,14,10,14,10,11,10,15,11,15,11,12,12,13,12,15,13,14,14,15];
const shri_t = [4,0,12,0,1,0,3,0,5,0,15,0,5,1,13,1,2,1,6,1,12,1,6,2,14,2,3,2,7,2,13,2,7,3,15,3,4,3,14,3,8,4,5,4,7,4,9,4,9,5,6,5,10,5,10,6,7,6,11,6,11,7,8,7,12,8,9,8,11,8,13,8,13,9,10,9,14,9,14,10,11,10,15,10,15,11,12,11,13,12,15,12,14,13,15,14];

const dr = await deltaDressFit({ numVertices: 16, sources: rook_s, targets: rook_t, k: 1, keepMultisets: true });
const ds = await deltaDressFit({ numVertices: 16, sources: shri_s, targets: shri_t, k: 1, keepMultisets: true });

console.log(`Rook:       ${dr.histSize} bins, ${dr.numSubgraphs} subgraphs`);
console.log(`Shrikhande: ${ds.histSize} bins, ${ds.numSubgraphs} subgraphs`);

// Compare histograms
const histSame = dr.histSize === ds.histSize &&
    dr.histogram.every((v, i) => v === ds.histogram[i]);
console.log(`Histograms differ:  ${!histSame}`);

// Canonicalize multisets: reshape to [numSubgraphs x E], sort each row, then sort rows
const E = rook_s.length;
function canonicalize(ms, nSub) {
    const rows = [];
    for (let i = 0; i < nSub; i++) {
        const row = Array.from(ms.slice(i * E, (i + 1) * E)).sort((a, b) => a - b);
        rows.push(row);
    }
    rows.sort((a, b) => {
        for (let j = 0; j < a.length; j++) {
            if (a[j] !== b[j]) return a[j] - b[j];
        }
        return 0;
    });
    return rows;
}

const cr = canonicalize(dr.multisets, dr.numSubgraphs);
const cs = canonicalize(ds.multisets, ds.numSubgraphs);
const msSame = cr.length === cs.length &&
    cr.every((row, i) => row.every((v, j) => Math.abs(v - cs[i][j]) < 1e-12 || (isNaN(v) && isNaN(cs[i][j]))));
console.log(`Multisets differ:   ${!msSame}`);
