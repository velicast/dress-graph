/**
 * dress.js — High-level JavaScript API for the DRESS WebAssembly module.
 *
 * Works in both browser and Node.js environments.
 *
 * Usage (ES module):
 *
 *   import { dressFit } from './dress.js';
 *
 *   const result = await dressFit({
 *     numVertices: 4,
 *     sources:     [0, 1, 2, 0],
 *     targets:     [1, 2, 3, 3],
 *   });
 *   console.log(result.edgeDress);   // Float64Array
 *   console.log(result.iterations);  // number
 */

// ── Module loader ───────────────────────────────────────────────────

let _modulePromise = null;

/**
 * Load (or return cached) the Emscripten WASM module.
 * Auto-detects browser vs Node.js.
 */
async function getModule() {
    if (_modulePromise) return _modulePromise;

    _modulePromise = (async () => {
        // In Node.js, require the glue file.  In the browser, it should
        // be loaded via <script> before this module, or bundled.
        let createDressModule;
        if (typeof globalThis.createDressModule === 'function') {
            createDressModule = globalThis.createDressModule;
        } else {
            // Node.js path — use createRequire for CJS interop
            const { createRequire } = await import('module');
            const require = createRequire(import.meta.url);
            createDressModule = require('./dress_wasm.cjs');
        }
        return createDressModule();
    })();

    return _modulePromise;
}

// ── Variant enum ────────────────────────────────────────────────────

/** Graph variant constants matching dress.h */
export const Variant = Object.freeze({
    UNDIRECTED: 0,
    DIRECTED:   1,
    FORWARD:    2,
    BACKWARD:   3,
});

// ── Core API ────────────────────────────────────────────────────────

/**
 * @typedef {Object} DressOptions
 * @property {number}         numVertices           - Number of vertices (ids in 0..N-1)
 * @property {Int32Array|number[]} sources          - Edge source vertices (0-based)
 * @property {Int32Array|number[]} targets          - Edge target vertices (0-based)
 * @property {Float64Array|number[]|null} [weights] - Optional edge weights
 * @property {number} [variant=0]                   - Variant (see Variant enum)
 * @property {number} [maxIterations=100]           - Max fitting iterations
 * @property {number} [epsilon=1e-6]                - Convergence threshold
 * @property {boolean} [precomputeIntercepts=true]  - Pre-compute intercepts
 */

/**
 * @typedef {Object} DressResult
 * @property {Int32Array}    sources     - Edge source vertices
 * @property {Int32Array}    targets     - Edge target vertices
 * @property {Float64Array}  edgeWeight  - Per-edge variant weights
 * @property {Float64Array}  edgeDress   - Per-edge dress similarity
 * @property {Float64Array}  nodeDress   - Per-node aggregated similarity
 * @property {number}        iterations  - Iterations performed
 * @property {number}        delta       - Final max per-edge change
 */

/**
 * Run the DRESS iterative fitting algorithm.
 *
 * @param {DressOptions} opts
 * @returns {Promise<DressResult>}
 */
export async function dressFit(opts) {
    const M = await getModule();

    const N = opts.numVertices;
    const E = opts.sources.length;

    if (opts.targets.length !== E) {
        throw new Error(`sources (${E}) and targets (${opts.targets.length}) must have equal length`);
    }
    if (opts.weights && opts.weights.length !== E) {
        throw new Error(`weights (${opts.weights.length}) must equal edge count (${E})`);
    }

    const variant     = opts.variant ?? Variant.UNDIRECTED;
    const maxIter     = opts.maxIterations ?? 100;
    const epsilon     = opts.epsilon ?? 1e-6;
    const precompute  = (opts.precomputeIntercepts ?? true) ? 1 : 0;

    // Allocate C arrays (ownership transfers to dress.c — freed by free_dress_graph)
    const uPtr = M._malloc(E * 4);
    const vPtr = M._malloc(E * 4);

    // Write source/target data into WASM heap
    const heap32 = M.HEAP32;
    const src = opts.sources;
    const tgt = opts.targets;
    for (let i = 0; i < E; i++) {
        heap32[(uPtr >> 2) + i] = src[i];
        heap32[(vPtr >> 2) + i] = tgt[i];
    }

    // Weights (nullable)
    let wPtr = 0;  // NULL
    if (opts.weights) {
        wPtr = M._malloc(E * 8);
        const heapF64 = M.HEAPF64;
        const w = opts.weights;
        for (let i = 0; i < E; i++) {
            heapF64[(wPtr >> 3) + i] = w[i];
        }
    }

    // Call init_dress_graph
    const g = M._init_dress_graph(N, E, uPtr, vPtr, wPtr, variant, precompute);
    if (g === 0) {
        throw new Error('init_dress_graph returned NULL');
    }

    // Allocate output params
    const iterPtr  = M._malloc(4);  // int*
    const deltaPtr = M._malloc(8);  // double*

    // Call fit
    M._fit(g, maxIter, epsilon, iterPtr, deltaPtr);

    const iterations = M.getValue(iterPtr, 'i32');
    const delta      = M.getValue(deltaPtr, 'double');

    // Read results from the C struct.
    // Struct field offsets (wasm32 — all pointers are 4 bytes):
    //   offset  0: variant        (i32)
    //   offset  4: N              (i32)
    //   offset  8: E              (i32)
    //   offset 12: *U             (ptr32)
    //   offset 16: *V             (ptr32)
    //   offset 20: *adj_offset    (ptr32)
    //   offset 24: *adj_target    (ptr32)
    //   offset 28: *adj_edge_idx  (ptr32)
    //   offset 32: *edge_weight   (ptr32)
    //   offset 36: *edge_dress    (ptr32)
    //   offset 40: *edge_dress_next (ptr32)
    //   offset 44: *node_dress    (ptr32)
    const ewPtr = M.getValue(g + 32, 'i32');  // edge_weight pointer
    const edPtr = M.getValue(g + 36, 'i32');  // edge_dress pointer
    const ndPtr = M.getValue(g + 44, 'i32');  // node_dress pointer

    // Copy results into JS-owned typed arrays
    const edgeWeight = new Float64Array(E);
    const edgeDress  = new Float64Array(E);
    const nodeDress  = new Float64Array(N);

    const heapF64 = M.HEAPF64;
    for (let i = 0; i < E; i++) {
        edgeWeight[i] = heapF64[(ewPtr >> 3) + i];
        edgeDress[i]  = heapF64[(edPtr >> 3) + i];
    }
    for (let i = 0; i < N; i++) {
        nodeDress[i] = heapF64[(ndPtr >> 3) + i];
    }

    // Copy sources/targets before free
    const sourcesOut = new Int32Array(E);
    const targetsOut = new Int32Array(E);
    for (let i = 0; i < E; i++) {
        sourcesOut[i] = heap32[(uPtr >> 2) + i];
        targetsOut[i] = heap32[(vPtr >> 2) + i];
    }

    // Clean up
    M._free_dress_graph(g);
    M._free(iterPtr);
    M._free(deltaPtr);

    return {
        sources:    sourcesOut,
        targets:    targetsOut,
        edgeWeight: edgeWeight,
        edgeDress:  edgeDress,
        nodeDress:  nodeDress,
        iterations: iterations,
        delta:      delta,
    };
}
