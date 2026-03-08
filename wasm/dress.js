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
 * @property {boolean} [precomputeIntercepts=false]  - Pre-compute intercepts
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
    const precompute  = (opts.precomputeIntercepts ?? false) ? 1 : 0;

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

    // Call dress_fit
    M._dress_fit(g, maxIter, epsilon, iterPtr, deltaPtr);

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
    //   offset 32: *W             (ptr32)  raw input weights
    //   offset 36: *edge_weight   (ptr32)
    //   offset 40: *edge_dress    (ptr32)
    //   offset 44: *edge_dress_next (ptr32)
    //   offset 48: *node_dress    (ptr32)
    const ewPtr = M.getValue(g + 36, 'i32');  // edge_weight pointer
    const edPtr = M.getValue(g + 40, 'i32');  // edge_dress pointer
    const ndPtr = M.getValue(g + 48, 'i32');  // node_dress pointer

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

// ── Persistent DressGraph class ─────────────────────────────────────

/**
 * A persistent DRESS graph that supports repeated fit and get calls.
 *
 * Usage:
 *   const g = await DressGraph.create({
 *     numVertices: 4,
 *     sources: [0,1,2,0],
 *     targets: [1,2,3,3],
 *   });
 *   g.fit();                          // fit with defaults
 *   const d = g.get(0, 2);           // virtual edge query
 *   const res = g.result();          // snapshot of current results
 *   g.free();                        // explicitly free C graph
 */
export class DressGraph {
    /** @private */
    constructor(module, gPtr, n, e, sources, targets) {
        this._M = module;
        this._g = gPtr;
        this._n = n;
        this._e = e;
        this._sources = new Int32Array(sources);
        this._targets = new Int32Array(targets);
        this._iterPtr  = module._malloc(4);
        this._deltaPtr = module._malloc(8);
    }

    /**
     * Create a persistent DressGraph.
     *
     * @param {Object} opts
     * @param {number}              opts.numVertices
     * @param {Int32Array|number[]} opts.sources
     * @param {Int32Array|number[]} opts.targets
     * @param {Float64Array|number[]|null} [opts.weights]
     * @param {number} [opts.variant=0]
     * @param {boolean} [opts.precomputeIntercepts=false]
     * @returns {Promise<DressGraph>}
     */
    static async create(opts) {
        const M = await getModule();

        const N = opts.numVertices;
        const E = opts.sources.length;

        if (opts.targets.length !== E)
            throw new Error('sources and targets must have equal length');

        const variant    = opts.variant ?? Variant.UNDIRECTED;
        const precompute = (opts.precomputeIntercepts ?? false) ? 1 : 0;

        const uPtr = M._malloc(E * 4);
        const vPtr = M._malloc(E * 4);
        for (let i = 0; i < E; i++) {
            M.HEAP32[(uPtr >> 2) + i] = opts.sources[i];
            M.HEAP32[(vPtr >> 2) + i] = opts.targets[i];
        }

        let wPtr = 0;
        if (opts.weights && opts.weights.length === E) {
            wPtr = M._malloc(E * 8);
            for (let i = 0; i < E; i++) {
                M.HEAPF64[(wPtr >> 3) + i] = opts.weights[i];
            }
        }

        const g = M._init_dress_graph(N, E, uPtr, vPtr, wPtr, variant, precompute);
        if (g === 0) throw new Error('init_dress_graph returned NULL');

        return new DressGraph(M, g, N, E, opts.sources, opts.targets);
    }

    /**
     * Fit the DRESS model.
     * @param {number} [maxIterations=100]
     * @param {number} [epsilon=1e-6]
     * @returns {{iterations: number, delta: number}}
     */
    fit(maxIterations = 100, epsilon = 1e-6) {
        if (!this._g) throw new Error('DressGraph already freed');
        const M = this._M;
        M._dress_fit(this._g, maxIterations, epsilon, this._iterPtr, this._deltaPtr);
        return {
            iterations: M.getValue(this._iterPtr, 'i32'),
            delta:      M.getValue(this._deltaPtr, 'double'),
        };
    }

    /**
     * Query the DRESS value for an edge (existing or virtual).
     * @param {number} u - source vertex (0-based)
     * @param {number} v - target vertex (0-based)
     * @param {number} [maxIterations=100]
     * @param {number} [epsilon=1e-6]
     * @param {number} [edgeWeight=1.0]
     * @returns {number}
     */
    get(u, v, maxIterations = 100, epsilon = 1e-6, edgeWeight = 1.0) {
        if (!this._g) throw new Error('DressGraph already freed');
        return this._M._dress_get(this._g, u, v, maxIterations, epsilon, edgeWeight);
    }

    /**
     * Extract a snapshot of the current results.
     * @returns {DressResult}
     */
    result() {
        if (!this._g) throw new Error('DressGraph already freed');
        const M = this._M;
        const E = this._e;
        const N = this._n;

        // WASM32 offsets
        const ewPtr = M.getValue(this._g + 36, 'i32');
        const edPtr = M.getValue(this._g + 40, 'i32');
        const ndPtr = M.getValue(this._g + 48, 'i32');

        const edgeWeight = new Float64Array(E);
        const edgeDress  = new Float64Array(E);
        const nodeDress  = new Float64Array(N);

        for (let i = 0; i < E; i++) {
            edgeWeight[i] = M.HEAPF64[(ewPtr >> 3) + i];
            edgeDress[i]  = M.HEAPF64[(edPtr >> 3) + i];
        }
        for (let i = 0; i < N; i++) {
            nodeDress[i] = M.HEAPF64[(ndPtr >> 3) + i];
        }

        return {
            sources:    new Int32Array(this._sources),
            targets:    new Int32Array(this._targets),
            edgeWeight,
            edgeDress,
            nodeDress,
            iterations: 0,
            delta:      0,
        };
    }

    /**
     * Free the underlying C graph.
     */
    free() {
        if (this._g) {
            const M = this._M;
            M._free_dress_graph(this._g);
            M._free(this._iterPtr);
            M._free(this._deltaPtr);
            this._g = 0;
        }
    }
}

// ── Delta-k-DRESS API ───────────────────────────────────────────────

/**
 * @typedef {Object} DeltaDressOptions
 * @property {number}              numVertices       - Number of vertices (ids in 0..N-1)
 * @property {Int32Array|number[]} sources           - Edge source vertices (0-based)
 * @property {Int32Array|number[]} targets           - Edge target vertices (0-based)
 * @property {number} [k=0]                          - Vertices to remove per subset
 * @property {number} [variant=0]                    - Variant (see Variant enum)
 * @property {number} [maxIterations=100]            - Max fitting iterations
 * @property {number} [epsilon=1e-6]                 - Convergence threshold / bin width
 * @property {boolean} [precompute=false]            - Pre-compute intercepts
 */

/**
 * @typedef {Object} DeltaDressResult
 * @property {Float64Array} histogram - Bin counts (as Float64 for BigInt-free access)
 * @property {number}       histSize  - Number of bins
 */

/**
 * Compute the Delta-k-DRESS histogram.
 *
 * Exhaustively removes all k-vertex subsets and measures
 * the change in edge similarity values.
 *
 * @param {DeltaDressOptions} opts
 * @returns {Promise<DeltaDressResult>}
 */
export async function deltaDressFit(opts) {
    const M = await getModule();

    const N = opts.numVertices;
    const E = opts.sources.length;

    if (opts.targets.length !== E) {
        throw new Error(`sources (${E}) and targets (${opts.targets.length}) must have equal length`);
    }

    const k           = opts.k ?? 0;
    const variant     = opts.variant ?? Variant.UNDIRECTED;
    const maxIter     = opts.maxIterations ?? 100;
    const epsilon     = opts.epsilon ?? 1e-6;
    const precompute  = (opts.precompute ?? false) ? 1 : 0;
    const keepMS      = (opts.keepMultisets ?? false) ? 1 : 0;

    // Allocate C arrays (ownership transfers to init_dress_graph)
    const uPtr = M._malloc(E * 4);
    const vPtr = M._malloc(E * 4);

    const heap32 = M.HEAP32;
    const src = opts.sources;
    const tgt = opts.targets;
    for (let i = 0; i < E; i++) {
        heap32[(uPtr >> 2) + i] = src[i];
        heap32[(vPtr >> 2) + i] = tgt[i];
    }

    let wPtr = 0;
    if (opts.weights && opts.weights.length === E) {
        wPtr = M._malloc(E * 8);
        const heapF64 = M.HEAPF64;
        for (let i = 0; i < E; i++) {
            heapF64[(wPtr >> 3) + i] = opts.weights[i];
        }
    }

    // Build graph
    const g = M._init_dress_graph(N, E, uPtr, vPtr, wPtr, variant, precompute);
    if (g === 0) {
        throw new Error('init_dress_graph returned NULL');
    }

    // Allocate out-param for hist_size
    const histSizePtr = M._malloc(4);

    // Allocate out-params for multisets (pointer-to-pointer and num_subgraphs)
    let msPtrPtr = 0;
    let numSubPtr = 0;
    if (keepMS) {
        msPtrPtr   = M._malloc(4);  // pointer to double*
        numSubPtr  = M._malloc(8);  // int64_t
        M.setValue(msPtrPtr, 0, 'i32');
        M.setValue(numSubPtr, 0, 'i32');
        M.setValue(numSubPtr + 4, 0, 'i32');
    }

    // Call delta_dress_fit  (returns int64_t* — pointer to histogram on heap)
    const histPtr = M._delta_dress_fit(g, k, maxIter, epsilon, histSizePtr,
                                 keepMS, msPtrPtr, numSubPtr);

    const histSize = M.getValue(histSizePtr, 'i32');

    // Copy histogram into JS Float64Array (int64 values cast to double)
    // WASM int64_t is 8 bytes; read as pairs of i32 (little-endian)
    const histogram = new Float64Array(histSize);
    for (let i = 0; i < histSize; i++) {
        const lo = M.HEAPU32[(histPtr >> 2) + i * 2];
        const hi = M.HEAP32[(histPtr >> 2) + i * 2 + 1];
        histogram[i] = hi * 4294967296 + lo;
    }

    // Extract multisets if requested
    let multisets = null;
    let numSubgraphs = 0;
    if (keepMS && msPtrPtr) {
        const msPtr = M.getValue(msPtrPtr, 'i32');  // double*
        // Read int64 num_subgraphs as lo/hi pair
        const nsLo = M.HEAPU32[(numSubPtr >> 2)];
        const nsHi = M.HEAP32[(numSubPtr >> 2) + 1];
        numSubgraphs = nsHi * 4294967296 + nsLo;

        if (msPtr !== 0 && numSubgraphs > 0) {
            const totalVals = numSubgraphs * E;
            multisets = new Float64Array(totalVals);
            for (let i = 0; i < totalVals; i++) {
                multisets[i] = M.HEAPF64[(msPtr >> 3) + i];
            }
            M._free(msPtr);
        }
        M._free(msPtrPtr);
        M._free(numSubPtr);
    }

    // Cleanup
    M._free(histPtr);
    M._free(histSizePtr);
    M._free_dress_graph(g);

    return {
        histogram:     histogram,
        histSize:      histSize,
        multisets:     multisets,
        numSubgraphs:  numSubgraphs,
    };
}

