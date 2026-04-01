/**
 * TypeScript type declarations for the DRESS WebAssembly wrapper.
 */

export declare const Variant: {
    readonly UNDIRECTED: 0;
    readonly DIRECTED: 1;
    readonly FORWARD: 2;
    readonly BACKWARD: 3;
};

export interface DressOptions {
    /** Number of vertices (vertex ids must be in 0..numVertices-1) */
    numVertices: number;
    /** Edge source vertices (0-based) */
    sources: Int32Array | number[];
    /** Edge target vertices (0-based) */
    targets: Int32Array | number[];
    /** Optional edge weights (same length as sources) */
    weights?: Float64Array | number[] | null;
    /** Optional vertex weights (length numVertices) */
    vertexWeights?: Float64Array | number[] | null;
    /** Graph variant (default: Variant.UNDIRECTED) */
    variant?: number;
    /** Maximum fitting iterations (default: 100) */
    maxIterations?: number;
    /** Convergence threshold (default: 1e-6) */
    epsilon?: number;
    /** Pre-compute neighbourhood intercepts (default: false) */
    precomputeIntercepts?: boolean;
}

export interface DressResult {
    /** Edge source vertices */
    sources: Int32Array;
    /** Edge target vertices */
    targets: Int32Array;
    /** Per-edge variant-specific weights */
    edgeWeight: Float64Array;
    /** Per-edge dress similarity values */
    edgeDress: Float64Array;
    /** Per-vertex aggregated dress similarity */
    vertexDress: Float64Array;
    /** Number of iterations performed */
    iterations: number;
    /** Final maximum per-edge change */
    delta: number;
}

/**
 * Run the DRESS iterative fitting algorithm on an edge list.
 */
export declare function fit(opts: DressOptions): Promise<DressResult>;

export interface DRESSOptions {
    /** Number of vertices (vertex ids must be in 0..numVertices-1) */
    numVertices: number;
    /** Edge source vertices (0-based) */
    sources: Int32Array | number[];
    /** Edge target vertices (0-based) */
    targets: Int32Array | number[];
    /** Optional edge weights (same length as sources) */
    weights?: Float64Array | number[] | null;
    /** Optional vertex weights (length numVertices) */
    vertexWeights?: Float64Array | number[] | null;
    /** Graph variant (default: Variant.UNDIRECTED) */
    variant?: number;
    /** Pre-compute neighbourhood intercepts (default: false) */
    precomputeIntercepts?: boolean;
}

/**
 * A persistent DRESS graph supporting repeated fit / get calls.
 */
export declare class DRESS {
    private constructor();
    static create(opts: DRESSOptions): Promise<DRESS>;
    fit(maxIterations?: number, epsilon?: number): { iterations: number; delta: number };
    get(u: number, v: number, maxIterations?: number, epsilon?: number, edgeWeight?: number): number;
    result(): DressResult;
    free(): void;
}

export interface DeltaDressOptions {
    /** Number of vertices (vertex ids must be in 0..numVertices-1) */
    numVertices: number;
    /** Edge source vertices (0-based) */
    sources: Int32Array | number[];
    /** Edge target vertices (0-based) */
    targets: Int32Array | number[];
    /** Optional edge weights (same length as sources) */
    weights?: Float64Array | number[] | null;
    /** Optional vertex weights (length numVertices) */
    vertexWeights?: Float64Array | number[] | null;
    /** Vertices to remove per subset (default: 0 = original graph) */
    k?: number;
    /** Graph variant (default: Variant.UNDIRECTED) */
    variant?: number;
    /** Maximum fitting iterations (default: 100) */
    maxIterations?: number;
    /** Convergence threshold / bin width (default: 1e-6) */
    epsilon?: number;
    /** Number of random subgraphs to sample (0 = exhaustive, default: 0) */
    nSamples?: number;
    /** Random seed for sampling (default: 0) */
    seed?: number;
    /** Pre-compute neighbourhood intercepts (default: false) */
    precompute?: boolean;
    /** Return per-subgraph edge values (default: false) */
    keepMultisets?: boolean;
    /** Compute histogram (default: true) */
    computeHistogram?: boolean;
}

export interface DeltaDressResult {
    /** Exact sparse histogram entries as (value, count) pairs */
    histogram: HistogramEntry[];
    /** Per-subgraph edge values, row-major C(N,k) × E (NaN = removed edge; null when not requested) */
    multisets: Float64Array | null;
    /** Number of subgraphs C(N,k) */
    numSubgraphs: number;
}

export interface HistogramEntry {
    value: number;
    count: number;
}

/**
 * Compute the Delta-k-DRESS histogram by exhaustively removing
 * all k-vertex subsets and measuring edge similarity changes.
 */
export declare function deltaFit(opts: DeltaDressOptions): Promise<DeltaDressResult>;

export interface NablaDressOptions {
    /** Number of vertices (vertex ids must be in 0..numVertices-1) */
    numVertices: number;
    /** Edge source vertices (0-based) */
    sources: Int32Array | number[];
    /** Edge target vertices (0-based) */
    targets: Int32Array | number[];
    /** Optional edge weights (same length as sources) */
    weights?: Float64Array | number[] | null;
    /** Optional vertex weights (length numVertices) */
    vertexWeights?: Float64Array | number[] | null;
    /** Vertices to remove per tuple (default: 0 = original graph) */
    k?: number;
    /** Graph variant (default: Variant.UNDIRECTED) */
    variant?: number;
    /** Maximum fitting iterations (default: 100) */
    maxIterations?: number;
    /** Convergence threshold / bin width (default: 1e-6) */
    epsilon?: number;
    /** Number of random tuples to sample (0 = exhaustive, default: 0) */
    nSamples?: number;
    /** Random seed for sampling (default: 0) */
    seed?: number;
    /** Pre-compute neighbourhood intercepts (default: false) */
    precompute?: boolean;
    /** Return per-tuple edge values (default: false) */
    keepMultisets?: boolean;
    /** Compute histogram (default: true) */
    computeHistogram?: boolean;
}

export interface NablaDressResult {
    /** Exact sparse histogram entries as (value, count) pairs */
    histogram: HistogramEntry[];
    /** Per-tuple edge values, row-major (null when not requested) */
    multisets: Float64Array | null;
    /** Number of tuples */
    numTuples: number;
}

/**
 * Compute the Nabla-k-DRESS histogram by exhaustively removing
 * all k-vertex tuples and measuring edge similarity changes.
 */
export declare function nablaFit(opts: NablaDressOptions): Promise<NablaDressResult>;

