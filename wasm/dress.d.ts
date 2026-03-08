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
    /** Per-node aggregated dress similarity */
    nodeDress: Float64Array;
    /** Number of iterations performed */
    iterations: number;
    /** Final maximum per-edge change */
    delta: number;
}

/**
 * Run the DRESS iterative fitting algorithm on an edge list.
 */
export declare function dressFit(opts: DressOptions): Promise<DressResult>;

export interface DressGraphOptions {
    /** Number of vertices (vertex ids must be in 0..numVertices-1) */
    numVertices: number;
    /** Edge source vertices (0-based) */
    sources: Int32Array | number[];
    /** Edge target vertices (0-based) */
    targets: Int32Array | number[];
    /** Optional edge weights (same length as sources) */
    weights?: Float64Array | number[] | null;
    /** Graph variant (default: Variant.UNDIRECTED) */
    variant?: number;
    /** Pre-compute neighbourhood intercepts (default: false) */
    precomputeIntercepts?: boolean;
}

/**
 * A persistent DRESS graph supporting repeated fit / get calls.
 */
export declare class DressGraph {
    private constructor();
    static create(opts: DressGraphOptions): Promise<DressGraph>;
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
    /** Vertices to remove per subset (default: 0 = original graph) */
    k?: number;
    /** Graph variant (default: Variant.UNDIRECTED) */
    variant?: number;
    /** Maximum fitting iterations (default: 100) */
    maxIterations?: number;
    /** Convergence threshold / bin width (default: 1e-6) */
    epsilon?: number;
    /** Pre-compute neighbourhood intercepts (default: false) */
    precompute?: boolean;
    /** Return per-subgraph edge values (default: false) */
    keepMultisets?: boolean;
}

export interface DeltaDressResult {
    /** Bin counts of edge delta values (as Float64 for BigInt-free access) */
    histogram: Float64Array;
    /** Number of histogram bins (floor(dmax/epsilon) + 1; dmax = 2 unweighted) */
    histSize: number;
    /** Per-subgraph edge values, row-major C(N,k) × E (NaN = removed edge; null when not requested) */
    multisets: Float64Array | null;
    /** Number of subgraphs C(N,k) */
    numSubgraphs: number;
}

/**
 * Compute the Delta-k-DRESS histogram by exhaustively removing
 * all k-vertex subsets and measuring edge similarity changes.
 */
export declare function deltaDressFit(opts: DeltaDressOptions): Promise<DeltaDressResult>;

