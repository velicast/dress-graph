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
    /** Pre-compute neighbourhood intercepts (default: true) */
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
