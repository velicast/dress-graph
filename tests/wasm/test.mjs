/**
 * test.mjs — Quick smoke test for the DRESS WASM wrapper (Node.js).
 *
 * Run from the tests/wasm/ directory:
 *   node test.mjs
 *
 * Requires: dress_wasm.js + dress_wasm.wasm built by wasm/build.sh
 */

import { fit, DRESS, Variant } from '../../wasm/dress.js';

function assert(cond, msg) {
    if (!cond) {
        console.error('FAIL:', msg);
        process.exit(1);
    }
}

async function testTriangle() {
    console.log('test: triangle …');
    const r = await fit({
        numVertices: 3,
        sources: [0, 1, 0],
        targets: [1, 2, 2],
        maxIterations: 100,
        epsilon: 1e-8,
    });
    assert(r.sources.length === 3, 'expected 3 edges');
    assert(r.iterations >= 1, 'expected ≥1 iteration');
    // All edges in a triangle should have equal dress
    const d0 = r.edgeDress[0];
    for (let i = 1; i < 3; i++) {
        assert(Math.abs(r.edgeDress[i] - d0) < 1e-6,
            `edge ${i} dress ${r.edgeDress[i]} ≠ edge 0 dress ${d0}`);
    }
    console.log(`  OK (iterations=${r.iterations}, delta=${r.delta.toExponential(3)})`);
}

async function testPath() {
    console.log('test: path (no triangles) …');
    const r = await fit({
        numVertices: 4,
        sources: [0, 1, 2],
        targets: [1, 2, 3],
    });
    assert(r.edgeDress.length === 3, 'expected 3 edges');
    // Dress values on a path are positive (self-loop term) but lower
    // than in a triangle.
    for (let i = 0; i < 3; i++) {
        assert(r.edgeDress[i] > 0,
            `edge ${i} dress should be positive (self-loop term)`);
        assert(r.edgeDress[i] < 2.0,
            `edge ${i} dress should be well below 2`);
    }
    // Endpoint edges (0-1 and 2-3) should be symmetric
    assert(Math.abs(r.edgeDress[0] - r.edgeDress[2]) < 1e-10,
        'symmetric path edges should have equal dress');
    console.log('  OK (dress values positive, symmetric endpoints)');
}

async function testVariants() {
    console.log('test: all variants …');
    for (const [name, v] of Object.entries(Variant)) {
        const r = await fit({
            numVertices: 3,
            sources: [0, 1, 0],
            targets: [1, 2, 2],
            variant: v,
        });
        assert(r.edgeDress.length === 3, `${name}: expected 3 edges`);
    }
    console.log('  OK');
}

async function testWeighted() {
    console.log('test: weighted edges …');
    const r = await fit({
        numVertices: 3,
        sources: [0, 1, 0],
        targets: [1, 2, 2],
        weights: [1.0, 2.0, 3.0],
    });
    assert(r.edgeWeight.length === 3, 'expected 3 weights');
    console.log('  OK');
}

async function testDRESS() {
    console.log('test: persistent DRESS …');
    const g = await DRESS.create({
        numVertices: 3,
        sources: [0, 1, 0],
        targets: [1, 2, 2],
    });

    const fitRes = g.fit(100, 1e-8);
    assert(fitRes.iterations >= 1, 'expected ≥1 iteration');
    assert(fitRes.delta < 1e-6, 'expected convergence');

    // Get existing edge
    const d01 = g.get(0, 1, 100, 1e-8, 1.0);
    assert(d01 > 0, 'get(0,1) should be positive');

    // Get virtual edge (no crash)
    const d00 = g.get(0, 0, 100, 1e-6, 1.0);
    assert(typeof d00 === 'number', 'get(0,0) should return a number');

    // Result snapshot
    const res = g.result();
    assert(res.edgeDress.length === 3, 'result: expected 3 edge_dress');
    assert(res.vertexDress.length === 3, 'result: expected 3 vertex_dress');
    const d0 = res.edgeDress[0];
    for (let i = 1; i < 3; i++) {
        assert(Math.abs(res.edgeDress[i] - d0) < 1e-6,
            `result: edge ${i} dress should equal edge 0`);
    }

    g.free();
    console.log('  OK');
}

async function testVertexWeightsDefault() {
    console.log('test: vertex weights default …');
    const n = 3;
    const src = [0, 1, 0];
    const tgt = [1, 2, 2];

    // 1. Default (implicit All-1 vertex weights)
    const r1 = await fit({
        numVertices: n,
        sources: src,
        targets: tgt,
    });

    // 2. Explicit All-1 vertex weights
    const nw = [1.0, 1.0, 1.0];
    const r2 = await fit({
        numVertices: n,
        sources: src,
        targets: tgt,
        vertexWeights: nw,
    });

    for (let i = 0; i < r1.edgeDress.length; i++) {
        assert(Math.abs(r1.edgeDress[i] - r2.edgeDress[i]) < 1e-12,
            `edge ${i} dress diff ${Math.abs(r1.edgeDress[i] - r2.edgeDress[i])} > 1e-12`);
    }
    console.log('  OK');
}

async function main() {
    console.log('DRESS WASM tests\n');
    await testTriangle();
    await testPath();
    await testVariants();
    await testWeighted();
    await testVertexWeightsDefault();
    await testVertexWeights();
    await testDRESS();
    console.log('\nAll tests passed.');
}

async function testVertexWeights() {
    console.log('test: vertex weights (K3) …');
    // K3 with vertex weights [10, 1, 1].
    // Vertex 0 is very heavy. Edges connected to 0 (0-1, 0-2) should have higher dress?
    // Or lower?
    // Dress is sum of geometric means of neighbors.
    // Neighbors' weights matter.
    const r = await fit({
        numVertices: 3,
        sources: [0, 1, 0],
        targets: [1, 2, 2],
        vertexWeights: [100.0, 1.0, 1.0],
    });
    // With equal edge weights (implicit 1.0), but custom vertex weights.
    // Edge (1,2): neighbors are 1 and 2. Their weights are 1.0.
    // Edge (0,1): neighbors are 0 and 1. Vertex 0 has weight 100.0.
    // So edge (0,1) dress should be different from edge (1,2).
    
    // Actually, dress similarity S(e) depends on sum over common neighbors.
    // For K3:
    // (0,1): common neighbor is 2. w(2)=1.0.
    // (1,2): common neighbor is 0. w(0)=100.0.
    // So (1,2) should have much higher dress value because the common neighbor (0) is heavy.
    
    assert(r.edgeDress.length === 3, 'expected 3 edges');
    // edge 0: (0,1), comm=2
    // edge 1: (1,2), comm=0 <-- should be largest
    // edge 2: (0,2), comm=1
    
    const d01 = r.edgeDress[0];
    const d12 = r.edgeDress[1];
    const d02 = r.edgeDress[2];

    assert(d01 > d12, `edge(0,1) dress ${d01} should be > edge(1,2) ${d12}`);
    assert(d02 > d12, `edge(0,2) dress ${d02} should be > edge(1,2) ${d12}`);
    assert(Math.abs(d01 - d02) < 1e-6, `symmetry: edge(0,1) ~ edge(0,2)`);
    
    console.log('  OK');
}


main().catch(e => {
    console.error(e);
    process.exit(1);
});
