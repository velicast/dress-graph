/**
 * test.mjs — Quick smoke test for the DRESS WASM wrapper (Node.js).
 *
 * Run from the tests/wasm/ directory:
 *   node test.mjs
 *
 * Requires: dress_wasm.js + dress_wasm.wasm built by wasm/build.sh
 */

import { dressFit, Variant } from '../../wasm/dress.js';

function assert(cond, msg) {
    if (!cond) {
        console.error('FAIL:', msg);
        process.exit(1);
    }
}

async function testTriangle() {
    console.log('test: triangle …');
    const r = await dressFit({
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
    const r = await dressFit({
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
        const r = await dressFit({
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
    const r = await dressFit({
        numVertices: 3,
        sources: [0, 1, 0],
        targets: [1, 2, 2],
        weights: [1.0, 2.0, 3.0],
    });
    assert(r.edgeWeight.length === 3, 'expected 3 weights');
    console.log('  OK');
}

async function main() {
    console.log('DRESS WASM tests\n');
    await testTriangle();
    await testPath();
    await testVariants();
    await testWeighted();
    console.log('\nAll tests passed.');
}

main().catch(e => {
    console.error(e);
    process.exit(1);
});
