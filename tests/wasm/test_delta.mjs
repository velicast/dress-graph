/**
 * test_delta.mjs — Tests for the WASM delta-k-DRESS wrapper (Node.js).
 *
 * Run from the tests/wasm/ directory:
 *   node test_delta.mjs
 *
 * Requires: dress.js + dress_wasm.wasm built by wasm/build.sh
 */

import { deltaDressFit } from '../../wasm/dress.js';

let passed = 0;
let failed = 0;

function assert(cond, msg) {
    if (!cond) {
        console.error('  FAIL:', msg);
        failed++;
    } else {
        passed++;
    }
}

function assertEqual(a, b, msg) {
    assert(a === b, `${msg} (got ${a}, expected ${b})`);
}

function histTotal(r) {
    let s = 0;
    for (let i = 0; i < r.histogram.length; i++) s += r.histogram[i];
    return s;
}

const K3 = { numVertices: 3, sources: [0, 1, 0], targets: [1, 2, 2] };
const K4 = { numVertices: 4, sources: [0, 0, 0, 1, 1, 2], targets: [1, 2, 3, 2, 3, 3] };
const P4 = { numVertices: 4, sources: [0, 1, 2], targets: [1, 2, 3] };
const EPS = 1e-3;

async function testHistSize() {
    console.log('test: histogram size …');
    const r = await deltaDressFit({ ...K3, k: 0, epsilon: 1e-3 });
    assertEqual(r.histSize, 2001, 'histSize with eps=1e-3');
    assertEqual(r.histogram.length, r.histSize, 'histogram length == histSize');

    const r2 = await deltaDressFit({ ...K3, k: 0, epsilon: 1e-6 });
    assertEqual(r2.histSize, 2000001, 'histSize with eps=1e-6');
    console.log('  OK');
}

async function testWeightedHistSize() {
    console.log('test: weighted histogram size …');
    const r = await deltaDressFit({
        ...K3,
        weights: [1.0, 10.0, 1.0],
        k: 0,
        epsilon: 1e-3,
    });
    assert(r.histSize > 2001, `weighted histSize > 2001, got ${r.histSize}`);
    assertEqual(r.histogram.length, r.histSize, 'weighted histogram length == histSize');
    assertEqual(histTotal(r), 3, 'weighted K3 delta0 total = 3');
    console.log('  OK');
}

async function testDelta0K3() {
    console.log('test: delta-0 on K3 …');
    const r = await deltaDressFit({ ...K3, k: 0, epsilon: EPS });
    assertEqual(histTotal(r), 3, 'K3 delta0 total = 3');

    // All in top bin (dress=2.0 for complete graph)
    assert(r.histogram[r.histSize - 1] > 0, 'K3 delta0 top bin > 0');

    const nonzero = r.histogram.filter(x => x > 0).length;
    assertEqual(nonzero, 1, 'K3 delta0 single non-zero bin');
    console.log('  OK');
}

async function testDelta1K3() {
    console.log('test: delta-1 on K3 …');
    const r = await deltaDressFit({ ...K3, k: 1, epsilon: EPS });
    assertEqual(histTotal(r), 3, 'K3 delta1 total = 3');
    console.log('  OK');
}

async function testDelta2K3() {
    console.log('test: delta-2 on K3 …');
    const r = await deltaDressFit({ ...K3, k: 2, epsilon: EPS });
    assertEqual(histTotal(r), 0, 'K3 delta2 total = 0');
    console.log('  OK');
}

async function testDelta0K4() {
    console.log('test: delta-0 on K4 …');
    const r = await deltaDressFit({ ...K4, k: 0, epsilon: EPS });
    assertEqual(histTotal(r), 6, 'K4 delta0 total = 6');
    assertEqual(r.histogram[r.histSize - 1], 6, 'K4 delta0 top bin = 6');
    console.log('  OK');
}

async function testDelta1K4() {
    console.log('test: delta-1 on K4 …');
    const r = await deltaDressFit({ ...K4, k: 1, epsilon: EPS });
    assertEqual(histTotal(r), 12, 'K4 delta1 total = 12');
    assertEqual(r.histogram[r.histSize - 1], 12, 'K4 delta1 top bin = 12');
    console.log('  OK');
}

async function testDelta2K4() {
    console.log('test: delta-2 on K4 …');
    const r = await deltaDressFit({ ...K4, k: 2, epsilon: EPS });
    assertEqual(histTotal(r), 6, 'K4 delta2 total = 6');
    console.log('  OK');
}

async function testKGeqN() {
    console.log('test: k >= N (empty) …');
    const r1 = await deltaDressFit({ ...K3, k: 3, epsilon: EPS });
    assertEqual(histTotal(r1), 0, 'k=N total = 0');

    const r2 = await deltaDressFit({ ...K3, k: 10, epsilon: EPS });
    assertEqual(histTotal(r2), 0, 'k>N total = 0');
    console.log('  OK');
}

async function testPrecompute() {
    console.log('test: precompute flag …');
    const r1 = await deltaDressFit({ ...K4, k: 1, epsilon: EPS, precompute: false });
    const r2 = await deltaDressFit({ ...K4, k: 1, epsilon: EPS, precompute: true });
    assertEqual(r1.histSize, r2.histSize, 'precompute: same histSize');
    for (let i = 0; i < r1.histSize; i++) {
        assert(r1.histogram[i] === r2.histogram[i],
            `precompute: histogram[${i}] match`);
    }
    console.log('  OK');
}

async function testPathP4() {
    console.log('test: path P4 …');
    const r = await deltaDressFit({ ...P4, k: 0, epsilon: EPS });
    assertEqual(histTotal(r), 3, 'P4 delta0 total = 3');

    const nonzero = r.histogram.filter(x => x > 0).length;
    assert(nonzero >= 2, 'P4 at least 2 distinct bins');
    console.log('  OK');
}

async function testDelta1P4() {
    console.log('test: delta-1 on P4 …');
    const r = await deltaDressFit({ ...P4, k: 1, epsilon: EPS });
    assertEqual(histTotal(r), 6, 'P4 delta1 total = 6');
    console.log('  OK');
}

async function testLengthMismatch() {
    console.log('test: length mismatch …');
    try {
        await deltaDressFit({ numVertices: 3, sources: [0, 1], targets: [1, 2, 2], epsilon: EPS });
        assert(false, 'should have thrown');
    } catch (e) {
        assert(e.message.includes('length'), 'error mentions length');
    }
    console.log('  OK');
}

async function testMultisetsDisabled() {
    console.log('test: multisets disabled …');
    const r = await deltaDressFit({ ...K3, k: 0, epsilon: EPS });
    assertEqual(r.multisets, null, 'multisets should be null');
    console.log('  OK');
}

async function testMultisetsDelta0K3() {
    console.log('test: multisets Δ^0 K3 …');
    const r = await deltaDressFit({ ...K3, k: 0, epsilon: EPS, keepMultisets: true });
    assertEqual(r.numSubgraphs, 1, 'C(3,0)=1 subgraph');
    assert(r.multisets instanceof Float64Array, 'multisets is Float64Array');
    assertEqual(r.multisets.length, 3, '1 subgraph * 3 edges = 3');
    for (let i = 0; i < 3; i++) {
        assert(Math.abs(r.multisets[i] - 2.0) < EPS,
            `multisets[${i}]=${r.multisets[i]}, expected ~2.0`);
    }
    console.log('  OK');
}

async function testMultisetsDelta1K3() {
    console.log('test: multisets Δ^1 K3 NaN pattern …');
    const r = await deltaDressFit({ ...K3, k: 1, epsilon: EPS, keepMultisets: true });
    assertEqual(r.numSubgraphs, 3, 'C(3,1)=3 subgraphs');
    assertEqual(r.multisets.length, 9, '3 subgraphs * 3 edges = 9');
    const E = 3;
    for (let s = 0; s < 3; s++) {
        let nans = 0;
        for (let e = 0; e < E; e++) {
            if (Number.isNaN(r.multisets[s * E + e])) nans++;
            else assert(Math.abs(r.multisets[s * E + e] - 2.0) < EPS,
                `row ${s} col ${e}: expected ~2.0`);
        }
        assertEqual(nans, 2, `row ${s}: expected 2 NaN`);
    }
    console.log('  OK');
}

async function main() {
    console.log('Delta-k-DRESS WASM tests\n');
    await testHistSize();
    await testWeightedHistSize();
    await testDelta0K3();
    await testDelta1K3();
    await testDelta2K3();
    await testDelta0K4();
    await testDelta1K4();
    await testDelta2K4();
    await testKGeqN();
    await testPrecompute();
    await testPathP4();
    await testDelta1P4();
    await testLengthMismatch();
    await testMultisetsDisabled();
    await testMultisetsDelta0K3();
    await testMultisetsDelta1K3();
    console.log(`\n${passed} passed, ${failed} failed.`);
    if (failed > 0) process.exit(1);
}

main().catch(e => {
    console.error(e);
    process.exit(1);
});
