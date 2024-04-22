export function mcmc(initial, nWalkers, nIterations, logP) {
    const walkers = init(initial, nWalkers)
    for (let i = 0; i < nIterations; i++) {
        for (let w = 0; w < nWalkers; w++) {
            const v = randomInt(nWalkers)
            if (v != w) {
                const z = genZ()
                const candidate = move(walkers[w], walkers[v], z)
                const oldP = logP(walkers[w])
                const newP = logP(candidate)
                if (accept(oldP, newP, z, candidate.length)) {
                    console.log("+")
                    walkers[w] = candidate
                } else {
                    console.log(".")
                }
            }
        }
    }
    return walkers
}

function init(initial, nWalkers) {
    const walkers = []
    for (let i = 0; i < nWalkers; i++)
        walkers.push(initWalker(initial))
    return walkers
}

function initWalker(initial) {
    const walker = []
    for (let i = 0; i < initial.length; i++)
        walker.push(initial[i] + Math.random())
    return walker
}

function move(walker1, walker2, z) {
    const result = []
    for (let i = 0; i < walker1.length; i++) {
        result[i] = walker1[i] + z * (walker2[i] - walker1[i])
    }
    return result
}

function genZ() {
    return Math.pow(Math.random() + 1, 2) / 2
}

function accept(oldP, newP, z, nDims) {
    const diffP = newP - oldP
    const q = Math.min(0, (Math.log(z) * (nDims - 1)) + diffP)
    const r = Math.random()
    return (Math.log(r) < q)
}

function randomInt(n) {
    return Math.floor(Math.random() * n)
}