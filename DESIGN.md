# Design of residue-rs

This document describes the mathematics behind residue-rs and the decisions
taken while building it. It started as a plan written before any code, and was
revised as the implementation showed where the plan was wrong; those
corrections are recorded where they happened. Benchmark numbers live in the
README.

## Goal and scope

residue-rs is a library for exact integer arithmetic in a Residue Number System
(RNS), running on CPUs, with a SIMD path for arrays and matrices. It is meant
for programs that perform many operations on integers of moderate size, roughly
64 to 2000 bits: exact dot products, element-wise products over large arrays,
matrix products, determinants, and similar workloads.

Some things are deliberately out of scope. The library does not try to beat GMP
on a single multiplication of very large numbers, where GMP's algorithms are
the right tool. It does not replace `u64` or `i128` for values that fit in
them. It does not compress anything either: an RNS representation uses slightly
more memory than binary (about 3 percent with 31-bit moduli, plus a redundant
channel). The gain is in time, not in space.

## The central trade-off

Every design choice follows from one observation. With `n` channels, an
addition or multiplication in RNS costs `O(n)` word operations, and these are
independent of each other, so they run in parallel. Converting a number into
RNS or back to binary costs `O(n²)`. The total cost of a computation is

```
total = C_in + k · C_op + C_out
```

and RNS only pays off when the number of operations `k` per conversion is
large compared to `n`. The library therefore has three jobs: make operations
inside RNS cheap, make it rarely necessary to leave RNS (which means comparison,
sign and base extension must work without converting back, and these are
exactly the classical weak points of RNS), and provide operations such as dot
products and matrix products in which `k` is large by construction.

## Two types for two kinds of arithmetic

For a base of distinct primes `m_1, ..., m_n` with product `M`, the Chinese
Remainder Theorem gives a ring isomorphism

```
φ : ℤ/Mℤ → ℤ/m_1ℤ × ... × ℤ/m_nℤ,    x ↦ (x mod m_1, ..., x mod m_n)
```

This is an isomorphism of `ℤ/Mℤ`, not of `ℤ`. Most users want true integers, so
the library keeps the two worlds apart. `RnsMod` is an element of the ring
`ℤ/Mℤ`: it never overflows and has no sign or order, since `ℤ/Mℤ` is not an
ordered ring. `Rns` is an exact signed integer: it carries extra information
that makes sign, comparison and overflow detection exact, and it refuses any
operation whose result might not fit.

## Choice of moduli

The moduli are distinct odd primes below `2^31`, and residues are stored in
`u32`. Three reasons decided this. Vector instruction sets offer a 32 by 32 to
64 bit multiplication (`vpmuludq` on x86, `vmull_u32` on ARM) but no 64 by 64
to 128 bit one, so 32-bit residues are what SIMD can multiply. Keeping one bit
of headroom means that `a + b` never overflows a `u32`, which makes modular
addition branchless and leaves room for delayed reduction. Using primes rather
than merely coprime moduli makes every channel a field, so every non-zero
residue is invertible; this is needed for exact division and for the
determinant.

A base of `n` such primes covers about `31n` bits. The `largest_primes` table
starts at `2^31 − 1`, which is itself prime (the Mersenne prime `M31`).

## Arithmetic in one channel

Everything rests on a `Channel`: one prime `p` together with a few precomputed
constants.

Addition computes `s = a + b`, which fits in 32 bits, and returns
`min(s, s − p)` with a wrapping subtraction. If `s ≥ p` the subtraction gives
the smaller value; otherwise it wraps around to something larger than `s`. There
is no branch, and in SIMD this is a single `pminud`. Subtraction works the same
way.

Multiplication uses Montgomery form with `R = 2^32`: a residue `a` is stored as
`ã = a·R mod p`, and the reduction `REDC(t) = t·R⁻¹ mod p` needs three 32-bit
multiplications and no division. The constant `−p⁻¹ mod 2^32` is computed by
Newton-Hensel lifting, starting from `p` itself, which is correct modulo 8 for
every odd `p`; five steps give all 32 bits. Multiplication by a fixed constant,
used in Garner's algorithm, follows Shoup's method with a precomputed quotient
estimate. Inverses come from Fermat's little theorem, `a⁻¹ = a^(p−2)`.

## Bases at compile time

A base is declared with the `rns_base!` macro, which creates a zero-sized
marker type. Numbers on different bases are therefore different types and
cannot be mixed by accident. `Base::new` is a `const fn`: it checks that every
modulus is a prime in `[3, 2^31)` with a deterministic Miller-Rabin test
(witnesses 2, 3, 5 and 7 are exact below `3 215 031 751`), rejects duplicates,
and computes all constants, including `M`, the cofactors `M_i = M/m_i`, the CRT
coefficients and the conversion tables. Because the base lives in a `static`, an
invalid base is a compile error.

## The redundant channel

`Rns` stores, besides the `n` residues, the value `x mod 2^64`, maintained with
ordinary wrapping `u64` arithmetic. This channel follows the true signed
integer `x`, not its representative in `[0, M)`, which is only valid as long as
no operation leaves the allowed range; range tracking guarantees that.

With `M_i = M/m_i`, `y_i = M_i⁻¹ mod m_i` and `ξ_i = r_i·y_i mod m_i`, the
sum `S = Σ ξ_i·M_i` satisfies `S ≡ x (mod M)` and `0 ≤ S < n·M`, so
`S = x + α'·M` for an integer `α'` between 0 and `n`. Reducing modulo `2^64`,
where `M` is invertible because it is odd,

```
α' = (S mod 2^64 − x mod 2^64) · M⁻¹ mod 2^64
```

and both terms on the right are known: `S mod 2^64` is a wrapping sum, and
`x mod 2^64` is the redundant channel. Since `α' ≤ n`, the value is exact. This
is the method of Shenoy and Kumaresan (1989), adapted so that the redundant
channel tracks the signed integer. It gives the exact signed value
`x = S − α'·M` without Garner's sequential chain, and it is the basis of base
extension and of the sign test. The `R⁻¹` factor of the Montgomery form costs
nothing here: a Montgomery product of `r̃_i` with the ordinary constant `y_i`
yields `ξ_i` directly in ordinary form.

## Conversions

To convert a binary integer into RNS, each 64-bit limb `ℓ = h·2^32 + l` at
position `j` is split into two halves. With a precomputed table of
`R²·2^(64j) mod p` and `R³·2^(64j) mod p`, the Montgomery residue of the whole
number is a sum of `REDC(l·c_lo) + REDC(h·c_hi)` over the limbs: two
reductions per limb and per channel, and no division. A first version used
128-bit divisions and cost ten times more.

Converting back uses `x = S − α'·M`, computed in two's complement over only the
limbs that an in-range value can occupy (`ceil(bits(M)/64)`, about half of
`n`). For `RnsMod`, and for results whose redundant channel is unknown, such as a
division candidate or a determinant, Garner's mixed-radix algorithm is used
instead, since it needs only the residues.

## Range tracking

Every `Rns` carries a certified bound `bits` with `|x| < 2^bits`. Addition and
subtraction give `max(a, b) + 1`, multiplication gives `a + b`, and negation
keeps the bound. The largest allowed bound is `max_bits = bits(M) − 2`, which
guarantees `|x| < M/2` and therefore a unique signed representative.

An operation whose bound would exceed `max_bits` is refused. The `checked_`
methods return `None`, and the operators panic. Unlike primitive integers in
release builds, they never wrap, because a wrapped value would silently break
the redundant channel. Bounds can become pessimistic, for instance after
`(a + b) − b`, and `tighten` replaces the bound with the exact bit length at the
cost of one reconstruction.

Comparison never needs a larger bound. If the signs of `a` and `b` differ, they
decide. If they agree, `|a − b| < max(|a|, |b|)`, so the difference fits under
the existing bound and its sign is the answer.

## Sign detection

Sign is the classical problem of RNS. The plan proposed a fixed-point estimate
of `x/M` with about `log2 M` bits of precision. During implementation it turned
out that at that precision the estimate costs `n²/2` word operations, the same
as reconstructing `x` outright, so it was no faster. The implemented test has
three tiers and is exact in all of them.

When the bound certifies `|x| < 2^63`, the redundant channel read as an `i64`
is `x` itself, and the sign costs nothing. Otherwise the test computes
`g = Σ ξ_i·⌊2^64/m_i⌋ − α'·2^64`, which satisfies
`g ≤ 2^64·x/M ≤ g + Σ ξ_i`; if that interval excludes zero, the sign is known
after `O(n)` work. This decides values that are large compared to `M`. The
values in between, roughly `2^63 ≤ |x| ≲ n·2^31·M/2^64`, fall back to exact
reconstruction.

The tests count which tier decides for values of every bit length and require
all three to be reached; on a base of 9 channels the split was 512, 296 and
1416 out of 2224 values. A possible improvement is to refine the second tier 64
bits at a time, so that the cost of the third tier grows with how small `|x|` is
compared to `M` instead of always being `n²/2`.

## Base extension

The residue of `x` modulo any new prime `q` is

```
x mod q = (Σ ξ_i·(M_i mod q) − α'·(M mod q)) mod q
```

which is exact because `α'` is exact. With the table `M_i mod q_j` computed
once, an `Extender` moves a value between two bases in `O(n_s · n_t)` word
operations, with no reconstruction. Moving to a wider base makes room for
larger results; moving to a narrower one succeeds when the value fits.

## Division

Division cannot be done locally: `ℤ/Mℤ` is not a domain, and each digit of a
quotient depends on the whole number. The general operators (`/`, `%`,
`div_euclid`, `rem_euclid`) therefore reconstruct both operands, divide in
binary and convert back, at a cost of `O(n²)`. When every in-range value fits in
two limbs, native `u128` division is used. Otherwise the division is Knuth's
Algorithm D with 64-bit digits. Its rare "add back" correction, taken with
probability about `2^−63` on random inputs, is exercised by tests built from
boundary digits (0, 1, `2^63`, `2^64 − 1`), where it runs thousands of times.

Exact division is cheaper. If `y` divides `x`, the quotient is `x·y⁻¹` in
every channel, and the residues of the candidate `q` cost `O(n)` plus one
inverse per channel. The library proves that the candidate is the quotient
rather than assuming it. Let `bq` and `by` be the exact bit lengths of `q` and
`y`. If `y` divides `x`, the true quotient is in range and has the same
residues, so it is the candidate. If `bq + by ≥ bits(M)`, then
`|q·y| ≥ 2^(bq+by−2) ≥ 2^max_bits > |x|`, so `q·y ≠ x`. Otherwise `|q·y| < M`,
and `q·y ≡ x (mod M)` forces `q·y − x` to be one of `−M`, 0 or `M`; since `M` is
odd, the redundant channel tells 0 apart from `±M`. When `y` shares a prime
with `M`, it has no inverse and the general division is used.

## Batches

A scalar `Rns` keeps the residues of one number side by side, each with a
different modulus. `RnsVec` turns this around: channel `i` is one contiguous
`Vec<u32>` holding residue `i` of every element, next to a `Vec<u64>` for the
redundant channel and a single bound for the whole vector. A channel is then
processed with a single modulus, broadcast once, which is the shape SIMD needs.

The kernels are plain loops without branches, written so that LLVM vectorizes
them. With the `std` feature they run inside a function compiled for AVX2 or
AVX-512, chosen at runtime through the `pulp` crate, so the crate itself
contains no `unsafe` code. One lesson came from measuring: a closure passed to
`pulp` is not inlined into the function compiled for AVX, so the first version
ran SSE2 code on the AVX path. The timings and the generated assembly both
showed it. Every batch operation is now a concrete type whose `run` method is
`inline(always)`.

The dot product uses delayed reduction. Each product of two residues is below
`p² < 2^62`; its low and high 32-bit halves go into two separate `u64` sums,
which cannot overflow for fewer than `2^32` terms. The loop contains only a
multiplication, a mask, a shift and two additions, and each channel is reduced
once at the end.

## Matrices and determinants

`RnsMatrix` stores its entries in an `RnsVec` in row-major order, so each
channel is a contiguous matrix of residues and the matrix product is `n`
independent `u32` matrix products. The inner loop multiplies one entry of the
left matrix by a row of the right one, which vectorizes across columns, and
accumulates the two 32-bit halves as in the dot product. The final reduction
needs no division. The products are of Montgomery forms, `t = ab·R²`, and the
wanted result is `Σab·R = Σt·R⁻¹`. With `Σt = h·2^32 + l` and `R = 2^32`, that
is `h + REDC(l)`, and `h mod p = to_mont(REDC(h))`: three reductions per output
entry. This requires the inner dimension to be below the smallest modulus;
otherwise a slower path with divisions is taken.

The determinant uses the multi-modular method instead of fraction-free Bareiss
elimination. Gaussian elimination runs separately over each field `ℤ/m_iℤ`,
each channel choosing its own pivots, since a zero modulo one prime is not a
zero modulo another. The `n` determinants are then combined by the CRT.
Hadamard's inequality bounds the result in advance: with `n` rows and entries
`|a_ij| < 2^b`, `|det A| < 2^(n·b + ceil(n·ceil(log2 n)/2))`, and the
computation is refused if that exceeds `max_bits`. Intermediate values never
need to fit in the range, because the whole elimination happens modulo `M` and
only the final determinant has to be in range.

## Costs

| Operation | `RnsMod` | `Rns` |
|---|---|---|
| addition, subtraction, multiplication | O(n) | O(n) |
| conversion from binary | O(n·L) | O(n·L) |
| conversion to binary | O(n²), Garner | O(n²), no sequential chain |
| extension to one new modulus | | O(n) |
| sign and comparison | not defined | O(1), O(n) or O(n²) by tier |
| general division | | O(n²) |
| exact division | | O(n) plus one reconstruction |

Here `L` is the number of 64-bit limbs of the binary input.

## Testing

Every operation is checked against `num-bigint`, an independent implementation,
with property tests on bases from 1 to 16 channels. These include a base of very
small primes (3 to 13) and a base mixing large and small primes, which reach code
paths that bases of large primes never touch. Generated values have a bit length
chosen uniformly at random, so small, medium and near-full-range values all
appear. Some branches are rare enough to need dedicated tests: the three sign
tiers, the add-back step of Algorithm D, determinants divisible by moduli of the
base, and dot products of hundreds of thousands of maximal residues. The SIMD
build and the scalar build run the same suite, and so does a debug build, where
Rust checks every arithmetic overflow.

## Measuring

Three lessons from benchmarking shaped the code. The closure problem described
above was invisible until the generated assembly was inspected. Without
`black_box` on the inputs, the compiler moved pure computations out of the
timed loop, and an early measurement showed a result faster than AVX-512 should
allow. Beyond about a million elements, element-wise operations are limited by
memory bandwidth, at about 10 ns per element whatever the instruction set.

A fourth lesson came from the cloud machine itself. During one run the session
moved to a different processor model, and the matrix product of residue went
from 0.33 ms to 0.41 ms while GMP and FLINT did not change. Results are only
compared within a single run, and the published figures all come from one.

## Not done yet

Modular arithmetic modulo a large odd `N` with RNS Montgomery multiplication
(Bajard, Didier and Kornerup; Kawamura and others), the usual RNS approach in
cryptography, needs two bases and two base extensions per multiplication and is
planned as the next step.

FLINT is faster than residue on matrix products and on large determinants. It
uses the same multi-modular idea, but with primes of about 60 bits, so it needs
roughly half as many channels for the same range, and this is the most likely
source of most of the gap. Moduli of 62 bits with 128-bit products would trade
the cheap 32-bit SIMD lanes for fewer channels. Whether that wins on current
CPUs has to be measured; it is the main open question on performance.

Other possible work includes Python bindings,
multithreading across channels (which are independent, so the determinant and
the matrix product parallelize naturally), and runtime dispatch on ARM, where
the kernels are currently compiled for the baseline NEON instruction set only.

## References

Szabó and Tanaka, *Residue Arithmetic and Its Applications to Computer
Technology*, 1967. Garner, "The residue number system", IRE Transactions on
Electronic Computers, 1959. Montgomery, "Modular multiplication without trial
division", Mathematics of Computation, 1985. Shenoy and Kumaresan, "Fast base
extension using a redundant modulus in RNS", IEEE Transactions on Computers,
1989. Bajard, Didier and Kornerup, "An RNS Montgomery modular multiplication
algorithm", IEEE Transactions on Computers, 1998. Kawamura, Koike, Sano and
Shimbo, "Cox-Rower architecture for fast parallel Montgomery multiplication",
EUROCRYPT 2000. Knuth, *The Art of Computer Programming*, volume 2, section
4.3.1. Shoup's constant multiplication is described in the NTL library.
