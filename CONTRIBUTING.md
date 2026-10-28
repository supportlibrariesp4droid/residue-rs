# Contributing to residue-rs

Thanks for your interest in the project. Bug reports, fixes, documentation and
performance work are all welcome.

## Ground rules

- The crate is `#![no_std]` and `#![forbid(unsafe_code)]`. Both are hard
  constraints: no contribution may add `unsafe`, and code must build without the
  standard library. SIMD is reached through [`pulp`](https://crates.io/crates/pulp),
  which keeps the `unsafe` on its side of the boundary.
- Correctness comes first. Every arithmetic operation is checked against
  `num-bigint`; new operations must be too.
- The public API is small on purpose. New surface area should earn its place.

## Getting started

```sh
git clone https://github.com/supportlibrariesp4droid/residue-rs
cd residue-rs
cargo test --all-features        # debug build: overflow checks are on
```

## Before you open a pull request

Run the same checks CI runs:

```sh
cargo fmt --all --check
cargo clippy --all-features --lib --tests
cargo test --release --all-features                         # SIMD kernels
cargo test --release --no-default-features --features alloc # scalar kernels
cargo test --all-features                                   # overflow checks on
```

A no_std sanity build is quick and catches accidental `std` use:

```sh
rustup target add thumbv7em-none-eabi
cargo build --lib --no-default-features --features alloc --target thumbv7em-none-eabi
```

## Tests

Tests live in `tests/` and lean on [`proptest`](https://crates.io/crates/proptest).
When you add or change arithmetic:

- Compare against `num-bigint` across bases of different sizes (the suite covers
  1 to 16 primes, including all-small and mixed bases, because those reach code
  paths that large-prime bases never touch).
- Add a targeted test for any rare branch the property tests are unlikely to hit
  often (sign-test tiers, division corrections, determinants divisible by a base
  prime, and so on).

## Benchmarks

`cargo bench --bench compare` builds GMP from source once (a few minutes) and
writes results under `target/criterion`. The FLINT comparison and the charts are
optional and documented in the README. If a change affects performance, please
include before/after numbers.

## Commit and PR style

- Keep commits focused; a readable history is valued.
- Describe *why* in the PR, not just *what*.
- Note any public API change in `CHANGELOG.md` under `Unreleased`.

## Design notes

`DESIGN.md` explains the mathematics — exact reconstruction, the overflow
coefficient, the three sign-test tiers, exact division. If you touch those
areas, read it first and keep it in sync.

## Licensing

By contributing you agree that your contribution is licensed under the
Apache License 2.0, as in [LICENSE](LICENSE).
