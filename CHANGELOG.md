# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the crate is at `0.x`, minor versions may contain breaking changes.

## [Unreleased]

## [0.1.0] - 2026-10-28

Initial release.

### Added

- `Channel`: single-prime modular arithmetic in Montgomery form, the base layer
  everything is built on.
- `rns_base!`: declares an RNS base whose primes and derived constants are
  checked and computed at compile time; a composite or repeated modulus is a
  compile error.
- `RnsMod`: the plain ring `ℤ/Mℤ`, with no sign and no overflow tracking.
- `Rns`: exact signed integers with overflow detection, exact sign and
  comparison, and division (`div_exact`, general `/` and `%`).
- `Extender`: moves integers between bases without converting to binary.
- `RnsVec` and `RnsMatrix` (feature `alloc`): a channel-major layout for many
  integers, with SIMD dot products, exact matrix products and determinants.
- Runtime AVX2 / AVX-512 kernel selection via `pulp` (feature `std`).
- Conversions to and from `num_bigint::BigInt` / `BigUint` (feature `num-bigint`).

[Unreleased]: https://github.com/supportlibrariesp4droid/residue-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/supportlibrariesp4droid/residue-rs/releases/tag/v0.1.0
