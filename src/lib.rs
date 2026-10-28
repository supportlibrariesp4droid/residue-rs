#![no_std]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
//! # residue
//!
//! Exact integer arithmetic in a Residue Number System (RNS).
//!
//! An integer is represented by its remainders modulo a set of distinct primes.
//! Addition, subtraction and multiplication then act independently on each
//! remainder, with no carry propagation, which makes the representation
//! naturally parallel and well suited to SIMD.
//!
//! ```
//! use residue::{rns_base, RnsMod};
//!
//! rns_base!(pub B4: largest 4);              // ~124-bit dynamic range
//! type Z = RnsMod<B4, 4>;
//!
//! let a: Z = "123456789012345678901234567890".parse().unwrap();
//! let b = Z::from_i128(-987_654_321);
//! let c = a * b + a;                          // carry-free, channel by channel
//! assert_eq!(c - a, a * b);
//! ```
//!
//! ## Main types
//!
//! A base is declared with [`rns_base!`], which checks the primes and computes
//! every constant at compile time. [`Rns`] is an exact signed integer with
//! overflow detection, exact sign and comparison, and division. [`RnsMod`] is
//! the plain ring `ℤ/Mℤ`, with no sign and no overflow. [`Extender`] moves
//! integers from one base to another without converting to binary. With the
//! `alloc` feature, `RnsVec` holds many integers in a layout suited to SIMD,
//! and `RnsMatrix` adds exact matrix products and determinants. [`Channel`] is
//! the arithmetic of a single prime, on which everything else is built.
//!
//! See `README.md` for benchmarks and `DESIGN.md` for the mathematics.

#[cfg(feature = "alloc")]
extern crate alloc;

mod base;
#[cfg(feature = "alloc")]
mod batch;
mod exact;
mod extend;
mod limbs;
mod modular;
pub mod primes;
pub mod word;

pub use base::{Base, RnsBase};
#[cfg(feature = "alloc")]
pub use batch::{RnsMatrix, RnsVec};
pub use exact::{OutOfRange, ParseExactError, Rns};
pub use extend::Extender;
pub use modular::{ParseRnsError, RnsMod};
pub use word::Channel;

#[cfg(feature = "alloc")]
#[doc(hidden)]
pub use limbs::__divmod_limbs;
