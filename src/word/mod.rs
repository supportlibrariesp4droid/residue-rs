//! Single-channel modular arithmetic, on which the rest of the library is built.
//!
//! A [`Channel`] bundles an odd prime modulus `p < 2^31` with the constants
//! needed for fast arithmetic. Addition and subtraction are branchless, because
//! the spare top bit guarantees that `a + b` fits a `u32`. Multiplication uses
//! Montgomery form with radix `R = 2^32`, and multiplication by a fixed constant
//! uses Shoup's method. Neither needs a division at runtime.
//!
//! Residues come in two flavours. *Ordinary* residues are plain values in
//! `[0, p)`. *Montgomery* residues are `a·R mod p`; multiply them with
//! [`Channel::mul_mont`] and convert back with [`Channel::from_mont`]. Add and
//! sub work on either flavour unchanged (Montgomery form is linear).

/// A single RNS channel: an odd prime modulus `p < 2^31` plus precomputed
/// constants for Montgomery arithmetic. Cheap to copy (12 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Channel {
    p: u32,
    /// `-p^{-1} mod 2^32`, used by Montgomery reduction.
    neg_pinv: u32,
    /// `R^2 mod p` with `R = 2^32`, used to enter Montgomery form.
    r2: u32,
}

/// `-p^{-1} mod 2^32` for odd `p`, via Newton-Hensel lifting.
///
/// `x = p` is already the inverse modulo `2^3` (every odd `p` satisfies
/// `p^2 ≡ 1 mod 8`), and each step doubles the number of correct low bits:
/// 3 → 6 → 12 → 24 → 48, so five steps cover all 32 bits.
const fn compute_neg_pinv(p: u32) -> u32 {
    let mut x = p;
    let mut i = 0;
    while i < 5 {
        x = x.wrapping_mul(2u32.wrapping_sub(p.wrapping_mul(x)));
        i += 1;
    }
    x.wrapping_neg()
}

/// `R^2 mod p = 2^64 mod p`, computed in two steps to avoid 128-bit intermediates.
const fn compute_r2(p: u32) -> u32 {
    let r = (1u64 << 32) % p as u64; // R mod p
    ((r * r) % p as u64) as u32 // R^2 mod p
}

impl Channel {
    /// Build a channel for an odd prime modulus `p` with `3 <= p < 2^31`.
    ///
    /// Primality is *not* re-checked here (use [`crate::primes::is_prime`] to
    /// choose `p`); the asserts guard the invariants the arithmetic relies on.
    pub const fn new(p: u32) -> Self {
        assert!(p >= 3, "modulus must be >= 3");
        assert!(p & 1 == 1, "modulus must be odd");
        assert!(p < (1 << 31), "modulus must be < 2^31");
        Channel {
            p,
            neg_pinv: compute_neg_pinv(p),
            r2: compute_r2(p),
        }
    }

    /// The modulus `p`.
    #[inline(always)]
    pub const fn modulus(&self) -> u32 {
        self.p
    }

    /// `(a + b) mod p`, branchless. Requires `a, b < p`.
    ///
    /// `s = a + b < 2^32` because `a, b < 2^31`, so the sum never overflows a
    /// `u32`. If `s >= p` then `s - p < s`; otherwise `s.wrapping_sub(p)` wraps
    /// to something `> s`. Either way the result is `min(s, s - p)`.
    #[inline(always)]
    pub const fn add(&self, a: u32, b: u32) -> u32 {
        let s = a + b;
        let t = s.wrapping_sub(self.p);
        if s < t {
            s
        } else {
            t
        }
    }

    /// `(a - b) mod p`, branchless. Requires `a, b < p`.
    #[inline(always)]
    pub const fn sub(&self, a: u32, b: u32) -> u32 {
        let d = a.wrapping_sub(b); // a-b, or a-b+2^32 on underflow
        let d2 = d.wrapping_add(self.p);
        if d < d2 {
            d
        } else {
            d2
        }
    }

    /// `-a mod p`. Requires `a < p`.
    #[inline(always)]
    pub const fn neg(&self, a: u32) -> u32 {
        if a == 0 {
            0
        } else {
            self.p - a
        }
    }

    /// Montgomery reduction: `REDC(t) = t · R^{-1} mod p`, with `R = 2^32`.
    /// Requires `t < p · 2^32` (always true for products of residues `< p`).
    #[inline(always)]
    pub const fn redc(&self, t: u64) -> u32 {
        let m = (t as u32).wrapping_mul(self.neg_pinv); // (t mod R)·(-p^{-1}) mod R
        let u = ((t + (m as u64) * (self.p as u64)) >> 32) as u32; // < 2p
        if u >= self.p {
            u - self.p
        } else {
            u
        }
    }

    /// Multiply two Montgomery-form residues, result in Montgomery form.
    #[inline(always)]
    pub const fn mul_mont(&self, a: u32, b: u32) -> u32 {
        self.redc((a as u64) * (b as u64))
    }

    /// Convert an ordinary residue in `[0, p)` to Montgomery form `a·R mod p`.
    #[inline(always)]
    pub const fn to_mont(&self, a: u32) -> u32 {
        self.redc((a as u64) * (self.r2 as u64))
    }

    /// Convert a Montgomery-form residue back to an ordinary residue in `[0, p)`.
    #[inline(always)]
    pub const fn from_mont(&self, a: u32) -> u32 {
        self.redc(a as u64)
    }

    /// Precompute the Shoup factor `floor(w · 2^32 / p)` for a fixed ordinary
    /// multiplier `w < p`. Pass the result to [`Channel::mul_shoup`].
    #[inline(always)]
    pub const fn shoup_factor(&self, w: u32) -> u32 {
        (((w as u64) << 32) / self.p as u64) as u32
    }

    /// `(a · w) mod p` for ordinary residues, using the precomputed Shoup factor
    /// `w_shoup = self.shoup_factor(w)`. Requires `a < p` and `w < p`.
    ///
    /// The quotient estimate `q ≈ a·w / p` is at most 1 too small, so a single
    /// conditional subtraction corrects the result.
    #[inline(always)]
    pub const fn mul_shoup(&self, a: u32, w: u32, w_shoup: u32) -> u32 {
        let q = (((a as u64) * (w_shoup as u64)) >> 32) as u32;
        let r = a.wrapping_mul(w).wrapping_sub(q.wrapping_mul(self.p));
        if r >= self.p {
            r - self.p
        } else {
            r
        }
    }

    /// `a^e` for a Montgomery-form `a`; the result is in Montgomery form.
    pub const fn pow_mont(&self, a: u32, mut e: u32) -> u32 {
        let mut acc = self.to_mont(1);
        let mut b = a;
        while e > 0 {
            if e & 1 == 1 {
                acc = self.mul_mont(acc, b);
            }
            b = self.mul_mont(b, b);
            e >>= 1;
        }
        acc
    }

    /// Inverse of a non-zero Montgomery-form residue, in Montgomery form, by
    /// Fermat's little theorem (`a^(p-2)`). Returns 0 for `a = 0`. Requires `p`
    /// prime, which every [`crate::Base`] guarantees.
    #[inline]
    pub const fn inv_mont(&self, a: u32) -> u32 {
        self.pow_mont(a, self.p - 2)
    }
}
