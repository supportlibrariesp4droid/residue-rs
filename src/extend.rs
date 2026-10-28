//! Exact base extension: move an [`Rns`] to another base without leaving RNS.
//!
//! For any target modulus `q`,
//! `x mod q = (Σ ξ_i·(M_i mod q) − α'·(M mod q)) mod q`,
//! which is exact because `α'` is exact (see [`crate::exact`]). With the table
//! `M_i mod q_j` precomputed once, each extension costs `O(N_s · N_t)` word
//! operations: no reconstruction, no big-integer arithmetic, no sequential chain.

use core::marker::PhantomData;

use crate::base::RnsBase;
use crate::exact::Rns;
use crate::limbs;

/// Precomputed tables to convert exact integers from base `S` (`NS` channels)
/// to base `T` (`NT` channels). Build once, reuse for many values.
///
/// The target may be larger (widening, to make room for bigger results),
/// smaller (narrowing), or simply different. Moduli may overlap.
///
/// ```
/// use residue::{rns_base, Extender, Rns};
///
/// rns_base!(Narrow: largest 4);   // ~122-bit integers
/// rns_base!(Wide: largest 9);     // ~277-bit integers
///
/// let ext = Extender::<Narrow, 4, Wide, 9>::new();
/// let x: Rns<Narrow, 4> = "-123456789012345678901234567890".parse().unwrap();
/// let y: Rns<Wide, 9> = ext.extend(&x).unwrap();
/// assert_eq!(x.to_string(), y.to_string());
/// ```
pub struct Extender<S: RnsBase<NS>, const NS: usize, T: RnsBase<NT>, const NT: usize> {
    /// `mi_mod_q[i][j] = M_i mod q_j`.
    mi_mod_q: [[u32; NT]; NS],
    /// `M mod q_j`.
    m_mod_q: [u32; NT],
    _bases: PhantomData<(S, T)>,
}

impl<S: RnsBase<NS>, const NS: usize, T: RnsBase<NT>, const NT: usize> Extender<S, NS, T, NT> {
    /// Precompute the extension tables: `O(NS² · NT)` word operations, once.
    pub fn new() -> Self {
        let src = S::base();
        let dst = T::base().channels();
        let mi_mod_q = core::array::from_fn(|i| {
            core::array::from_fn(|j| limbs::mod_small(&src.crt_mi()[i], dst[j].modulus()))
        });
        let m_mod_q =
            core::array::from_fn(|j| limbs::mod_small(src.modulus_limbs(), dst[j].modulus()));
        Extender {
            mi_mod_q,
            m_mod_q,
            _bases: PhantomData,
        }
    }

    /// Convert `x` to the target base. `None` if `x`'s certified bound exceeds
    /// the target's `max_bits` (only possible when narrowing).
    pub fn extend(&self, x: &Rns<S, NS>) -> Option<Rns<T, NT>> {
        if x.bits_bound() > T::base().max_bits() {
            return None;
        }
        let (xi, alpha) = x.crt_parts();
        let dst = T::base().channels();
        let r = core::array::from_fn(|j| {
            let q = dst[j].modulus() as u128;
            let acc: u128 = xi
                .iter()
                .zip(&self.mi_mod_q)
                .map(|(&x, row)| x as u128 * row[j] as u128)
                .sum();
            let pos = (acc % q) as u64;
            let neg = (alpha as u128 * self.m_mod_q[j] as u128 % q) as u64;
            let v = ((pos + q as u64 - neg) % q as u64) as u32;
            dst[j].to_mont(v)
        });
        Some(Rns::from_parts_unchecked(r, x.redundant(), x.bits_bound()))
    }
}

impl<S: RnsBase<NS>, const NS: usize, T: RnsBase<NT>, const NT: usize> Default
    for Extender<S, NS, T, NT>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B: RnsBase<N>, const N: usize> Rns<B, N> {
    /// Convert to another base in one call. Convenience wrapper that builds an
    /// [`Extender`] each time; for repeated conversions build it once.
    pub fn rebase<T: RnsBase<NT>, const NT: usize>(&self) -> Option<Rns<T, NT>> {
        Extender::<B, N, T, NT>::new().extend(self)
    }
}
