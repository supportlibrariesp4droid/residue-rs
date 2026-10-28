//! [`RnsMatrix`]: dense matrices of exact integers.
//!
//! Storage is an [`RnsVec`] in row-major order, so each channel is one
//! contiguous `rows × cols` array of residues. Element-wise operations come for
//! free from `RnsVec`; this module adds the matrix product and the determinant.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::ops::Mul;

use super::{ceil_log2, kernels, mul_bits, overflow, RnsVec};
use crate::base::RnsBase;
use crate::exact::{OutOfRange, Rns};
use crate::word::Channel;

/// A dense `rows × cols` matrix of exact integers in RNS, with one certified
/// bound `|a_ij| < 2^bits` for all entries.
pub struct RnsMatrix<B: RnsBase<N>, const N: usize> {
    rows: usize,
    cols: usize,
    data: RnsVec<B, N>,
}

// SIMD-dispatched operations (concrete types: see kernels::Op).

struct GemmOp<'a, const N: usize> {
    ch: &'a [Channel; N],
    m: usize,
    k: usize,
    p: usize,
    a: &'a [Vec<u32>; N],
    b: &'a [Vec<u32>; N],
    a_red: &'a [u64],
    b_red: &'a [u64],
    out: &'a mut [Vec<u32>; N],
    out_red: &'a mut [u64],
}

// Index loops on purpose: everything stays in this `inline(always)` body.
#[allow(clippy::needless_range_loop)]
impl<const N: usize> kernels::Op for GemmOp<'_, N> {
    type Out = ();
    #[inline(always)]
    fn run(self) {
        let (m, k, p) = (self.m, self.k, self.p);
        let mut lo = vec![0u64; p];
        let mut hi = vec![0u64; p];
        for i in 0..N {
            let ch = self.ch[i];
            // The division-free reduction needs every half-sum below p·2^32.
            let fast = (k as u64) < ch.modulus() as u64;
            let (a, b) = (&self.a[i], &self.b[i]);
            let out = &mut self.out[i];
            for r in 0..m {
                kernels::gemm_row_acc(&a[r * k..(r + 1) * k], b, p, &mut lo, &mut hi);
                let o = &mut out[r * p..(r + 1) * p];
                if fast {
                    kernels::gemm_row_reduce(ch, &lo, &hi, o);
                } else {
                    kernels::gemm_row_reduce_slow(ch, &lo, &hi, o);
                }
            }
        }
        for r in 0..m {
            kernels::gemm_row_red(
                &self.a_red[r * k..(r + 1) * k],
                self.b_red,
                p,
                &mut self.out_red[r * p..(r + 1) * p],
            );
        }
    }
}

struct DetOp<'a, const N: usize> {
    ch: &'a [Channel; N],
    n: usize,
    a: &'a [Vec<u32>; N],
}

#[allow(clippy::needless_range_loop)]
impl<const N: usize> kernels::Op for DetOp<'_, N> {
    type Out = [u32; N];
    #[inline(always)]
    fn run(self) -> [u32; N] {
        let mut work = vec![0u32; self.n * self.n];
        let mut det = [0u32; N];
        for i in 0..N {
            work.copy_from_slice(&self.a[i]);
            det[i] = kernels::det_channel(self.ch[i], self.n, &mut work);
        }
        det
    }
}

impl<B: RnsBase<N>, const N: usize> RnsMatrix<B, N> {
    /// Wrap a vector of `rows · cols` entries in row-major order.
    ///
    /// # Panics
    /// If `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: RnsVec<B, N>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "residue: matrix data length mismatch"
        );
        RnsMatrix { rows, cols, data }
    }

    /// Build from `i128` entries in row-major order. `Err` if any is out of range.
    ///
    /// # Panics
    /// If `xs.len() != rows * cols`.
    pub fn from_i128s(rows: usize, cols: usize, xs: &[i128]) -> Result<Self, OutOfRange> {
        assert_eq!(
            xs.len(),
            rows * cols,
            "residue: matrix data length mismatch"
        );
        Ok(Self::from_vec(rows, cols, RnsVec::from_i128s(xs)?))
    }

    /// Build entry by entry.
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> Rns<B, N>) -> Self {
        let mut data = RnsVec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        RnsMatrix { rows, cols, data }
    }

    /// The `rows × cols` zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        RnsMatrix {
            rows,
            cols,
            data: RnsVec::zeros(rows * cols),
        }
    }

    /// The `n × n` identity matrix.
    pub fn identity(n: usize) -> Self {
        Self::from_fn(n, n, |r, c| if r == c { Rns::one() } else { Rns::zero() })
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Entry `(r, c)`, or `None` if out of bounds.
    pub fn get(&self, r: usize, c: usize) -> Option<Rns<B, N>> {
        if r < self.rows && c < self.cols {
            self.data.get(r * self.cols + c)
        } else {
            None
        }
    }

    /// The certified bound shared by all entries: `|a_ij| < 2^bits_bound()`.
    #[inline]
    pub fn bits_bound(&self) -> u32 {
        self.data.bits_bound()
    }

    /// The entries in row-major order.
    #[inline]
    pub fn as_vec(&self) -> &RnsVec<B, N> {
        &self.data
    }

    /// Consume the matrix, returning its entries in row-major order.
    #[inline]
    pub fn into_vec(self) -> RnsVec<B, N> {
        self.data
    }

    /// Replace the bound with the exact maximum bit length over all entries.
    pub fn tighten(&mut self) {
        self.data.tighten();
    }

    /// The transpose.
    pub fn transpose(&self) -> Self {
        Self::from_fn(self.cols, self.rows, |r, c| self.get(c, r).unwrap())
    }

    /// Exact matrix product `self · other`, or `None` if the certified bound
    /// `bits_a + bits_b + ⌈log2 k⌉` (with `k` the inner dimension) exceeds
    /// `max_bits`.
    ///
    /// Each channel is an independent `u32` matrix product with lazy reduction:
    /// raw 64-bit products are accumulated in two halves and every output entry
    /// is reduced once, with three Montgomery reductions and no division (when
    /// `k` is below the smallest modulus; otherwise a slower path is used).
    /// The inner loop runs along rows of `other`, vectorized across columns.
    ///
    /// # Panics
    /// If `self.cols() != other.rows()`.
    pub fn checked_matmul(&self, other: &Self) -> Option<Self> {
        assert_eq!(
            self.cols, other.rows,
            "residue: matrix product dimension mismatch ({}×{} · {}×{})",
            self.rows, self.cols, other.rows, other.cols
        );
        let (m, k, p) = (self.rows, self.cols, other.cols);
        let bits = mul_bits(self.bits_bound(), other.bits_bound());
        let bits = if bits == 0 { 0 } else { bits + ceil_log2(k) };
        if bits > Rns::<B, N>::max_bits() {
            return None;
        }
        let mut out = RnsVec::<B, N>::zeros(m * p);
        let (a, b) = (&self.data, &other.data);
        kernels::dispatch(GemmOp {
            ch: B::base().channels(),
            m,
            k,
            p,
            a: &a.res,
            b: &b.res,
            a_red: &a.red,
            b_red: &b.red,
            out: &mut out.res,
            out_red: &mut out.red,
        });
        out.bits = bits;
        Some(RnsMatrix {
            rows: m,
            cols: p,
            data: out,
        })
    }

    /// A certified bound on the bit length of the determinant, from Hadamard's
    /// inequality `|det A| <= ∏ ||row_i||`. With `n` rows and entries
    /// `|a_ij| < 2^b`: `|det A| < 2^(n·b + ⌈n·⌈log2 n⌉ / 2⌉)`.
    pub fn determinant_bits_bound(&self) -> u32 {
        let (n, b) = (self.rows as u32, self.bits_bound());
        if n == 0 || b == 0 {
            return 0;
        }
        n * b + (n * ceil_log2(self.rows)).div_ceil(2)
    }

    /// The exact determinant, or `None` if the Hadamard bound
    /// ([`RnsMatrix::determinant_bits_bound`]) exceeds the base's `max_bits`:
    /// then use a wider base (see [`crate::Extender`]).
    ///
    /// Multi-modular algorithm: Gaussian elimination is run independently in
    /// each channel, over the field `ℤ/m_iℤ`, each with its own pivots; the
    /// `N` determinants mod `m_i` are then combined by the Chinese Remainder
    /// Theorem. No big-integer arithmetic and no division of big integers
    /// happens during the elimination. Intermediate values never need to "fit":
    /// only the final determinant must lie in range, which Hadamard guarantees.
    ///
    /// Cost: `O(N·n³)` word operations (vectorized across columns) plus one
    /// reconstruction.
    ///
    /// # Panics
    /// If the matrix is not square.
    pub fn determinant(&self) -> Option<Rns<B, N>> {
        assert_eq!(
            self.rows, self.cols,
            "residue: determinant of a non-square matrix"
        );
        let n = self.rows;
        if n == 0 {
            return Some(Rns::one());
        }
        if self.determinant_bits_bound() > Rns::<B, N>::max_bits() {
            return None;
        }
        let det = kernels::dispatch(DetOp {
            ch: B::base().channels(),
            n,
            a: &self.data.res,
        });
        // In range by Hadamard, so the representative is the true determinant.
        Some(Rns::from_mont_residues_signed(det).expect("Hadamard bound guarantees range"))
    }
}

/// Panicking matrix product. See [`RnsMatrix::checked_matmul`].
impl<B: RnsBase<N>, const N: usize> Mul for &RnsMatrix<B, N> {
    type Output = RnsMatrix<B, N>;
    fn mul(self, rhs: Self) -> RnsMatrix<B, N> {
        self.checked_matmul(rhs).unwrap_or_else(|| overflow())
    }
}

impl<B: RnsBase<N>, const N: usize> Clone for RnsMatrix<B, N> {
    fn clone(&self) -> Self {
        RnsMatrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

impl<B: RnsBase<N>, const N: usize> PartialEq for RnsMatrix<B, N> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}
impl<B: RnsBase<N>, const N: usize> Eq for RnsMatrix<B, N> {}

impl<B: RnsBase<N>, const N: usize> fmt::Debug for RnsMatrix<B, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "RnsMatrix({}×{}, |a| < 2^{}) [",
            self.rows,
            self.cols,
            self.bits_bound()
        )?;
        for r in 0..self.rows {
            f.write_str("  [")?;
            for c in 0..self.cols {
                if c > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{}", self.get(r, c).unwrap())?;
            }
            f.write_str("]\n")?;
        }
        f.write_str("]")
    }
}
