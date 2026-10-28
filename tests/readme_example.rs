//! The example from README.md, kept here so that it always compiles and runs.

#[test]
#[allow(unused_variables, clippy::identity_op, clippy::erasing_op)]
fn readme_example() {
    use residue::{rns_base, Extender, Rns, RnsMatrix, RnsVec};

    // A base of 4 primes below 2^31 holds integers up to 2^122 in absolute value.
    rns_base!(Narrow: largest 4);
    // 9 primes allow up to 2^277.
    rns_base!(Wide: largest 9);

    let a: Rns<Narrow, 4> = "-123456789012345678901234567890".parse().unwrap();
    let b = Rns::<Narrow, 4>::from_i64(42).unwrap();
    assert!(a < b); // exact comparison, without converting to binary
    let c = a * b; // panics on overflow; checked_mul returns None instead

    // When a result needs more room, move to a wider base without converting.
    let ext = Extender::<Narrow, 4, Wide, 9>::new();
    let big = ext.extend(&a).unwrap() * ext.extend(&a).unwrap();

    // Many integers at once, processed with SIMD.
    let xs = RnsVec::<Wide, 9>::from_i128s(&[1, -2, 3]).unwrap();
    let ys = RnsVec::<Wide, 9>::from_i128s(&[4, 5, -6]).unwrap();
    assert_eq!(xs.dot(&ys).unwrap().to_i128(), Some(1 * 4 - 2 * 5 - 3 * 6));

    // Exact matrices: products and determinants.
    let m = RnsMatrix::<Wide, 9>::from_i128s(2, 2, &[3, 1, 4, 1]).unwrap();
    assert_eq!(m.determinant().unwrap().to_i128(), Some(-1));
    let m2 = &m * &m;

    // Division: exact when the divisor divides, or general with a remainder.
    let x = Rns::<Wide, 9>::from_i64(91).unwrap();
    let y = Rns::<Wide, 9>::from_i64(7).unwrap();
    assert_eq!(x.div_exact(y).unwrap().to_i128(), Some(13));
    assert_eq!((x % Rns::from_i64(10).unwrap()).to_i128(), Some(1));
}
