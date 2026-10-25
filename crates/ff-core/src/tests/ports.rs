//! Tests for module `ports`.

use super::*;

#[test]
fn slice_range_checks_bounds_and_overflow() {
    let all = [1, 2, 3, 4, 5];
    assert_eq!(slice_range(&all, "t", 1, 3).unwrap(), [2, 3, 4]);
    assert_eq!(slice_range(&all, "t", 5, 0).unwrap(), Vec::<u8>::new());
    assert!(slice_range(&all, "t", 3, 3).is_err());
    assert!(slice_range(&all, "t", u64::MAX, 1).is_err());
}
