//! Carving caller-owned scratch into named buffers, so heap-free code can take its
//! working memory from one slice.

/// Splits the first `len` values off `buffer` and leaves the rest in it.
///
/// # Panics
///
/// Panics if `buffer` holds fewer than `len` values.
pub(crate) fn take<'a>(buffer: &mut &'a mut [f64], len: usize) -> &'a mut [f64] {
    let (head, tail) = core::mem::take(buffer).split_at_mut(len);
    *buffer = tail;
    head
}
