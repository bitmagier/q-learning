pub mod mechanics;


fn min_f32<I>(iter: I) -> f32
    where I: Iterator<Item = f32>
{
    let r = iter.fold(f32::INFINITY, |a, b| a.min(b));
    assert!(r.is_finite());
    r
}

fn max_f32<I>(iter: I) -> f32
    where I: Iterator<Item = f32>
{
    let r = iter.fold(f32::NEG_INFINITY, |a, b| a.max(b));
    assert!(r.is_finite());
    r
}
