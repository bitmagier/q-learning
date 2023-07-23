//! Implementation of [DBSCAN](https://de.wikipedia.org/wiki/DBSCAN) clustering algorithm
//! which define clustering as a connected dense region in data space.
//!
//! This density-based clustering algorithm works on a list of values. It requires, that we can calculate a distance between any two elements.
//!
//! _The initial use-case here at the moment is just to have a method for enhancing the log output when it comes to represent a list of results_

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::{Debug, Display, Formatter};
use std::slice::Iter;

use itertools::Itertools;
use num_traits::Num;

pub trait Distance<Rhs = Self>: PartialOrd {
    type Output: PartialOrd + Copy + Debug;
    fn distance(&self, rhs: &Self) -> Self::Output;
}

impl<T: Num + PartialOrd + Copy + Debug> Distance for T {
    type Output = T;

    fn distance(&self, rhs: &Self) -> Self::Output {
        if self >= rhs {
            self.sub(*rhs)
        } else {
            rhs.sub(*self)
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct Clusters(pub Vec<Vec<usize>>);
impl Clusters {
    pub fn contains(&self, element_idx: usize) -> bool {
        self.0.iter().any(|c| c.contains(&element_idx))
    }
}

#[derive(PartialEq, Debug)]
pub struct ClusterAnalysisResult<'a, T>
where
    T: Distance,
{
    pub elements: &'a [T],
    pub clusters: Clusters,
    pub noise: Vec<usize>,
    max_neighbor_distance: <T as Distance>::Output,
    core_point_min_neighbors: usize,
}

impl<'a, T> ClusterAnalysisResult<'a, T>
where
    T: Distance,
{
    pub fn clusters(&self) -> ClusterIterator<T> {
        ClusterIterator {
            analysis_result: self,
            cluster_iter: self.clusters.0.iter(),
        }
    }
}

fn f32_cmp(a: &f32, b: &f32) -> Ordering {
    match (a, b) {
        (a, _) if a.is_nan() => Ordering::Less,
        (_, b) if b.is_nan() => Ordering::Greater,
        (a, b) if a == b => Ordering::Equal,
        (a, b) if a < b => Ordering::Less,
        _ => Ordering::Greater,
    }
}

/// Yx(B..C), ..., Yx(noise)
impl Display for ClusterAnalysisResult<'_, f32> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        fn cluster_range(f: &mut Formatter<'_>, max_neighbor_distance: f32, from: f32, to: f32) -> std::fmt::Result {
            match max_neighbor_distance {
                n if n < 0.00001 => write!(f, "{:.6}..{:.6}", from, to),
                n if n < 0.0001 => write!(f, "{:.5}..{:.5}", from, to),
                n if n < 0.001 => write!(f, "{:.4}..{:.4}", from, to),
                n if n < 0.01 => write!(f, "{:.3}..{:.3}", from, to),
                n if n < 0.1 => write!(f, "{:.2}..{:.2}", from, to),
                _ => write!(f, "{:.1}..{:.1}", from, to),
            }
        }
        
        // cluster
        for (i, c) in self
            .clusters()
            .sorted_by(|a, b| f32_cmp(a.first().unwrap(), b.first().unwrap()))
            .enumerate()
        {
            if i != 0 {
                write!(f, ", ")?;
            }
            let &&from = c.iter().min_by(|&&a,&&b| f32_cmp(a,b)).expect("cluster should not be empty");
            let &&to = c.iter().max_by(|&&a, &&b| f32_cmp(a,b)).unwrap();
            write!(f, "{}x(", c.len())?;
            cluster_range(f, self.max_neighbor_distance, from, to)?;
            write!(f, ")")?;
        }
        f.write_str(", ")?;
        // noise
        write!(f, "{}x(noise)", self.noise.len())
    }
}

pub struct ClusterIterator<'a, T>
where
    T: Distance,
{
    analysis_result: &'a ClusterAnalysisResult<'a, T>,
    cluster_iter: Iter<'a, Vec<usize>>,
}

impl<'a, T> Iterator for ClusterIterator<'a, T>
where
    T: Distance,
{
    type Item = Vec<&'a T>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(c) = self.cluster_iter.next() {
            let result = c.iter().map(|&i| &self.analysis_result.elements[i]).collect_vec();
            Some(result)
        } else {
            None
        }
    }
}

/// Größenordnung
/// 0-10: 1
/// 11-100: 2
/// 101-1000: 3
/// ...
#[allow(dead_code)]
pub fn magnitude(value: usize) -> usize {
    if value == 0 {
        return 1;
    }
    let mut v = value - 1;
    let mut mag = 1;
    loop {
        v = v / 10;
        if v == 0 {
            return mag;
        } else {
            mag += 1
        }
    }
}

/// DBSCAN Cluster analysis algorithm implementation
///
/// ```pseudo
/// GDBSCAN(D, getNeighbors, isCorePoint)
///     C = 0
///     for each unvisited point P in dataset D
///       mark P as visited
///       N = getNeighbors(P)
///       if isCorePoint(P, N)
///          C = next cluster
///          expandCluster(P, N, C)
///       else
///          mark P as NOISE
///
/// expandCluster(P, N, C)
///    add P to cluster C
///    for each point P' in N
///       if P' is not visited
///          mark P' as visited
///          N' = getNeighbors(P')
///          if isCorePoint(P', N')
///             N = N joined with N'
///       if P' is not yet member of any cluster
///          add P' to cluster C
///          unmark P' as NOISE if necessary
/// ```
/// Note: Sure, we could implement an optimized version when we have just a sequence of integers to analyze (e.g. having them sorted at the first step).
/// But having an optimized version is not what we gonna need.
/// Better we stay with the generalized version, which might be of use for other, more complex data types is other use cases later.
pub fn cluster_analysis<T>(
    elements: &[T],
    max_neighbor_distance: <T as Distance>::Output,
    core_point_min_neighbors: usize,
) -> ClusterAnalysisResult<T>
where
    T: Distance,
{
    // `unvisited_idx` is and stays sorted
    let mut unvisited: VecDeque<usize> = (0..elements.len()).collect();
    debug_assert!(unvisited.iter().is_sorted());

    let mut clusters = Clusters(vec![]);
    let mut noise = vec![];

    while let Some(p) = unvisited.pop_front() {
        let neighbors = region_query(elements, p, max_neighbor_distance);
        debug_assert!(neighbors.len() > 0);
        debug_assert!(neighbors.contains(&p));

        // core point?
        if neighbors.len() > core_point_min_neighbors {
            let c = build_cluster(
                elements,
                p,
                neighbors,
                &mut unvisited,
                max_neighbor_distance,
                core_point_min_neighbors,
                &clusters,
                &mut noise,
            );
            clusters.0.push(c);
        } else {
            noise.push(p);
        }
    }

    clusters.0.sort_unstable_by_key(|e| *e.first().unwrap());
    debug_assert!(noise.is_sorted());

    ClusterAnalysisResult {
        elements,
        clusters,
        noise,
        max_neighbor_distance,
        core_point_min_neighbors,
    }
}

/// region query
/// returns indices of all neighbors of p (including p)
fn region_query<T>(elements: &[T], p_idx: usize, max_neighbor_distance: <T as Distance>::Output) -> Vec<usize>
where
    T: Distance,
{
    let p = &elements[p_idx];
    elements
        .iter()
        .enumerate()
        .filter(|(_, e)| p.distance(e) <= max_neighbor_distance)
        .map(|(i, _)| i)
        .collect_vec()
}

/// returns an ordered Vector of cluster elements
/// - `unvisited` is and stays sorted
/// - `noise` is and stays sorted
fn build_cluster<'a, T>(
    elements: &'a [T],
    p: usize,
    neighbors: Vec<usize>,
    unvisited: &mut VecDeque<usize>,
    max_neighbor_distance: <T as Distance>::Output,
    core_point_min_neighbors: usize,
    existing_clusters: &Clusters,
    noise: &mut Vec<usize>,
) -> Vec<usize>
where
    T: Distance,
{
    let mut forming_cluster = vec![p];

    debug_assert!(neighbors.len() > 0);
    let mut neighbors = neighbors;
    let mut neighbors_idx = 0;

    loop {
        let pn = neighbors[neighbors_idx];

        debug_assert!(unvisited.iter().is_sorted());
        if let Ok(i) = unvisited.binary_search(&pn) {
            unvisited.remove(i);
            let nn = region_query(elements, pn, max_neighbor_distance);
            if nn.len() > core_point_min_neighbors {
                append_new(&mut neighbors, nn);
            }
        }

        if !forming_cluster.contains(&pn) && !existing_clusters.contains(pn) {
            forming_cluster.push(pn);

            if let Ok(i) = noise.binary_search(&pn) {
                noise.remove(i);
            }
        }

        neighbors_idx += 1;
        if neighbors_idx >= neighbors.len() {
            break;
        }
    }

    forming_cluster.sort_unstable();
    forming_cluster
}

/// append new elements (seen from `base`) from `new` into `base` (ignore entries, which are already present in base)
fn append_new(base: &mut Vec<usize>, new: Vec<usize>) {
    for e in new {
        if !base.contains(&e) {
            base.push(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0, 1)]
    #[case(1, 1)]
    #[case(9, 1)]
    #[case(10, 1)]
    #[case(11, 2)]
    #[case(100, 2)]
    #[case(101, 3)]
    #[case(110, 3)]
    #[case(999, 3)]
    #[case(1000, 3)]
    #[case(1001, 4)]
    fn test_magnitude(#[case] v: usize, #[case] m: usize) {
        assert_eq!(magnitude(v), m);
    }

    #[rstest]
    #[case(&[1, 2, 3, 5, 10, 12, 20, 21], 2, 2, Clusters(vec![ vec![0,1,2,3]]), vec![4,5,6,7])]
    #[case(&[1, 2, 3, 5, 10, 12, 20, 21], 2, 1, Clusters(vec![ vec![0,1,2,3], vec![4,5], vec![6,7]]), vec![])]
    #[case(&[0.9, 1.2, 1.1, 5.5, 10.1, 10.2, 1.1], 1.0, 1, Clusters(vec![ vec![0, 1, 2, 6], vec![4,5]]), vec![3])]
    #[case(&[0, 0, 1, 2, 3, 6, 5, 0, 778, 780, 783, 1012, 1014, 1018, 1019, 1500], 3, 2, Clusters(vec![vec![0,1,2,3,4,5,6,7], vec![8,9,10]]), vec![11,12,13,14,15])]
    fn test_cluster_analysis<T: Distance + Debug>(
        #[case] elements: &[T],
        #[case] max_neighbor_distance: <T as Distance>::Output,
        #[case] core_point_min_neighbors: usize,
        #[case] expected_clusters_idx: Clusters,
        #[case] expected_noise_idx: Vec<usize>,
    ) {
        let result = cluster_analysis(elements, max_neighbor_distance, core_point_min_neighbors);
        assert_eq!(
            result,
            ClusterAnalysisResult {
                elements,
                clusters: expected_clusters_idx,
                noise: expected_noise_idx.into_iter().collect::<Vec<_>>(),
                max_neighbor_distance,
                core_point_min_neighbors
            }
        );
    }
}
