use crate::automata::Automaton;
use anyhow::{anyhow, Result};
use regex_automata::nfa::thompson::State;
use regex_syntax::hir::Hir;
use std::collections::BTreeSet;
use std::collections::HashSet;
use std::{
    collections::{HashMap, VecDeque},
    future::Future,
};
use tracing::instrument;

// Page size in MacOS
pub const BLOCK_SIZE: usize = usize::pow(2, 14);

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ShiftAndDistAutomaton {
    pub max_dist: usize,
    pub mask_start: u64,
    pub mask_initial: u64,
    pub mask_final: u64,
    pub masks_char: [u64; 256],
    pub masks_dist: [u64; 256],
}

impl ShiftAndDistAutomaton {
    pub fn from_automaton(dfa: &Automaton) -> Result<Self> {
        let mut masks_char: [u64; 256] = [0; 256];
        let mut masks_dist: [u64; 256] = [0; 256];
        // Save incoming edges per state
        let mut incoming_edges = vec![0_u64; dfa.states.len()];
        for (state, data) in &dfa.states {
            for targets in data.transitions.values() {
                for &target in targets {
                    if target != *state {
                        incoming_edges[target] += 1;
                    }
                }
            }
        }

        let mut queue = VecDeque::new();
        let mut graph_edges: Vec<Vec<(char, i32)>> = vec![Vec::new(); dfa.states.len()];

        let mut current_dist: i32 = if dfa.start_anchor { 0 } else { -1 };
        let mut mask_final: u64 = 0;
        let mut mask_initial: u64 = 0;

        queue.push_back(dfa.start);
        incoming_edges[dfa.start] = 0;

        while let Some(state_id) = queue.pop_front() {
            for &(c, d) in &graph_edges[state_id] {
                // Using char c you can come to this state from distance d
                masks_char[c as usize] |=
                    1u64.checked_shl(u32::try_from(current_dist)?).unwrap_or(0);
                // From the state d you can
                if d >= 0 {
                    masks_dist[usize::try_from(current_dist - d)?] |=
                        1u64.checked_shl(u32::try_from(d)?).unwrap_or(0);
                } else {
                    mask_initial |= 1u64.checked_shl(u32::try_from(current_dist)?).unwrap_or(0);
                }
            }

            // Final states
            if dfa.states[&state_id].is_final {
                mask_final |= 1u64.checked_shl(u32::try_from(current_dist)?).unwrap_or(0);
            }

            // Char transitions
            for (&ch, targets) in &dfa.states[&state_id].transitions {
                for &target in targets {
                    if target != state_id {
                        graph_edges[target].push((ch, current_dist));

                        incoming_edges[target] -= 1;
                        if incoming_edges[target] == 0 {
                            queue.push_back(target);
                        }
                    } else if current_dist >= 0 {
                        // Self Loops
                        masks_char[ch as usize] |= 1u64
                            .checked_shl(u32::try_from(current_dist).unwrap_or(0))
                            .unwrap_or(0);
                        masks_dist[0] |= 1u64
                            .checked_shl(u32::try_from(current_dist).unwrap_or(0))
                            .unwrap_or(0);
                    }
                }
            }

            current_dist += 1;

            if current_dist > 64 {
                return Err(anyhow!("Too many states"));
            }
        }

        if incoming_edges.iter().any(|v| *v > 0) {
            for (i, e) in incoming_edges.iter().enumerate() {
                if *e > 0 {
                    println!("Incoming edges: {i} {e}");
                }
            }
            // println!("start {}", &dfa.start);
            // for (state, edges) in &dfa.states {
            //     println!("State: {state}");
            //     for (c, goal) in &edges.transitions {
            //         println!("  char: {c} -> {goal:?}");
            //     }
            // }

            return Err(anyhow!(
                "didn't visit all edges, graph probably has a cycle bigger than 1"
            ));
        }

        // Find max dist
        let mut max_dist = 0;
        for (i, dist) in masks_dist.iter().enumerate() {
            if *dist != 0 {
                max_dist = i;
            }
        }

        // Set start mask to one if we have an anchor
        let mask_start: u64 = u64::from(dfa.start_anchor);

        Ok(Self {
            max_dist,
            mask_start,
            mask_initial,
            mask_final,
            masks_char,
            masks_dist,
        })
    }

    pub fn from_nfa(nfa: &Automaton) -> Result<Self> {
        let mut masks_char: [u64; 256] = [0; 256];
        let mut masks_dist: [u64; 256] = [0; 256];

        // Precompute epsilon closures for all states
        let mut closures = HashMap::new();
        for &state in nfa.states.keys() {
            closures.insert(state, nfa.epsilon_closure(state));
        }

        // Build state sets using subset construction
        let mut state_sets = HashMap::new();
        let mut id_to_set = Vec::new();

        // Start with epsilon closure of the start state
        let initial_set = closures[&nfa.start].clone();
        state_sets.insert(initial_set.clone(), 0);
        id_to_set.push(initial_set);

        // Track incoming edges for topological sort
        let mut incoming_edges = vec![0]; // Start state has 0 incoming edges

        // BFS traversal to discover all state sets
        let mut queue = VecDeque::from([0]); // Start with initial state set
        let mut graph_edges = vec![Vec::new()]; // Edge list for each state set

        let mut current_dist: usize = 0;
        let mut mask_final: u64 = 0;

        // If we have a start anchor, we need to shift the masks
        let right_shift = usize::from(!nfa.start_anchor);

        // Check if initial state set contains any final states
        if id_to_set[0].iter().any(|&s| nfa.states[&s].is_final) {
            mask_final |= 1 << (current_dist - right_shift);
        }

        while let Some(current_id) = queue.pop_front() {
            let current_set = id_to_set[current_id].clone();
            // let current_set = &id_to_set[current_id];

            for (c, d) in &graph_edges[current_id] {
                // Using char c you can come to this state from distance d
                masks_char[*c as usize] |= (1 << current_dist) >> right_shift;
                // From the state d you can
                masks_dist[current_dist - d] |= (1 << d) >> right_shift;
            }

            // Find all outgoing transitions by character
            let mut transitions: HashMap<char, BTreeSet<usize>> = HashMap::new();

            for &state in &current_set {
                for (&ch, targets) in &nfa.states[&state].transitions {
                    if ch != 'ε' {
                        // We've already handled epsilon transitions via closures
                        let entry = transitions.entry(ch).or_default();
                        for &target in targets {
                            // Add target and its epsilon closure
                            entry.extend(closures[&target].iter().copied());
                        }
                    }
                }
            }

            // Process transitions to create new state sets
            for (ch, target_set) in transitions {
                if target_set.is_empty() {
                    continue;
                }

                // Check if this is a self-loop
                if target_set == current_set {
                    // Self loop
                    masks_char[ch as usize] |= 1 << (current_dist - right_shift);
                    masks_dist[0] |= (1 << current_dist) >> right_shift;
                    continue;
                }

                // Get or create state ID for target set
                let target_id = match state_sets.get(&target_set) {
                    Some(&id) => id,
                    None => {
                        // New state set
                        let new_id = state_sets.len();
                        state_sets.insert(target_set.clone(), new_id);
                        id_to_set.push(target_set.clone());

                        // Initialize tracking for this new state
                        incoming_edges.push(1); // Start with one incoming edge
                        graph_edges.push(Vec::new());

                        // Check if this new state set contains any final states
                        if target_set.iter().any(|&s| nfa.states[&s].is_final) {
                            // We'll update the mask_final later when we process this state
                        }

                        new_id
                    }
                };

                // Add edge information
                graph_edges[target_id].push((ch, current_dist));

                // Update incoming edge count and potentially queue for processing
                incoming_edges[target_id] -= 1;
                if incoming_edges[target_id] == 0 {
                    queue.push_back(target_id);
                }
            }

            current_dist += 1;

            if current_dist > 64 {
                return Err(anyhow!("Too many states (> 64)"));
            }
        }

        // Check for unreachable states or cycles
        if incoming_edges.iter().any(|v| *v > 0) {
            return Err(anyhow!(
                "Graph has unreachable states or cycles larger than 1"
            ));
        }

        // Update final mask for all state sets
        for (state_id, state_set) in id_to_set.iter().enumerate() {
            if state_set.iter().any(|&s| nfa.states[&s].is_final) {
                mask_final |= 1 << (state_id - right_shift);
            }
        }

        // Find max dist
        let mut max_dist = 0;
        for (i, dist) in masks_dist.iter().enumerate() {
            if *dist != 0 {
                max_dist = i;
            }
        }

        // Set start mask to one if we have an anchor
        let mask_start: u64 = u64::from(nfa.start_anchor);

        Ok(Self {
            max_dist,
            mask_initial: 1u64,
            mask_start,
            mask_final,
            masks_char,
            masks_dist,
        })
    }
}

// --------------------------
// Bitmap Compilation
// --------------------------
#[allow(dead_code)]
pub trait ShiftAndDist {
    #[instrument]
    fn compile_hir(hir: &Hir) -> Result<ShiftAndDistAutomaton> {
        let nfa = Automaton::from_thompson_construction(hir)?;
        let dfa = nfa
            .remove_epsilon()
            .ok_or_else(|| anyhow!("Failed to remove epsilon"))?;
        // let follow = dfa
        //     .minimize()
        //     .ok_or_else(|| anyhow!("Failed to minimize"))?;
        ShiftAndDistAutomaton::from_automaton(&dfa)
        // ShiftAndDistAutomaton::from_nfa(&nfa)
    }

    fn count_matches(&mut self, text: &str) -> usize;

    fn count_match_lines(&self, _text: &[[u8; BLOCK_SIZE]]) -> Result<usize> {
        Ok(0)
    }

    fn match_lines_future(
        &self,
        _text: &[[u8; BLOCK_SIZE]],
    ) -> Result<impl Future<Output = Vec<bool>>> {
        Ok(async { vec![] })
    }

    fn count_match_lines_future(
        &self,
        text: &[[u8; BLOCK_SIZE]],
    ) -> Result<impl Future<Output = usize>> {
        // const CHUNKS: usize = 4;

        // Calculate chunk size
        // let chunk_size = (text.len() + CHUNKS - 1) / CHUNKS; // Ceiling division

        // Collect futures for each chunk
        // let mut futures = Vec::with_capacity(CHUNKS);

        // for chunk in text.chunks(chunk_size) {
        //     if let Ok(future) = Self::match_lines_future(self, chunk) {
        //         futures.push(future);
        //     }
        // }

        // Return a future that awaits all chunk futures and combines the results
        // Ok(async move {
        //     let mut total_count = 0;

        //     // Process all futures concurrently
        //     for future in futures {
        //         let result = future.await;
        //         total_count += result.into_iter().map(usize::from).sum::<usize>();
        //     }

        //     total_count
        // })

        Self::match_lines_future(self, text).map(|f| async {
            let res = f.await;
            res.into_iter().map(usize::from).sum()
        })
    }
}

// --------------------------
// Tests
// --------------------------
#[cfg(test)]
mod tests {
    use super::*;

    struct TestImpl;
    impl ShiftAndDist for TestImpl {
        fn count_matches(&mut self, _: &str) -> usize {
            0
        }
    }

    #[test]
    fn test_loop_error() {
        let pattern = "a(ab)*c";
        let hir = regex_syntax::parse(pattern).unwrap();
        let automaton = TestImpl::compile_hir(&hir);
        assert!(automaton.is_err());
    }

    #[test]
    fn test_basic() {
        let pattern = "ab{0,2}c";
        let hir = regex_syntax::parse(pattern).unwrap();
        let automaton = TestImpl::compile_hir(&hir).unwrap();
        let (mask_final, masks_char, masks_dist, max_dist) = (
            automaton.mask_final,
            automaton.masks_char,
            automaton.masks_dist,
            automaton.max_dist,
        );

        // println!("Final: {:04b}", mask_final);
        // for (ch, mask) in masks_char.iter().enumerate() {
        //     if *mask != 0 {
        //         println!("Char: {:?} {:04b}", ch as u8 as char, mask);
        //     }
        // }
        // for (dist, mask) in masks_dist.iter().enumerate() {
        //     if *mask != 0 {
        //         println!("Dist: {:?} {:04b}", dist, mask);
        //     }b
        // }

        assert_eq!(max_dist, 3);
        assert_eq!(mask_final, 0b1000);
        // This masks says from which states you are allowed to move to this character
        assert_eq!(masks_char[b'a' as usize], 1 << 0); // come from state -1 (_)
        assert_eq!(
            masks_char[b'b' as usize],
            1 << 1 | 1 << 2,
            "{:04b} {:04b}",
            masks_char[b'b' as usize],
            1 << 1 | 1 << 2,
        ); // come from state 0 (a), 1 (b)
        assert_eq!(masks_char[b'c' as usize], 1 << 3); // come from state 0 (a), 1 (b), 2 (b)
                                                       // This masks says from which states you are allowed to move to this distance
        assert_eq!(
            masks_dist[0], 0b0000,
            "{:04b} {:04b}",
            masks_dist[0], 0b0000
        ); // Self-loops for states (none here)
        assert_eq!(
            masks_dist[1], 0b0111,
            "{:04b} {:04b}",
            masks_dist[1], 0b0111
        ); // From state 0, 1, 2, you are allowed to move 1
        assert_eq!(
            masks_dist[2], 0b0010,
            "{:04b} {:04b}",
            masks_dist[2], 0b0010
        ); // From state 1, you are allowed to move 2
        assert_eq!(
            masks_dist[3], 0b0001,
            "{:04b} {:04b}",
            masks_dist[3], 0b0001
        ); // From state 0, you are allowed to move 3
    }
}
