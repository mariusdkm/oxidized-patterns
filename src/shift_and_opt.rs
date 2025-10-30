use anyhow::Result;
use regex_syntax::hir::Hir;
use std::collections::{BTreeSet, HashMap, VecDeque};
use tracing::instrument;

use crate::automata::{Automaton, State};

const M: usize = 128;
const N: usize = 128;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ShiftAndOpsAutomaton {
    mask_initial: u64,
    mask_final: u64,
    masks_shift: [u64; M],
    masks_dist: [u64; M],
    masks_src: [u64; N],
    masks_dst: [u64; N],
    masks_char: [u64; N],
    max_dist: u64,
}

// --------------------------
// Bitmap Compiler for ShiftAndOps
// --------------------------
#[derive(Debug)]
pub struct BitmapCompiler {
    max_shift_distance: u32,
}

impl BitmapCompiler {
    pub const fn new(max_shift_distance: u32) -> Self {
        Self { max_shift_distance }
    }

    pub fn compile(&self, dfa: &Automaton) -> ShiftAndOpsAutomaton {
        // Assign state positions via BFS
        let mut positions = HashMap::new();
        let mut queue = VecDeque::new();
        let start = dfa.start;
        queue.push_back(start);
        positions.insert(start, 0);
        let mut current_pos = 1;

        let mut incoming_transitions: Vec<BTreeSet<State>> =
            vec![BTreeSet::new(); dfa.states.len()];

        while let Some(state) = queue.pop_front() {
            if let Some(state_data) = dfa.states.get(&state) {
                for targets in state_data.transitions.values() {
                    for target in targets {
                        incoming_transitions[*target].insert(state);
                        if !positions.contains_key(target) {
                            positions.insert(*target, current_pos);
                            current_pos += 1;
                            queue.push_back(*target);
                        }
                    }
                }
            }
        }

        // Compute masks_char
        let mut masks_char = [0; 128];
        for (s, state_data) in &dfa.states {
            let s_pos = positions[s];
            for ch in state_data.transitions.keys() {
                if ch.is_ascii() {
                    let c = *ch as usize;
                    masks_char[c] |= 1 << s_pos;
                }
            }
        }

        // Compute shift_ops and multi_edge_transitions
        let mut shift_ops: HashMap<u64, u64> = HashMap::new();
        let mut multi_edge_transitions = Vec::new();

        for (s, state_data) in &dfa.states {
            let s_pos = positions[s];
            for targets in state_data.transitions.values() {
                for t in targets {
                    let t_pos = positions[t];
                    let distance = t_pos - s_pos;
                    if distance >= 0 && (distance as u32) <= self.max_shift_distance {
                        let entry = shift_ops.entry(distance as u64).or_insert(0);
                        *entry |= 1 << s_pos;
                    } else {
                        multi_edge_transitions.push((*s, *t));
                    }
                }
            }
        }

        // Convert shift_ops to masks_shift and shift_dist
        let mut masks_shift: [u64; M] = [0; M];
        let mut shift_dist: [u64; M] = [0; M];
        let mut sorted_distances: Vec<_> = shift_ops.keys().copied().collect();
        sorted_distances.sort_unstable();
        for d in sorted_distances {
            masks_shift[d as usize] = shift_ops[&d];
            shift_dist[d as usize] = shift_ops[&d];
        }

        // Group multi_edge_transitions into one_to_many and many_to_one
        let mut one_to_many: HashMap<State, Vec<State>> = HashMap::new();
        let mut many_to_one: HashMap<State, Vec<State>> = HashMap::new();

        for (s, t) in multi_edge_transitions {
            one_to_many.entry(s).or_default().push(t);
            many_to_one.entry(t).or_default().push(s);
        }

        // Generate masks_src and masks_dst for one_to_many (source has multiple destinations)
        let mut masks_src: [u64; N] = [0; N];
        let mut masks_dst: [u64; N] = [0; N];

        for (s, ts) in one_to_many {
            if ts.len() >= 2 {
                let src_mask = 1 << positions[&s];
                let dst_mask = ts.iter().map(|t| 1 << positions[t]).fold(0, |a, b| a | b);
                masks_src[ts.len()] = src_mask;
                masks_dst[ts.len()] = dst_mask;
            }
        }

        // Generate masks_src and masks_dst for many_to_one (multiple sources to one destination)
        for (t, ss) in many_to_one {
            if ss.len() >= 2 {
                let src_mask = ss.iter().map(|s| 1 << positions[s]).fold(0, |a, b| a | b);
                let dst_mask = 1 << positions[&t];
                masks_src[ss.len()] = src_mask;
                masks_dst[ss.len()] = dst_mask;
            }
        }

        // Compute mask_final
        let mask_final = dfa
            .states
            .iter()
            .filter(|(_, state)| state.is_final)
            .map(|(s, _)| 1 << positions[s])
            .fold(0, |a, b| a | b);

        ShiftAndOpsAutomaton {
            mask_initial: 1 << positions[&dfa.start],
            mask_final,
            masks_shift,
            masks_dist: shift_dist,
            masks_src,
            masks_dst,
            masks_char,
            max_dist: u64::from(self.max_shift_distance),
        }
    }
}

#[allow(dead_code)]
pub trait ShiftAndOps {
    #[instrument]
    fn compile_hir(hir: &Hir) -> Result<ShiftAndOpsAutomaton> {
        let auto = Automaton::from_thompson_construction(hir)?;
        let dfa = auto
            .remove_epsilon()
            .ok_or_else(|| anyhow::anyhow!("Failed to remove epsilon"))?;

        println!("DFA");
        for (s, state_data) in &dfa.states {
            println!("State {s}");
            for (ch, targets) in &state_data.transitions {
                println!("  {ch} -> {targets:?}");
            }
        }
        // minimize(&mut dfa);
        let compiler = BitmapCompiler::new(10);
        Ok(compiler.compile(&dfa))
    }

    fn count_matches(&mut self, text: &str) -> usize;
}

#[derive(Debug)]
struct CPUImpl {
    automaton: ShiftAndOpsAutomaton,
}

#[allow(dead_code)]
impl CPUImpl {
    fn new(pattern: &str) -> Self {
        let hir = regex_syntax::parse(pattern).unwrap();
        let automaton = Self::compile_hir(&hir).unwrap();
        Self { automaton }
    }
}

impl ShiftAndOps for CPUImpl {
    fn count_matches(&mut self, text: &str) -> usize {
        let auto = self.automaton;
        let mut states = auto.mask_initial;
        let mut matches = 0;

        for ch in text.bytes() {
            // Compute next states through shift operations
            let mut next = 0;

            // Apply shifts with different distances
            for i in 0..M {
                let mask = auto.masks_shift[i];
                let d = auto.masks_dist[i];
                next |= (states & mask) << d;
            }

            // Multi-edge transitions
            for i in 0..N {
                if (states & auto.masks_src[i]) != 0 {
                    next |= auto.masks_dst[i];
                }
            }

            // Character-based state transition
            let mask = auto.masks_char[ch as usize];
            states = (next | auto.mask_initial) & mask;

            // Count matches
            if states & auto.mask_final != 0 {
                matches += 1;
            }
        }

        matches
    }
}

// --------------------------
// Tests
// --------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use regex_syntax::parse;

    #[ignore]
    #[test]
    fn test_cpu_impl() {
        let pattern = "a(b|d)h";
        let mut cpu = CPUImpl::new(pattern);

        let auto = cpu.automaton;

        println!("{:06b}", auto.mask_initial);
        println!("{:06b}", auto.mask_final);

        println!("masks_char");
        for (ch, mask) in auto.masks_char.iter().enumerate() {
            if *mask != 0 {
                println!("{:?} {:06b}", ch as u8 as char, mask);
            }
        }

        println!("masks_shift");
        for (d, mask) in auto.masks_shift.iter().enumerate() {
            if *mask != 0 {
                println!("{d:?} {mask:06b}");
            }
        }

        println!("masks_dist");
        for (d, mask) in auto.masks_dist.iter().enumerate() {
            if *mask != 0 {
                println!("{d:?} {mask:06b}");
            }
        }

        println!("masks_src");
        for (i, mask) in auto.masks_src.iter().enumerate() {
            if *mask != 0 {
                println!("{i:?} {mask:06b}");
            }
        }

        println!("masks_dst");
        for (i, mask) in auto.masks_dst.iter().enumerate() {
            if *mask != 0 {
                println!("{i:?} {mask:06b}");
            }
        }

        let text = "abc";
        let matches = cpu.count_matches(text);
        assert_eq!(matches, 1);
    }

    #[ignore]
    #[test]
    fn test_basic_shift_and_ops() {
        let pattern = "a(bc|de|fg)h";
        let hir = parse(pattern).unwrap();
        let auto = CPUImpl::compile_hir(&hir).unwrap();
        let (mask_final, masks_char, _masks_shift, _shift_dist, masks_src, masks_dst) = (
            auto.mask_final,
            auto.masks_char,
            auto.masks_shift,
            auto.masks_dist,
            auto.masks_src,
            auto.masks_dst,
        );

        // Verify mask_final has the correct final state(s)
        // Assuming 'h' is the final state, its position should be set
        assert!(mask_final > 0);

        // Verify masks_char for 'a' has the correct state
        let a_pos = masks_char[b'a' as usize].trailing_zeros();
        assert!(a_pos < 64);

        // Check for one-to-many transition from 'a' state to multiple destinations
        let mut found_one_to_many = false;
        for (i, src) in masks_src.iter().enumerate() {
            if *src == (1 << a_pos) {
                let dst = masks_dst[i];
                assert!(dst.count_ones() >= 2);
                found_one_to_many = true;
            }
        }
        assert!(found_one_to_many);

        // Check for many-to-one transition to 'h' state
        let h_pos = mask_final.trailing_zeros();
        let mut found_many_to_one = false;
        for (i, dst) in masks_dst.iter().enumerate() {
            if *dst == (1 << h_pos) {
                let src = masks_src[i];
                assert!(src.count_ones() >= 2);
                found_many_to_one = true;
            }
        }
        assert!(found_many_to_one);
    }
}
