use std::{sync::Arc, thread};

use crossbeam_channel::unbounded;
use regex_syntax::parse;
use tracing::instrument;

use anyhow::Result;

use crate::shift_and_dist::{ShiftAndDist, BLOCK_SIZE};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ShiftAndDistCPU {
    max_dist: usize,
    mask_initial: u64,
    mask_final: u64,
    masks_char: [u64; 256],
    masks_dist: [u64; 256],
}

#[allow(dead_code)]
impl ShiftAndDistCPU {
    #[instrument]
    pub fn new(pattern: &str) -> Self {
        // let tokens = parse_pattern(pattern).unwrap();
        let tokens = parse(pattern).unwrap();
        // println!("Parsed pattern {tokens}:");
        // print_pattern_tree(&tokens);

        let automaton = Self::compile_hir(&tokens).unwrap();

        // println!("{mask_initial:08b}");
        // println!("{mask_final:08b}");

        // println!("a {:08b}", masks_char[b'a' as usize]);
        // println!("b {:08b}", masks_char[b'b' as usize]);
        // println!("c {:08b}", masks_char[b'c' as usize]);

        // println!("1 {:08b}", masks_dist[0]);
        // println!("2 {:08b}", masks_dist[1]);
        // println!("3 {:08b}", masks_dist[2]);
        // println!("4 {:08b}", masks_dist[3]);

        Self {
            max_dist: automaton.max_dist,
            mask_initial: automaton.mask_initial,
            mask_final: automaton.mask_final,
            masks_char: automaton.masks_char,
            masks_dist: automaton.masks_dist,
        }
    }

    fn count_line_matches(&self, line: &[u8]) -> usize {
        let mut states = 0u64;
        let mut matches = 0;

        // debug!("Mask_final: {:08b}", self.mask_final);

        for ch in line {
            // debug!("Char: {} Current state: {states:08b}", *ch as char);
            let mut next = states & self.masks_dist[0];
            // debug!("Next: {next:08b}");
            for d in 1..=self.max_dist {
                next |= (states & self.masks_dist[d]) << d;
                // debug!("Next: {next:08b}");
            }
            let mask = self.masks_char[*ch as usize];
            // debug!("Mask: {mask:08b}");
            states = (next | self.mask_initial) & mask;

            if states & self.mask_final != 0 {
                // debug!("Match at char : {}", *ch as char);
                matches += 1;
            }
        }
        matches
    }
}

impl ShiftAndDist for ShiftAndDistCPU {
    #[instrument]
    fn count_matches(&mut self, text: &str) -> usize {
        self.count_line_matches(text.as_bytes())
    }

    #[instrument]
    fn count_match_lines(&self, text: &[[u8; BLOCK_SIZE]]) -> Result<usize> {
        let num_threads = 16;
        let (result_sender, result_receiver) = unbounded();

        let self_arc = Arc::new(*self);

        thread::scope(|s| {
            for chunk in text.chunks(text.len() / num_threads) {
                let sender = result_sender.clone();
                let matcher = self_arc.clone();
                s.spawn(move || {
                    let mut matches = 0;
                    for line in chunk {
                        matches += matcher.count_line_matches(line);
                    }
                    sender.send(matches).unwrap();
                });
            }
        });
        drop(result_sender);

        let mut matches = 0;
        while let Ok(match_count) = result_receiver.recv() {
            matches += match_count;
        }

        Ok(matches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_count_matches(pattern: &str, text: &str, expected: usize) {
        let mut sa = ShiftAndDistCPU::new(pattern);
        let matches = sa.count_matches(text);
        assert_eq!(matches, expected, "Pattern: {pattern}, Text: {text}");
    }

    // #[ignore]
    #[test]
    fn test_count_matches_simple() {
        test_count_matches("ab{1,2}c", "ac", 0);
        test_count_matches("ab{1,2}c", "abc", 1);
        test_count_matches("ab{0,2}c", "abbc", 1);
        test_count_matches("ab{0,2}c", "abbbc", 0);
        test_count_matches("ab{0,2}c", "abbbbc", 0);
        test_count_matches("ab{0,2}c", "abbcabc", 2);
        test_count_matches("ab{0,2}c", "acac", 2);
        test_count_matches("ab{1,2}c", "abcac", 1);
    }

    #[test]
    fn test_alternation() {
        let mut sa = ShiftAndDistCPU::new("a(bc|de)h");
        assert_eq!(sa.count_matches("abch"), 1, "abch");
        assert_eq!(sa.count_matches("adeh"), 1, "adeh");
        let mut sa = ShiftAndDistCPU::new("a(b|c|d)h");
        assert_eq!(sa.count_matches("abh"), 1, "abh");
        assert_eq!(sa.count_matches("ach"), 1, "ach");
        assert_eq!(sa.count_matches("adh"), 1, "adh");
        // test_count_matches("a(bc*|d|)h", "abch", 1);
        // test_count_matches("a(bc*|d|)h", "abcccch", 1);
        // test_count_matches("a(bc*|d|)h", "adh", 1);
        test_count_matches("a(bc*|d|)h", "ah", 1);
        // test_count_matches("a(bc+|d|)h", "abh", 0);
        // test_count_matches("a(bc|de|fg|)h", "abch", 1);
        // test_count_matches("a(bc|de|fg|)h", "ah", 1);
        // test_count_matches("a(bc|de|fg|)h", "adeh", 1);
        // test_count_matches("a(bc|de|fg|)h", "afgbh", 0);

        test_count_matches("a[b-g]z", "abz", 1);
        test_count_matches("a[b-g]z", "acz", 1);
        test_count_matches("a[b-g]z", "apz", 0);

        // test_count_matches("a(b|d|c)h", "abbh", 0);
        test_count_matches("a(bc|de|fg|)h", "abch", 1);
        // test_count_matches("a|b", "b", 1);
        // test_count_matches("a|b", "c", 0);
        // test_count_matches("a|b", "ab", 1);
        // test_count_matches("a|b", "ba", 1);
        // test_count_matches("a|b", "abab", 2);
        // test_count_matches("a|b", "ababab", 3);
        // test_count_matches("a|bc", "abc", 1);
        // test_count_matches("a|bc", "bc", 1);
        // test_count_matches("a|bc", "a", 1);
        // test_count_matches("a|bc", "b", 0);
    }

    // #[ignore]
    #[test]
    fn test_count_matches_repeating() {
        test_count_matches("ab{0,2}a", "abaa", 2);
        // abaa aaa aaba abaa
        test_count_matches("ab{0,2}ab{0,2}a", "abaaabaa", 4);
    }
}
