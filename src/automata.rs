use anyhow::{anyhow, Result};
use regex_syntax::hir::{Hir, HirKind, Literal};
use std::collections::{BTreeSet, HashMap, HashSet};
use tracing::{debug, info, instrument, warn};

/// Represents a state identifier in the automaton
pub type State = usize;

/// Contains data associated with each state in the automaton
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StateData {
    /// Maps input characters to sets of destination states
    pub transitions: HashMap<char, BTreeSet<State>>,
    /// Indicates whether this is an accepting/final state
    pub is_final: bool,
}

/// Represents a finite automaton (NFA or DFA) for regular expression matching
///
/// This struct implements both Thompson's Construction Algorithm for converting
/// regular expressions to NFAs and subset construction for converting NFAs to DFAs.
#[derive(Debug, Clone)]
pub struct Automaton {
    /// Maps state IDs to their associated data (transitions and final status)
    pub states: HashMap<State, StateData>,
    // pub final_states: HashSet<State>,
    /// The initial state of the automaton
    pub start: State,
    /// Counter for generating new unique state IDs
    next_state: State,
    pub start_anchor: bool,
}

impl Automaton {
    /// Creates a new empty automaton with a single non-accepting start state
    fn new() -> Self {
        let mut states = HashMap::new();
        states.insert(
            0,
            StateData {
                transitions: HashMap::new(),
                is_final: false,
            },
        );
        Self {
            states,
            start: 0,
            next_state: 1,
            start_anchor: false,
        }
    }

    /// Creates a new automaton with a specified start state
    ///
    /// # Arguments
    /// * `start` - The ID to use for the start state
    fn with_start_state(start: State) -> Self {
        let mut states = HashMap::new();
        states.insert(
            start,
            StateData {
                transitions: HashMap::new(),
                is_final: false,
            },
        );
        Self {
            states,
            start,
            next_state: start + 1,
            start_anchor: false,
        }
    }

    /// Creates a new state in the automaton and returns its ID
    fn add_state(&mut self) -> State {
        let state = self.next_state;
        self.states.insert(
            state,
            StateData {
                transitions: HashMap::new(),
                is_final: false,
            },
        );
        self.next_state += 1;
        state
    }

    /// Adds an epsilon (empty) transition between two states
    ///
    /// # Arguments
    /// * `from` - Source state ID
    /// * `to` - Destination state ID
    fn add_epsilon(&mut self, from: State, to: State) {
        let entry = self
            .states
            .get_mut(&from)
            .unwrap()
            .transitions
            .entry('ε')
            .or_default();
        entry.insert(to);
    }

    /// Adds a character transition between two states
    ///
    /// # Arguments
    /// * `from` - Source state ID
    /// * `ch` - The character that triggers this transition
    /// * `to` - Destination state ID
    fn add_transition(&mut self, from: State, ch: char, to: State) {
        self.states
            .entry(from)
            .or_default()
            .transitions
            .entry(ch)
            .or_default()
            .insert(to);
    }

    /// Converts a regular expression HIR (High-level Intermediate Representation)
    /// to an NFA using Thompson's Construction Algorithm
    ///
    /// # Arguments
    /// * `hir` - The HIR representation of the regular expression
    ///
    /// # Returns
    /// * `Result<Self>` - The resulting NFA or an error if the conversion fails
    pub fn from_thompson_construction(hir: &Hir) -> Result<Self> {
        let mut automaton = Self::new();
        let prev_start = automaton.start;
        let (start, end) = automaton.build_thompson(hir)?;
        automaton.add_epsilon(prev_start, start);
        automaton
            .states
            .get_mut(&end)
            .ok_or_else(|| anyhow!("End state not found"))?
            .is_final = true;

        debug!("e-NFA length: {}", automaton.states.len());
        Ok(automaton)
    }

    /// Recursively builds the NFA structure based on the HIR pattern
    ///
    /// # Arguments
    /// * `self` - The current automaton being constructed
    /// * `hir` - The current HIR node being processed
    ///
    /// # Returns
    /// * `Result<(State, State)>` - The start and end states of the constructed sub-NFA
    #[instrument]
    fn build_thompson(&mut self, hir: &Hir) -> Result<(State, State)> {
        match hir.kind() {
            HirKind::Literal(Literal(chars)) => {
                let mut prev = self.add_state();
                let start = prev;
                for &ch in chars {
                    if ch >= 128 {
                        warn!("Ignoring non-ASCII character: {}", ch);
                    }
                    let next = self.add_state();
                    self.add_transition(prev, ch as char, next);
                    prev = next;
                }
                Ok((start, prev))
            }
            HirKind::Repetition(rep) => {
                let start = self.add_state();
                let mut current = start;

                // Handle minimum required repetitions
                for _ in 0..rep.min {
                    let (sub_start, sub_end) = self.build_thompson(rep.sub.as_ref())?;
                    self.add_epsilon(current, sub_start);
                    current = sub_end;
                }

                // Handle optional repetitions (max - min)
                if let Some(max) = rep.max {
                    let optional_count = max - rep.min;
                    for _ in 0..optional_count {
                        let (sub_start, sub_end) = self.build_thompson(rep.sub.as_ref())?;
                        let opt_start = self.add_state();
                        let opt_end = self.add_state();

                        // Create branching path: match or skip
                        self.add_epsilon(current, opt_start);
                        self.add_epsilon(opt_start, sub_start);
                        self.add_epsilon(sub_end, opt_end);
                        self.add_epsilon(opt_start, opt_end);

                        current = opt_end;
                    }
                } else {
                    // Handle infinite repetitions after minimum
                    let loop_start = self.add_state();
                    let loop_end = self.add_state();
                    let (sub_start, sub_end) = self.build_thompson(rep.sub.as_ref())?;

                    self.add_epsilon(current, loop_start);
                    self.add_epsilon(loop_start, sub_start);
                    self.add_epsilon(sub_end, loop_start);
                    self.add_epsilon(loop_start, loop_end);

                    current = loop_end;
                }

                Ok((start, current))
            }
            HirKind::Concat(concat) => {
                let mut start = None;
                let mut prev_end = None;
                for sub in concat {
                    let (s, e) = self.build_thompson(sub)?;
                    if start.is_none() {
                        start = Some(s);
                    }
                    if let Some(pe) = prev_end {
                        self.add_epsilon(pe, s);
                    }
                    prev_end = Some(e);
                }
                Ok((start.unwrap(), prev_end.unwrap()))
            }
            HirKind::Alternation(alts) => {
                let start = self.add_state();
                let end = self.add_state();
                for sub in alts {
                    let (s, e) = self.build_thompson(sub)?;
                    self.add_epsilon(start, s);
                    self.add_epsilon(e, end);
                }
                Ok((start, end))
            }
            HirKind::Capture(capt) => self.build_thompson(capt.sub.as_ref()),
            HirKind::Class(class) => {
                // class.case_fold_simple();
                let start = self.add_state();
                let end = self.add_state();
                match class {
                    regex_syntax::hir::Class::Unicode(class) => {
                        class
                            .iter()
                            .flat_map(|range| (range.start() as u8)..=(range.end() as u8))
                            .filter(|&ch| ch < 128)
                            .for_each(|ch| {
                                // let before = self.add_state();
                                // let after = self.add_state();
                                self.add_transition(start, ch as char, end);
                                // self.add_epsilon(start, before);
                                // self.add_epsilon(after, end);
                            });
                    }
                    regex_syntax::hir::Class::Bytes(class) => {
                        class
                            .iter()
                            .flat_map(|range| (range.start())..=(range.end()))
                            .filter(|&ch| ch < 128)
                            .for_each(|ch| {
                                // let before = self.add_state();
                                // let after = self.add_state();
                                self.add_transition(start, ch as char, end);
                                // self.add_epsilon(start, before);
                                // self.add_epsilon(after, end);
                            });
                    }
                };
                Ok((start, end))
            }
            HirKind::Empty => {
                let start = self.add_state();
                let end = self.add_state();
                self.add_epsilon(start, end);
                Ok((start, end))
            }
            HirKind::Look(look) => {
                println!("!!!!! got look: {}", look.as_char());
                match *look {
                    regex_syntax::hir::Look::Start
                    | regex_syntax::hir::Look::StartLF
                    | regex_syntax::hir::Look::StartCRLF => {
                        self.start_anchor = true;
                        let start = self.add_state();
                        let end = self.add_state();
                        self.add_epsilon(start, end);
                        Ok((start, end))
                    }
                    regex_syntax::hir::Look::End
                    | regex_syntax::hir::Look::EndLF
                    | regex_syntax::hir::Look::EndCRLF => {
                        let start = self.add_state();
                        let end = self.add_state();
                        self.add_transition(start, 0x17 as char, end);
                        // self.add_epsilon(start, end);
                        Ok((start, end))
                    }
                    _ => Err(anyhow!("HirKind::Look not supported")),
                }
            }
        }
    }

    /// Converts an NFA to a DFA by removing epsilon transitions using the subset construction algorithm
    ///
    /// # Arguments
    /// * `self` - The input NFA to be converted
    ///
    /// # Returns
    /// * A new DFA equivalent to the input NFA but without epsilon transitions
    #[instrument]
    pub fn remove_epsilon(&self) -> Option<Self> {
        let mut dfa = Self::with_start_state(self.start);
        dfa.start_anchor = self.start_anchor;

        // Precompute epsilon closures for all states
        let mut closures = HashMap::new();
        for &state in self.states.keys() {
            closures.insert(state, self.epsilon_closure(state));
        }

        // Build DFA states from NFA state sets
        let mut state_sets = HashMap::new();
        let initial_set = closures[&self.start].clone();
        state_sets.insert(initial_set.clone(), dfa.start);
        dfa.states.get_mut(&dfa.start)?.is_final =
            initial_set.iter().any(|s| self.states[s].is_final);

        let mut queue: Vec<BTreeSet<State>> = vec![initial_set];

        while let Some(current_set) = queue.pop() {
            let current_state = state_sets[&current_set];

            // Find all unique characters in transitions
            let mut chars = HashSet::new();
            for &s in &current_set {
                if let Some(state) = self.states.get(&s) {
                    chars.extend(state.transitions.keys().filter(|c| **c != 'ε'));
                }
            }

            for ch in chars {
                let mut target_set = BTreeSet::new();

                // Collect all reachable states through this character
                for &s in &current_set {
                    if let Some(targets) = self.states[&s].transitions.get(&ch) {
                        for &t in targets {
                            target_set.extend(closures[&t].iter().copied());
                        }
                    }
                }

                if !target_set.is_empty() {
                    let target_state =
                        *state_sets
                            .entry(target_set)
                            .or_insert_with_key(|target_set| {
                                let new_state = dfa.add_state();
                                dfa.states.get_mut(&new_state).unwrap().is_final =
                                    target_set.iter().any(|s| self.states[s].is_final);
                                queue.push(target_set.clone());
                                new_state
                            });
                    dfa.add_transition(current_state, ch, target_state);
                }
            }
        }

        debug!("DFA length: {}", dfa.states.len());

        Some(dfa)
    }

    /// Computes the epsilon closure of a state in an NFA
    ///
    /// The epsilon closure of a state is the set of all states reachable from it
    /// by following only epsilon transitions.
    ///
    /// # Arguments
    /// * `nfa` - The NFA containing the state
    /// * `state` - The state to compute the closure for
    ///
    /// # Returns
    /// * A set of states reachable through epsilon transitions
    pub fn epsilon_closure(&self, state: State) -> BTreeSet<State> {
        let mut closure = BTreeSet::new();
        let mut stack = vec![state];
        closure.insert(state);

        while let Some(s) = stack.pop() {
            if let Some(epsilon_trans) = self.states[&s].transitions.get(&'ε') {
                for &t in epsilon_trans {
                    if closure.insert(t) {
                        stack.push(t);
                    }
                }
            }
        }
        closure
    }

    /// Minimizes a DFA using the Hopcroft algorithm
    ///
    /// # Arguments
    /// * `self` - The input DFA to be minimized
    ///
    /// # Returns
    /// * A new DFA equivalent to the input DFA but with the minimum number of states
    #[instrument]
    pub fn minimize(&self) -> Option<Self> {
        // 1. Initial Partition: Final states vs. Non-final states
        let mut partitions = vec![
            self.states
                .iter()
                .filter(|(_, s)| s.is_final)
                .map(|(id, _)| *id)
                .collect::<BTreeSet<_>>(),
            self.states
                .iter()
                .filter(|(_, s)| !s.is_final)
                .map(|(id, _)| *id)
                .collect::<BTreeSet<_>>(),
        ];

        // 2. Refine partitions until no more splits occur
        loop {
            let mut new_partitions = Vec::new();
            let mut changed = false;

            for part in &partitions {
                let mut split_map: HashMap<Vec<(char, Vec<usize>)>, BTreeSet<State>> =
                    HashMap::new();

                for &state in part {
                    let sig = self.signature(state, &partitions);
                    split_map.entry(sig).or_default().insert(state);
                }

                if split_map.len() > 1 {
                    changed = true;
                    new_partitions.extend(split_map.into_values());
                } else {
                    new_partitions.push(part.clone()); // No split, keep original partition
                }
            }

            partitions = new_partitions;
            if !changed {
                break; // No more partitions were split, minimization complete
            }
        }

        // 3. Construct minimized automaton use index of partition as new state ID
        let mut minimized = Self::new();
        minimized.start_anchor = self.start_anchor;
        let partition_map: HashMap<State, usize> = partitions
            .iter()
            .enumerate()
            .flat_map(|(i, part)| part.iter().map(move |state| (*state, i)))
            .collect();

        minimized.start = *partition_map.get(&self.start)?;

        for (part_index, part) in partitions.iter().enumerate() {
            let representative_state = *part.iter().next()?;
            let is_final = self.states[&representative_state].is_final;

            minimized.states.entry(part_index).or_default().is_final = is_final;

            for (ch, targets) in &self.states[&representative_state].transitions {
                let mut target_partition_indices = BTreeSet::new();
                for target_state in targets {
                    if let Some(target_partition_index) = partition_map.get(target_state) {
                        target_partition_indices.insert(*target_partition_index);
                    }
                }

                for target_partition_index in target_partition_indices {
                    minimized.add_transition(part_index, *ch, target_partition_index);
                }
            }
        }

        debug!("Minimized DFA length: {}", minimized.states.len());

        Some(minimized)
    }

    /// Computes the signature of a state in a DFA
    ///
    /// The signature of a state is a list of transitions sorted by input character,
    /// where each transition is represented by the character and the index of the partition
    /// containing the target state of the transition.
    ///
    /// # Arguments
    /// * `state` - The state to compute the signature format
    /// * `partitions` - The current partitioning of the DFA states
    ///
    /// # Returns
    /// * A list of transitions    
    fn signature(&self, state: State, partitions: &[BTreeSet<State>]) -> Vec<(char, Vec<usize>)> {
        let mut sig = Vec::new();
        for (ch, targets) in &self.states[&state].transitions {
            let mut part_indices = Vec::new();
            for &target in targets {
                let part_index = partitions.iter().position(|p| p.contains(&target)).unwrap();
                part_indices.push(part_index);
            }
            sig.push((*ch, part_indices));
        }
        sig.sort_by_key(|(ch, _)| *ch); // Ensure consistent order
        sig
    }
}
