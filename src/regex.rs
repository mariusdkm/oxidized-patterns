use crossbeam_channel::unbounded;
use regex::Regex;
use std::fs::File;
use std::hint::black_box;
use std::io::{self, BufRead, BufReader};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use tracing::{info, info_span, instrument, warn};

pub fn count_matches_regex_bytes(
    regex: &regex::bytes::Regex,
    lines: &[String],
    num_threads: usize,
) -> io::Result<usize> {
    // Create channels for distributing work and collecting results
    let (result_sender, result_receiver) = {
        let span = info_span!("channel_creation", channel_type = "unbounded");
        let _enter = span.enter();
        unbounded()
    };

    // Calculate chunk size for each thread
    let chunk_size = lines.len().div_ceil(num_threads);
    let mut workers = Vec::new();

    // Spawn worker threads
    for worker_id in 0..num_threads {
        let span = info_span!("spawn_worker", worker_id);
        let _enter = span.enter();
        let start_idx = worker_id * chunk_size;
        let result_sender = result_sender.clone();
        let regex = regex.clone();
        let lines_len = lines.len();
        // Copy just the chunk we need for this thread
        let end_idx = (start_idx + chunk_size).min(lines_len);
        let chunk = lines[start_idx..end_idx].to_vec();

        let worker = thread::spawn(move || {
            let span = info_span!("worker_thread", worker_id);
            let _enter = span.enter();
            let mut matches_found = 0;

            for line in &chunk {
                if regex.is_match(line.as_bytes()) {
                    matches_found += 1;
                }
            }
            if result_sender.send(matches_found).is_err() {
                warn!("Results channel closed unexpectedly");
            }
        });
        workers.push(worker);
    }

    drop(result_sender);

    // Wait for threads to complete
    for worker in workers {
        worker.join().expect("Worker thread panicked");
    }

    // Collect and sort results
    let matched_lines = {
        let _enter = info_span!("collect_results").entered();

        // let mut lines: Vec<(usize, String)> = Vec::new();
        let mut count = 0;
        while let Ok(matches_found) = result_receiver.recv() {
            count += matches_found;
            // lines.push((line_number, line));
        }

        count
    };

    Ok(matched_lines)
}

#[instrument]
pub fn count_matches_regex(filepath: &str, pattern: &str, num_threads: usize) -> io::Result<usize> {
    // First read the entire file
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);
    let mut lines: Vec<(usize, String)> = Vec::new();

    {
        let _ = info_span!("read_file").entered();

        for (line_number, line) in reader.lines().enumerate() {
            if let Ok(line) = line {
                lines.push((line_number, line));
            }
        }
    }

    // Create channels for distributing work and collecting results
    let (result_sender, result_receiver) = {
        let span = info_span!("channel_creation", channel_type = "unbounded");
        let _enter = span.enter();
        unbounded()
    };

    // Compile regex
    let regex = {
        let span = info_span!("compile_regex", pattern);
        let _enter = span.enter();
        Regex::new(pattern).expect("Invalid regex pattern")
    };

    println!("Regex {regex:#?}");

    let lines = Arc::new(lines);

    // Calculate chunk size for each thread
    let chunk_size = lines.len().div_ceil(num_threads);
    let mut workers = Vec::new();

    // Spawn worker threads
    for worker_id in 0..num_threads {
        let span = info_span!("spawn_worker", worker_id);
        let _enter = span.enter();

        let start_idx = worker_id * chunk_size;
        let lines = lines.clone();
        let result_sender = result_sender.clone();
        let regex = regex.clone();

        let worker = thread::spawn(move || {
            let span = info_span!("worker_thread", worker_id);
            let _enter = span.enter();

            let mut matches_found = 0;
            let worker_start = Instant::now();

            let end_idx = (start_idx + chunk_size).min(lines.len());
            let chunk = &lines[start_idx..end_idx];

            for (_, line) in chunk {
                if regex.is_match(line) {
                    matches_found += 1;
                }
            }

            if result_sender.send(matches_found).is_err() {
                warn!("Results channel closed unexpectedly");
            }

            let duration = worker_start.elapsed();
            info!(
                worker_id,
                chunk_size = end_idx - start_idx,
                total_matches = matches_found,
                duration = format!("{:.2?}", duration),
                rate = format!(
                    "{:.2} lines/sec",
                    (end_idx - start_idx) as f64 / duration.as_secs_f64()
                )
            );
        });

        workers.push(worker);
    }

    drop(result_sender);

    // Wait for threads to complete
    for worker in workers {
        worker.join().expect("Worker thread panicked");
    }

    // Collect and sort results
    let matched_lines = {
        let span = info_span!("collect_results");
        let _enter = span.enter();

        // let mut lines: Vec<(usize, String)> = Vec::new();
        let collect_start = Instant::now();
        let mut count = 0;

        while let Ok(matches_found) = result_receiver.recv() {
            count += matches_found;
            // lines.push((line_number, line));
        }

        info!(
            matches_collected = count,
            duration = format!("{:.2?}", collect_start.elapsed())
        );

        count
    };

    Ok(matched_lines)
}

// Single-threaded version with separate read and process phases
#[instrument]
pub fn count_matches_regex_single_phased(filepath: &str, pattern: &str) -> io::Result<usize> {
    // Compile regex first
    let regex = {
        let span = info_span!("compile_regex", pattern);
        let _enter = span.enter();
        Regex::new(pattern).expect("Invalid regex pattern")
    };

    // Read phase
    let lines = {
        let span = info_span!("read_phase");
        let _enter = span.enter();
        // let read_start = Instant::now();

        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().map_while(Result::ok).collect();
        lines
    };
    let mut match_count = 0;
    for _ in 0..1 {
        // Process phase
        match_count = {
            let span = info_span!("process_phase");
            let _enter = span.enter();

            let mut matches = 0;
            for line in &lines {
                if regex.is_match(black_box(line)) {
                    matches += 1;
                }
            }

            matches
        };
    }

    Ok(match_count)
}
