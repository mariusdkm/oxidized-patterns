use std::{
    fs,
    hint::black_box,
    sync,
    thread::available_parallelism,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use clap::{Parser, ValueEnum};
use oxidized_patterns::{
    cpu::ShiftAndDistCPU,
    metal_strategy::{
        ArgumentBufferStrategy, BasicStrategy, CombinedStrategy, IndirectCommandBufferStrategy, MetalStrategy
    },
    read_file_blocked2,
    regex::count_matches_regex,     
    shift_and_dist::{ShiftAndDist, BLOCK_SIZE},
    shift_and_dist_gpu::ShiftAndDistGPU,
};
use tracing_perfetto::PerfettoLayer;
use tracing_subscriber::prelude::*;

/// A pattern matching tool using GPU acceleration
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Pattern to search for
    #[clap(index = 1)]
    pattern: String,

    /// Input file to search in
    #[clap(index = 2)]
    file: String,

    /// Strategy to use for GPU acceleration
    #[clap(short, long, value_enum, default_value = "basic")]
    strategy: Strategy,

    /// Number of parallel chunks to process
    #[clap(short, long, default_value_t = 8)]
    chunks: usize,

    /// Enable performance tracing
    #[clap(short, long)]
    trace: bool,

    /// Compare results with regex implementation
    #[clap(
        short = 'v',
        long,
        help = "Validate results against regex implementation"
    )]
    validate: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Strategy {
    Basic,
    ArgumentBuffer,
    IndirectCommandBuffer,
    Combined
}

// Helper function to verify alignment
fn verify_alignment(data: &[[u8; BLOCK_SIZE]]) -> bool {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    data.iter()
        .all(|block| (block.as_ptr() as i64) % page_size == 0)
}

fn setup_perfetto(pattern: &str, file: &str) {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let file_name = file.split('/').last().unwrap();
    fs::create_dir_all("traces").unwrap();
    let pattern = pattern.replace('/', "_");

    let layer = PerfettoLayer::new(sync::Mutex::new(
        fs::File::create(format!("traces/{time}-{pattern}-{file_name}.pftrace",)).unwrap(),
    ));

    tracing_subscriber::registry().with(layer).init();
}

fn main() {
    let args = Args::parse();

    if args.trace {
        setup_perfetto(&args.pattern, &args.file);
    }

    let _span = tracing::info_span!("count_match_lines").entered();
    let text = read_file_blocked2(&args.file, 0x17).unwrap();
    assert!(verify_alignment(&text), "Data is not aligned");

    // Process text based on the selected strategy
    let (gpu_matches, gpu_duration) = {
        let start = Instant::now();
        let matches = match args.strategy {
            Strategy::Basic => process_with_strategy::<BasicStrategy>(&args.pattern, &text, args.chunks),
            Strategy::ArgumentBuffer => process_with_strategy::<ArgumentBufferStrategy>(&args.pattern, &text, args.chunks),
            Strategy::IndirectCommandBuffer => process_with_strategy::<IndirectCommandBufferStrategy>(&args.pattern, &text, args.chunks),
            Strategy::Combined => process_with_strategy::<CombinedStrategy>(&args.pattern, &text, args.chunks)
        };
        let duration = start.elapsed();
        (matches, duration)
    };

    // Print GPU results
    let total_gpu_matches: usize = gpu_matches.iter().sum();
    println!("\nGPU Results:");
    println!("Total matches found = {}", total_gpu_matches);
    for (i, count) in gpu_matches.iter().enumerate() {
        println!("Chunk {} matches found = {}", i, count);
    }


    // Compare with regex implementation if validation is requested
    if args.validate {
        println!("\nValidating with regex implementation...");
        let num_threads = available_parallelism().map(|p| p.get()).unwrap_or(1);
        match count_matches_regex(&args.file, &args.pattern, num_threads) {
            Ok(regex_matches) => {
                println!("Regex implementation found {} matches", regex_matches);
                if regex_matches == total_gpu_matches {
                    println!("✅ Results match!");
                } else {
                    println!("❌ Results differ!");
                    println!("GPU implementation: {} matches", total_gpu_matches);
                    println!("Regex implementation: {} matches", regex_matches);
                }
            }
            Err(e) => {
                println!("Error running regex implementation: {}", e);
            }
        }
    }
}

fn process_with_strategy<S>(
    pattern: &str,
    text: &[[u8; BLOCK_SIZE]],
    num_chunks: usize,
) -> Vec<usize>
where
    S: 'static + Clone + Send + Sync + MetalStrategy,
{
    let sa = ShiftAndDistGPU::<S>::new_with_strategy(black_box(pattern), false);
    let chunks = text
        .chunks(text.len() / num_chunks.max(1))
        .collect::<Vec<_>>();
    let mut results = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        // println!("Processing chunk of size = {}", chunk.len());
        // Fix: Process the current chunk instead of the entire text
        let count = sa.count_match_lines(black_box(chunk)).unwrap();
        results.push(count);
    }

    results
}
