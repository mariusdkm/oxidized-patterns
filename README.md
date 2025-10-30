# Oxidized Patterns

**Metal with Rust: High-Performance Regex Execution on Unified Memory GPUs**

A GPU-accelerated regular expression engine optimized for Apple's M1 unified memory architecture, implementing the Shift-And-Dist algorithm in Rust with Metal compute shaders.

## About This Thesis

This repository contains the complete implementation of a **Bachelor's Thesis in Informatics** completed at the **Technische Universität München (TUM)** in March 2025.

**📄 Full Thesis:** [thesis.pdf](thesis.pdf)

### Abstract

The explosive growth of data-intensive applications demands high-performance regular expression (regex) engines capable of processing large-scale text efficiently. While GPUs offer massive parallelism, traditional architectures suffer from CPU-GPU data transfer overhead, limiting practical performance gains. This thesis addresses these challenges through the design and implementation of a GPU-accelerated regex engine optimized for Apple's M1 unified memory architecture.

By eliminating data copying between CPU and GPU through unified memory, and leveraging Rust with the Metal API, this work develops a novel pipeline that compiles regex patterns into minimized deterministic finite automata (DFA) for execution via a bit-parallel Shift-And-Dist algorithm on the GPU.

### Key Contributions

- **Novel GPU Regex Engine**: First comprehensive implementation and evaluation of regex matching on Apple's M1 unified memory architecture
- **Just-in-Time (JIT) Shader Compilation**: Hard-codes automaton masks into shaders, reducing memory latency by 48%
- **Four Optimized Metal Execution Strategies**:
  - **Basic Strategy**: Direct buffer binding with minimal overhead
  - **Argument Buffer Strategy**: Indirect buffer addressing for complex pipelines
  - **Indirect Command Buffer Strategy**: Preserves pipeline state across executions
  - **Combined Strategy**: Hybrid approach leveraging multiple optimizations
- **Complete Regex Compilation Pipeline**: NFA → DFA → Minimized DFA → GPU-executable bitmasks
- **Performance Achievements**:
  - Peak throughput of **10.87 GiB/s** on complex patterns
  - Up to **4× speedup** over multi-threaded CPU baseline (Rust regex crate)
  - Sublinear scaling for inputs below 2 GiB through unified memory optimization

## Architecture

The implementation follows a complete regex-to-GPU compilation pipeline:

1. **Regex Parsing** → High-Level Intermediate Representation (HIR)
2. **Thompson's Construction** → Non-deterministic Finite Automaton (NFA)
3. **Subset Construction** → Deterministic Finite Automaton (DFA)
4. **Topological Sorting** (Kahn's Algorithm) → GPU-executable bitmasks
5. **JIT Shader Compilation** → Optimized Metal compute kernel
6. **GPU Execution** → Parallel pattern matching across input blocks

### Technical Highlights

- **Bit-Parallel Algorithm**: Shift-And-Dist with distance transitions supporting patterns up to 64 states
- **Page-Aligned Memory**: Input divided into 16 KiB blocks matching M1 page size
- **Unified Memory Integration**: Zero-copy data sharing via `MTLStorageModeShared`
- **Asynchronous Execution**: Future-based API for concurrent GPU workloads
- **Comprehensive Validation**: Results verified against Rust's `regex` crate

### Core Components

- `src/shift_and_dist.rs` - Shift-And-Dist algorithm and automaton construction
- `src/shift_and_dist_gpu.rs` - Metal GPU wrapper with async support
- `src/metal_strategy.rs` - Four execution strategies with different optimization approaches
- `shaders/shift_and_dist.metal` - Metal compute kernel (JIT-compiled)
- `src/automata.rs` - NFA/DFA construction and minimization
- `src/regex.rs` - CPU baseline for validation and comparison
- `src/cpu.rs` - Optimized multi-threaded CPU implementations

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4) recommended
  - Apple G13 GPU architecture (M1 series) or newer
  - Unified memory architecture required for optimal performance
- **Rust** 1.70 or later (tested with Rust 1.82)
- **Xcode Command Line Tools** (for Metal shader compilation)
  - Install via: `xcode-select --install`
- **macOS Sequoia 15.3** or later (tested on 15.3.1)

## Building

```bash
# Clone the repository
git clone https://github.com/mariusdkm/oxidized-patterns.git
cd oxidized-patterns

# Build in release mode (recommended for performance)
cargo build --release

# Or build in debug mode for development
cargo build
```

The build process will automatically compile the Metal shaders via `build.rs`.

## Usage

The main binary provides a command-line interface for pattern matching:

```bash
# Basic usage
cargo run --release -- <PATTERN> <FILE>

# Example: Search for email patterns
cargo run --release -- '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}' data.txt

# Specify GPU execution strategy
cargo run --release -- --strategy combined 'pattern' file.txt

# Validate results against CPU regex implementation
cargo run --release -- --validate 'pattern' file.txt

# Process with multiple chunks for parallelism
cargo run --release -- --chunks 16 'pattern' file.txt

# Enable performance tracing (creates .pftrace files)
cargo run --release -- --trace 'pattern' file.txt
```

### Command-Line Options

- `<PATTERN>` - Regular expression pattern to search for
- `<FILE>` - Input file to search in
- `-s, --strategy <STRATEGY>` - GPU execution strategy: `basic`, `argument-buffer`, `indirect-command-buffer`, `combined` (default: `basic`)
- `-c, --chunks <N>` - Number of parallel chunks to process (default: 8)
- `-v, --validate` - Validate GPU results against CPU regex implementation
- `-t, --trace` - Enable performance tracing (outputs to `traces/` directory)

### Example Workflows

```bash
# Compare different strategies on the same pattern
cargo run --release -- -s basic 'a[^b]{62}b' BeeMovieScript.txt
cargo run --release -- -s combined 'a[^b]{62}b' BeeMovieScript.txt

# Validate correctness while benchmarking
cargo run --release -- -v 'a(b|c)' data.txt

# Generate performance traces for analysis
cargo run --release -- -t -s combined 'pattern' large_file.txt
```

## Benchmarking

The project includes comprehensive benchmarks using the Criterion framework:

```bash
# Run all benchmarks
cargo bench

# Run benchmarks and generate HTML reports
cargo bench -- --verbose

# Run specific benchmark
cargo bench --bench throughput
```

### Benchmark Configuration

The `benches/throughput.rs` file includes:
- Comparison of all four GPU strategies
- CPU baseline (single-threaded regex)
- Multi-threaded CPU implementation (8 threads)
- Configurable input sizes (powers of 2)
- Throughput measurements in bytes/second

**Note:** Benchmarks require test data file at `../data/random_2.8gb.txt`. You can generate test data using the provided script:

```bash
# Generate 2.8 GB of random data
python3 generate_random_data.py 2.8

# The script creates files with random ASCII characters
# Line lengths uniformly distributed between 1 and 16 KiB
```

### Customizing Benchmarks

Edit `benches/throughput.rs` to:
- Change the regex pattern (line 36)
- Modify input data path (line 35)
- Adjust test input sizes (line 52)
- Configure number of iterations (line 50)

Example patterns to test:
```rust
let regex = "a(b|c)";                                           // Alternation
let regex = "a[^b]{62}b";                                       // Character class with repetition
let regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"; // Email pattern
```

## Performance Analysis

Results are generated in `target/criterion/` as HTML reports. View them by opening:
```bash
open target/criterion/report/index.html
```

Performance traces (when using `--trace` flag) are saved in the `traces/` directory as `.pftrace` files, which can be analyzed with tools like Chrome's Perfetto UI.

## Project Structure

```
oxidized-patterns/
├── src/
│   ├── main.rs                 # CLI interface
│   ├── lib.rs                  # Library root
│   ├── shift_and_dist.rs       # Core algorithm
│   ├── shift_and_dist_gpu.rs   # GPU implementation
│   ├── metal_strategy.rs       # Metal execution strategies
│   ├── automata.rs             # NFA/DFA construction
│   ├── regex.rs                # CPU baseline
│   └── cpu.rs                  # CPU-optimized implementations
├── shaders/
│   └── shift_and_dist.metal    # Metal compute kernel
├── benches/
│   └── throughput.rs           # Performance benchmarks
├── build.rs                    # Build script (compiles shaders)
├── Cargo.toml                  # Project dependencies
├── thesis.pdf                  # Full thesis document
└── README.md                   # This file
```

## Implementation Details

### Shift-And-Dist Algorithm

The Shift-And-Dist algorithm (Le Glaunec et al., 2023) extends the classic bit-parallel Shift-And algorithm with distance transitions:

- **Bit-Parallel State Representation**: Each bit in a 64-bit word represents one automaton state
- **Character Masks**: `masks_char[256]` encode valid transitions for each ASCII character
- **Distance Masks**: `masks_dist[]` enable variable-length jumps between states (e.g., for patterns like `a{0,3}b`)
- **Initial/Final Masks**: `mask_initial` and `mask_final` mark start and accepting states
- **Linear Time Complexity**: O(n) matching time when pattern length ≤ 64 states


### GPU Acceleration on Apple M1

The Metal implementation exploits the M1's unified memory architecture:

**Memory Management:**
- Input divided into 16 KiB blocks (matching M1 page size: 2^14 bytes)
- Shared memory buffers (`MTLStorageModeShared`) enable zero-copy CPU-GPU access
- Page-aligned allocations optimize memory controller access patterns

**Thread Organization:**
- Each GPU thread processes one 16 KiB block independently
- Threadgroups configured to match hardware (32 threads/SIMD-group, 1024 threads/threadgroup)
- `dispatchThreads` dynamically adjusts for non-uniform input sizes

**JIT Optimization:**
- Automaton masks hard-coded directly into shader at runtime
- Eliminates repeated device memory loads (48% latency reduction)
- Compiler optimizations specific to each regex pattern
- Integer type trimming (e.g., `uint8_t` for patterns with ≤8 states)

**Execution Strategies:**
1. **BasicStrategy**: Direct buffer binding, minimal overhead
2. **ArgumentBufferStrategy**: Indirect addressing for complex pipelines
3. **IndirectCommandBufferStrategy**: Preserves pipeline state across calls
4. **CombinedStrategy**: Combines argument + indirect command buffers

**Performance Characteristics:**
- Peak throughput: 10.87 GiB/s (4× CPU speedup on complex patterns)
- Sublinear scaling up to 2 GiB input due to unified memory
- Bimodal distribution beyond 2 GiB due to memory caching effects
- Asynchronous dispatching overlaps memory wiring with computation

## Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test module
cargo test automata

# Run with validation against regex
cargo run --release -- -v 'test_pattern' test_file.txt
```



## Citation

If you use this work in academic research, please cite:

```bibtex
@mastersthesis{dekuthymeurers2025metalrust,
  author = {De Kuthy Meurers, Marius},
  title = {Metal with Rust: High-Performance Regex Execution on Unified Memory GPUs},
  school = {Technische Universit\"{a}t M\"{u}nchen},
  year = {2025},
  month = {March},
  type = {Bachelor's Thesis in Informatics},
  address = {Munich, Germany},
}
```

## Acknowledgments

This thesis builds upon foundational work in GPU-accelerated pattern matching:

- **Shift-And algorithm**: Baeza-Yates & Gonnet (1992), Wu & Manber (1992)
- **Shift-And-Dist extension**: Le Glaunec et al. (2023) - HybridSA framework
- **GPU regex research**: Jakob et al. (2006), Vasiliadis et al. (2009-2010), Cascarano et al. (2010), Liu et al. (2020-2021)
- **Apple's Metal framework**: Unified memory architecture and compute shaders
- **Rust ecosystem**: `regex`, `regex-syntax`, `objc2`, `objc2-metal` crates
