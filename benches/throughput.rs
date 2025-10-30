use std::any::Any;
use std::fs::File;
use std::io::{BufRead, BufReader};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use libc::rand;
use shift_and::metal_strategy::{
    ArgumentBufferStrategy, BasicStrategy, CombinedStrategy, IndirectCommandBufferStrategy,
};
use shift_and::shift_and_dist_gpu::ShiftAndDistGPU;
use shift_and::{gpu, read_file_blocked2};

use criterion::async_executor::AsyncStdExecutor;
use shift_and::regex::count_matches_regex_bytes2;
use shift_and::shift_and_dist::ShiftAndDist;

use rand::Rng;

fn read_file_to_vec(path: &str) -> Vec<String> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut lines: Vec<String> = Vec::new();

    for line in reader.lines() {
        if let Ok(line) = line {
            lines.push(line);
        }
    }

    lines
}

pub fn criterion_benchmark(c: &mut Criterion) {
    static PAGE_SIZE: usize = usize::pow(2, 14);
    let input_file = "../data/random_2.8gb.txt";
    // let regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}";

    // let regex = "a(b|c)";
    let regex = "a[^b]{62}b";
    // let sa_pre_compile = gpu::ShiftAndDistGPU::new(regex, false);
    let sa_basic = ShiftAndDistGPU::<BasicStrategy>::new_with_strategy(regex, false);
    let sa_arguemt = ShiftAndDistGPU::<ArgumentBufferStrategy>::new_with_strategy(regex, false);
    let sa_indirect =
        ShiftAndDistGPU::<IndirectCommandBufferStrategy>::new_with_strategy(regex, false);
    let sa_combined = ShiftAndDistGPU::<CombinedStrategy>::new_with_strategy(regex, false);

    let regex = regex::bytes::Regex::new(regex).unwrap();

    let input_base: Box<[[u8; PAGE_SIZE]]> = read_file_blocked2(input_file, 0x17).unwrap();
    let input_regex: Vec<String> = read_file_to_vec(input_file);

    assert_eq!(input_base.len(), input_regex.len());

    let escaped_file_name = input_file.split('/').last().unwrap();

    let mut c = c.benchmark_group(format!("`{}` {}", regex, escaped_file_name));

    c.significance_level(0.05).sample_size(100);

    for len in 18..=18 {
        let num_rows = usize::pow(2, len);
        // let num_rows = len * 1024 + 1;

        c.throughput(criterion::Throughput::Bytes(
            input_regex[..num_rows]
                .iter()
                .map(|s| s.len())
                .sum::<usize>() as u64,
        ));
        c.bench_with_input(
            BenchmarkId::new("GPU: BasicStategy", num_rows),
            &input_base,
            |b, s| {
                let input = s[..num_rows].to_vec().into_boxed_slice();
                b.to_async(AsyncStdExecutor)
                    .iter(|| sa_basic.count_match_lines_future(&input).unwrap());
            },
        );
        c.bench_with_input(
            BenchmarkId::new("GPU: ArgumentBufferStrategy", num_rows),
            &input_base,
            |b, s| {
                let input = s[..num_rows].to_vec().into_boxed_slice();
                b.to_async(AsyncStdExecutor)
                    .iter(|| sa_arguemt.count_match_lines_future(&input).unwrap());
            },
        );
        c.bench_with_input(
            BenchmarkId::new("GPU: IndirectCommandBufferStrategy", num_rows),
            &input_base,
            |b, s| {
                let input = s[..num_rows].to_vec().into_boxed_slice();
                b.to_async(AsyncStdExecutor)
                    .iter(|| sa_indirect.count_match_lines_future(&input).unwrap());
            },
        );
        c.bench_with_input(
            BenchmarkId::new("GPU: CombinedStrategy", num_rows),
            &input_base,
            |b, s| {
                let input = s[..num_rows].to_vec().into_boxed_slice();
                b.to_async(AsyncStdExecutor)
                    .iter(|| sa_combined.count_match_lines_future(&input).unwrap());
            },
        );
        c.bench_with_input(
            BenchmarkId::new("CPU: Singlethreads", num_rows),
            &input_regex,
            |b, s| {
                let input = s[..num_rows].to_vec().into_boxed_slice();
                b.iter(|| {
                    let num_matches = input.iter().map(|row| regex.find(row.as_bytes())).count();
                    num_matches
                });
            },
        );
        c.bench_with_input(
            BenchmarkId::new("CPU: 8 Threads", num_rows),
            &input_regex,
            |b, s| {
                let input = s[..num_rows].to_vec().into_boxed_slice();
                b.iter(|| {
                    let num_matches = count_matches_regex_bytes2(&regex, &input, 7).unwrap();
                    num_matches
                });
            },
        );
        c.bench_with_input(
            BenchmarkId::new("CPU: 8 Threads", num_rows),
            &input_regex,
            |b, s| {
                let input = s[..num_rows].to_vec().into_boxed_slice();
                b.iter(|| {
                    let num_matches = count_matches_regex_bytes2(&regex, &input, 8).unwrap();
                    num_matches
                });
            },
        );
    }
}

// fn generate_random_data(size: usize) -> Box<[u8]> {
//     let mut rng = rand::rng();
//     let mut data = vec![0u8; size].into_boxed_slice();
//     rng.fill(&mut data[..]);
//     data
// }

// fn copy_memory(src: &[u8], dst: &mut [u8]) {
//     dst.copy_from_slice(src);
// }

// fn criterion_benchmark(c: &mut Criterion) {
//     static PAGE_SIZE: usize = usize::pow(2, 14); // 16KB pages

//     // Generate random data to use as source
//     let input_base = generate_random_data(usize::pow(2, 30)); // Enough data for largest test

//     let mut benchmark_group = c.benchmark_group("memory_copy_throughput");
//     benchmark_group.significance_level(0.05).sample_size(100);

//     for power in 20..=30 {
//         let size = usize::pow(2, power);
//         benchmark_group.throughput(Throughput::Bytes(size as u64));

//         benchmark_group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
//             // Pre-allocate destination buffer

//             // Use source data of appropriate size
//             let src = &input_base[0..size];

//             b.iter(|| {
//                 let mut dst = vec![0u8; size].into_boxed_slice();
//                 copy_memory(src, &mut dst)
//             });
//         });
//     }

//     benchmark_group.finish();
// }

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
