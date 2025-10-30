pub mod automata;
pub mod cpu;
pub mod gpu;
pub mod metal_strategy;
pub mod regex;
pub mod shift_and_dist;
pub mod shift_and_dist_gpu;
pub mod shift_and_opt;

use anyhow::Result;
use shift_and_dist::BLOCK_SIZE;

use pcap::{Capture, Offline};

use anyhow::anyhow;

use std::{
    env, fs,
    fs::File,
    future::Future,
    hint::black_box,
    io::{BufRead, BufReader},
    sync,
    thread::available_parallelism,
    time::{SystemTime, UNIX_EPOCH},
};
use tracing::{instrument, warn, Level};

#[instrument]
pub fn read_file_blocked(path: &str) -> Result<Box<[[u8; BLOCK_SIZE]]>> {
    let file = File::open(path)?;
    let lines = BufReader::new(file).lines();
    let mut blocks: Vec<[u8; BLOCK_SIZE]> = Vec::with_capacity(1000);

    for line in lines {
        let line = line?;
        let bytes = line.as_bytes();

        // allocate new block
        // Use reserve to ensure capacity
        blocks.reserve(1);

        let current_len = blocks.len();
        unsafe {
            // SAFETY: We've reserved the space, and we'll initialize it immediately
            blocks.set_len(current_len + 1);
        }
        let block = &mut blocks[current_len];

        let copy_size = bytes.len().min(BLOCK_SIZE - 1);
        block[..copy_size].copy_from_slice(&bytes[..copy_size]);
        block[copy_size] = 0x17;
        block[copy_size + 1..].fill(0);
    }

    Ok(blocks.into_boxed_slice())
}

#[instrument]
pub fn read_file_blocked2(path: &str, delimiter: u8) -> Result<Box<[[u8; BLOCK_SIZE]]>> {
    let file = File::open(path)?;
    let lines = BufReader::new(file).lines();
    let mut blocks = Vec::new();
    for line in lines {
        let line = line?;
        let mut block = [0u8; BLOCK_SIZE];
        let bytes = line.as_bytes();

        let copy_size = bytes.len().min(BLOCK_SIZE - 1);
        block[..copy_size].copy_from_slice(&bytes[..copy_size]);
        block[copy_size] = delimiter;
        blocks.push(block);
    }

    Ok(blocks.into_boxed_slice())
}

#[instrument]
pub fn read_pcap_blocked(path: &str) -> Result<Box<[[u8; BLOCK_SIZE]]>> {
    // Open the PCAP file
    let mut capture =
        Capture::from_file(path).map_err(|e| anyhow!("Failed to open PCAP file: {}", e))?;

    let mut blocks: Vec<[u8; BLOCK_SIZE]> = Vec::with_capacity(1000);

    // Process each packet
    while let Ok(packet) = capture.next_packet() {
        // Convert packet data to a string representation for processing
        let packet_data = String::from_utf8_lossy(&packet.data);

        // Process each line in the packet
        for line in packet_data.lines() {
            let bytes = line.as_bytes();

            // Allocate new block
            // Use reserve to ensure capacity
            blocks.reserve(1);
            let current_len = blocks.len();
            unsafe {
                // SAFETY: We've reserved the space, and we'll initialize it immediately
                blocks.set_len(current_len + 1);
            }

            let block = &mut blocks[current_len];
            let copy_size = bytes.len().min(BLOCK_SIZE - 1);
            block[..copy_size].copy_from_slice(&bytes[..copy_size]);
            block[copy_size] = 0x17; // Add sentinel value
            block[copy_size + 1..].fill(0); // Zero-fill the rest
        }
    }

    Ok(blocks.into_boxed_slice())
}
