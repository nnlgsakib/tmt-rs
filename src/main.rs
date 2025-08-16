mod tmt;
mod merkle;

use tmt::{TernaryMeshTree, TmtConfig};
use merkle::{MerkleTree, MerkleConfig, hash_to_hex as merkle_hash_to_hex};
use std::time::{Instant, Duration};
use std::sync::{Arc, Mutex};
use std::thread;

// Benchmark configuration
#[derive(Clone)]
struct BenchmarkConfig {
    data_sizes: Vec<usize>,
    iterations: usize,
    thread_counts: Vec<usize>,
    enable_concurrent_tests: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            data_sizes: vec![100, 1000, 10000, 50000],
            iterations: 5,
            thread_counts: vec![1, 2, 4, 8],
            enable_concurrent_tests: true,
        }
    }
}

// Benchmark results
#[derive(Debug, Default)]
struct BenchmarkResult {
    operation: String,
    tree_type: String,
    data_size: usize,
    avg_time_ms: f64,
    min_time_ms: f64,
    max_time_ms: f64,
    memory_usage_mb: f64,
    throughput_ops_per_sec: f64,
}

// Generate test data
fn generate_test_data(size: usize, data_length: usize) -> Vec<Vec<u8>> {
    (0..size)
        .map(|i| format!("test_data_block_{:08}_padding", i).as_bytes()[..data_length.min(32)].to_vec())
        .collect()
}

// Benchmark TMT operations
fn benchmark_tmt_operations(config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    println!("🚀 Benchmarking TMT Operations...");

    for &data_size in &config.data_sizes {
        println!("  📊 Testing with {} data blocks", data_size);

        // Build benchmark
        let mut build_times = Vec::new();
        let mut memory_usage = 0.0;

        for i in 0..config.iterations {
            let test_data = generate_test_data(data_size, 32);
            let mut tmt = TernaryMeshTree::new(TmtConfig {
                enable_metrics: true,
                enable_caching: true,
                max_cache_size: 50000,
                parallel_threshold: 1000,
            });

            let start = Instant::now();
            tmt.build(test_data).expect("TMT build failed");
            let duration = start.elapsed();

            build_times.push(duration.as_secs_f64() * 1000.0);

            if i == 0 {
                let metrics = tmt.get_metrics();
                memory_usage = metrics.memory_usage_bytes as f64 / (1024.0 * 1024.0);
            }
        }

        results.push(BenchmarkResult {
            operation: "Build".to_string(),
            tree_type: "TMT".to_string(),
            data_size,
            avg_time_ms: build_times.iter().sum::<f64>() / build_times.len() as f64,
            min_time_ms: build_times.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_time_ms: build_times.iter().fold(0.0, |a, &b| a.max(b)),
            memory_usage_mb: memory_usage,
            throughput_ops_per_sec: data_size as f64 / (build_times.iter().sum::<f64>() / 1000.0 / build_times.len() as f64),
        });

        // Verification benchmark
        let test_data = generate_test_data(data_size, 32);
        let mut tmt = TernaryMeshTree::with_default_config();
        tmt.build(test_data.clone()).expect("TMT build failed");

        let mut verify_times = Vec::new();
        for _ in 0..config.iterations {
            let test_index = data_size / 2;
            let start = Instant::now();
            let result = tmt.verify(test_index, &test_data[test_index]).expect("TMT verify failed");
            let duration = start.elapsed();

            assert!(result, "Verification should succeed");
            verify_times.push(duration.as_nanos() as f64 / 1_000_000.0);
        }

        results.push(BenchmarkResult {
            operation: "Verify".to_string(),
            tree_type: "TMT".to_string(),
            data_size,
            avg_time_ms: verify_times.iter().sum::<f64>() / verify_times.len() as f64,
            min_time_ms: verify_times.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_time_ms: verify_times.iter().fold(0.0, |a, &b| a.max(b)),
            memory_usage_mb: memory_usage,
            throughput_ops_per_sec: 1000.0 / (verify_times.iter().sum::<f64>() / verify_times.len() as f64),
        });

        // Update benchmark
        let mut update_times = Vec::new();
        for _ in 0..config.iterations {
            let test_index = data_size / 4;
            let new_data = format!("updated_data_{}", rand::random::<u32>()).into_bytes();

            let start = Instant::now();
            tmt.update(test_index, new_data.clone()).expect("TMT update failed");
            let duration = start.elapsed();

            update_times.push(duration.as_nanos() as f64 / 1_000_000.0);
        }

        results.push(BenchmarkResult {
            operation: "Update".to_string(),
            tree_type: "TMT".to_string(),
            data_size,
            avg_time_ms: update_times.iter().sum::<f64>() / update_times.len() as f64,
            min_time_ms: update_times.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_time_ms: update_times.iter().fold(0.0, |a, &b| a.max(b)),
            memory_usage_mb: memory_usage,
            throughput_ops_per_sec: 1000.0 / (update_times.iter().sum::<f64>() / update_times.len() as f64),
        });
    }

    results
}

// Benchmark Merkle tree operations
fn benchmark_merkle_operations(config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    println!("🌳 Benchmarking Merkle Tree Operations...");

    for &data_size in &config.data_sizes {
        println!("  📊 Testing with {} data blocks", data_size);

        // Build benchmark
        let mut build_times = Vec::new();
        let mut memory_usage = 0.0;

        for i in 0..config.iterations {
            let test_data = generate_test_data(data_size, 32);
            let mut merkle = MerkleTree::new(MerkleConfig {
                enable_metrics: true,
                enable_caching: true,
                max_cache_size: 50000,
            });

            let start = Instant::now();
            merkle.build(test_data).expect("Merkle build failed");
            let duration = start.elapsed();

            build_times.push(duration.as_secs_f64() * 1000.0);

            if i == 0 {
                let metrics = merkle.get_metrics();
                memory_usage = metrics.memory_usage_bytes as f64 / (1024.0 * 1024.0);
            }
        }

        results.push(BenchmarkResult {
            operation: "Build".to_string(),
            tree_type: "Merkle".to_string(),
            data_size,
            avg_time_ms: build_times.iter().sum::<f64>() / build_times.len() as f64,
            min_time_ms: build_times.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_time_ms: build_times.iter().fold(0.0, |a, &b| a.max(b)),
            memory_usage_mb: memory_usage,
            throughput_ops_per_sec: data_size as f64 / (build_times.iter().sum::<f64>() / 1000.0 / build_times.len() as f64),
        });

        // Verification benchmark
        let test_data = generate_test_data(data_size, 32);
        let mut merkle = MerkleTree::with_default_config();
        merkle.build(test_data.clone()).expect("Merkle build failed");

        let mut verify_times = Vec::new();
        for _ in 0..config.iterations {
            let test_index = data_size / 2;
            let start = Instant::now();
            let result = merkle.verify(test_index, &test_data[test_index]).expect("Merkle verify failed");
            let duration = start.elapsed();

            assert!(result, "Verification should succeed");
            verify_times.push(duration.as_nanos() as f64 / 1_000_000.0);
        }

        results.push(BenchmarkResult {
            operation: "Verify".to_string(),
            tree_type: "Merkle".to_string(),
            data_size,
            avg_time_ms: verify_times.iter().sum::<f64>() / verify_times.len() as f64,
            min_time_ms: verify_times.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_time_ms: verify_times.iter().fold(0.0, |a, &b| a.max(b)),
            memory_usage_mb: memory_usage,
            throughput_ops_per_sec: 1000.0 / (verify_times.iter().sum::<f64>() / verify_times.len() as f64),
        });

        // Update benchmark
        let mut update_times = Vec::new();
        for _ in 0..config.iterations {
            let test_index = data_size / 4;
            let new_data = format!("updated_data_{}", rand::random::<u32>()).into_bytes();

            let start = Instant::now();
            merkle.update(test_index, new_data.clone()).expect("Merkle update failed");
            let duration = start.elapsed();

            update_times.push(duration.as_nanos() as f64 / 1_000_000.0);
        }

        results.push(BenchmarkResult {
            operation: "Update".to_string(),
            tree_type: "Merkle".to_string(),
            data_size,
            avg_time_ms: update_times.iter().sum::<f64>() / update_times.len() as f64,
            min_time_ms: update_times.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_time_ms: update_times.iter().fold(0.0, |a, &b| a.max(b)),
            memory_usage_mb: memory_usage,
            throughput_ops_per_sec: 1000.0 / (update_times.iter().sum::<f64>() / update_times.len() as f64),
        });
    }

    results
}

// Concurrent benchmark
fn benchmark_concurrent_operations(config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    if !config.enable_concurrent_tests {
        return results;
    }

    println!("⚡ Benchmarking Concurrent Operations...");

    for &thread_count in &config.thread_counts {
        for &data_size in &[1000, 10000] {
            println!("  🧵 Testing with {} threads, {} data blocks", thread_count, data_size);

            // TMT concurrent verification
            let test_data = generate_test_data(data_size, 32);
            let mut tmt = TernaryMeshTree::with_default_config();
            tmt.build(test_data.clone()).expect("TMT build failed");
            let tmt = Arc::new(Mutex::new(tmt));

            let start = Instant::now();
            let handles: Vec<_> = (0..thread_count)
                .map(|thread_id| {
                    let tmt = Arc::clone(&tmt);
                    let test_data = test_data.clone();
                    thread::spawn(move || {
                        let operations_per_thread = 100;
                        for i in 0..operations_per_thread {
                            let index = (thread_id * operations_per_thread + i) % data_size;
                            let tmt = tmt.lock().unwrap();
                            let _ = tmt.verify(index, &test_data[index]);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }

            let duration = start.elapsed().as_secs_f64() * 1000.0;
            let total_ops = thread_count * 100;

            results.push(BenchmarkResult {
                operation: format!("Concurrent_Verify_{}_threads", thread_count),
                tree_type: "TMT".to_string(),
                data_size,
                avg_time_ms: duration,
                min_time_ms: duration,
                max_time_ms: duration,
                memory_usage_mb: 0.0,
                throughput_ops_per_sec: total_ops as f64 / (duration / 1000.0),
            });
        }
    }

    results
}

// Stress test with large datasets
fn stress_test(config: &BenchmarkConfig) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    println!("💪 Running Stress Tests...");

    let large_sizes = vec![100_000, 500_000];

    for &size in &large_sizes {
        println!("  🏋️ Stress testing TMT with {} blocks", size);

        // TMT Stress Test
        let test_data = generate_test_data(size, 32);
        let mut tmt = TernaryMeshTree::new(TmtConfig {
            enable_metrics: true,
            enable_caching: true,
            max_cache_size: 100_000,
            parallel_threshold: 1000,
        });

        let start = Instant::now();
        tmt.build(test_data.clone()).expect("TMT stress build failed");
        let duration = start.elapsed();

        let metrics = tmt.get_metrics();
        let memory_usage = metrics.memory_usage_bytes as f64 / (1024.0 * 1024.0);

        results.push(BenchmarkResult {
            operation: "Stress_Build".to_string(),
            tree_type: "TMT".to_string(),
            data_size: size,
            avg_time_ms: duration.as_secs_f64() * 1000.0,
            min_time_ms: 0.0,
            max_time_ms: 0.0,
            memory_usage_mb: memory_usage,
            throughput_ops_per_sec: size as f64 / duration.as_secs_f64(),
        });

        println!("  🏋️ Stress testing Merkle with {} blocks", size);

        // Merkle Stress Test
        let mut merkle = MerkleTree::new(MerkleConfig {
            enable_metrics: true,
            enable_caching: true,
            max_cache_size: 100_000,
        });

        let start = Instant::now();
        merkle.build(test_data).expect("Merkle stress build failed");
        let duration = start.elapsed();

        let metrics = merkle.get_metrics();
        let memory_usage = metrics.memory_usage_bytes as f64 / (1024.0 * 1024.0);

        results.push(BenchmarkResult {
            operation: "Stress_Build".to_string(),
            tree_type: "Merkle".to_string(),
            data_size: size,
            avg_time_ms: duration.as_secs_f64() * 1000.0,
            min_time_ms: 0.0,
            max_time_ms: 0.0,
            memory_usage_mb: memory_usage,
            throughput_ops_per_sec: size as f64 / duration.as_secs_f64(),
        });
    }

    results
}

// Print TMT results table
fn print_tmt_table(results: &[BenchmarkResult]) {
    println!("\n📊 TMT Results");
    println!("{:<25} | {:<10} | {:<15} | {:<20} | {:<15}",
             "Operation", "Data Size", "Avg Time (ms)", "Throughput (ops/sec)", "Memory (MB)");
    println!("{}", "-".repeat(90));

    for result in results.iter().filter(|r| r.tree_type == "TMT") {
        if result.operation.starts_with("Concurrent") || result.operation == "Stress_Build" {
            continue; // Handled separately
        }
        println!("{:<25} | {:<10} | {:<15.3} | {:<20.2} | {:<15.2}",
                 result.operation, result.data_size, result.avg_time_ms,
                 result.throughput_ops_per_sec, result.memory_usage_mb);
    }

    // Concurrent for TMT
    println!("\nConcurrent Verify (TMT)");
    println!("{:<8} | {:<10} | {:<15} | {:<20}", "Threads", "Data Size", "Avg Time (ms)", "Throughput (ops/sec)");
    println!("{}", "-".repeat(60));
    for &threads in &[1, 2, 4, 8] {
        for &size in &[1000, 10000] {
            let op = format!("Concurrent_Verify_{}_threads", threads);
            if let Some(res) = results.iter().find(|r| r.operation == op && r.data_size == size) {
                println!("{:<8} | {:<10} | {:<15.3} | {:<20.2}",
                         threads, size, res.avg_time_ms, res.throughput_ops_per_sec);
            }
        }
    }

    // Stress for TMT
    println!("\nStress Build (TMT)");
    println!("{:<10} | {:<15} | {:<20} | {:<15}", "Data Size", "Avg Time (ms)", "Throughput (ops/sec)", "Memory (MB)");
    println!("{}", "-".repeat(70));
    for &size in &[100000, 500000] {
        if let Some(res) = results.iter().find(|r| r.operation == "Stress_Build" && r.tree_type == "TMT" && r.data_size == size) {
            println!("{:<10} | {:<15.3} | {:<20.2} | {:<15.2}",
                     size, res.avg_time_ms, res.throughput_ops_per_sec, res.memory_usage_mb);
        }
    }
}

// Print Merkle results table
fn print_merkle_table(results: &[BenchmarkResult]) {
    println!("\n📊 Merkle Results");
    println!("{:<25} | {:<10} | {:<15} | {:<20} | {:<15}",
             "Operation", "Data Size", "Avg Time (ms)", "Throughput (ops/sec)", "Memory (MB)");
    println!("{}", "-".repeat(90));

    for result in results.iter().filter(|r| r.tree_type == "Merkle") {
        if result.operation == "Stress_Build" {
            continue; // Handled separately
        }
        println!("{:<25} | {:<10} | {:<15.3} | {:<20.2} | {:<15.2}",
                 result.operation, result.data_size, result.avg_time_ms,
                 result.throughput_ops_per_sec, result.memory_usage_mb);
    }

    // Stress for Merkle
    println!("\nStress Build (Merkle)");
    println!("{:<10} | {:<15} | {:<20} | {:<15}", "Data Size", "Avg Time (ms)", "Throughput (ops/sec)", "Memory (MB)");
    println!("{}", "-".repeat(70));
    for &size in &[100000, 500000] {
        if let Some(res) = results.iter().find(|r| r.operation == "Stress_Build" && r.tree_type == "Merkle" && r.data_size == size) {
            println!("{:<10} | {:<15.3} | {:<20.2} | {:<15.2}",
                     size, res.avg_time_ms, res.throughput_ops_per_sec, res.memory_usage_mb);
        }
    }
}

// Print comparisons, scoring, and winners
fn print_comparisons(results: &[BenchmarkResult]) {
    let operations = vec!["Build", "Verify", "Update", "Stress_Build"];
    let mut tmt_score = 0;
    let mut merkle_score = 0;

    println!("\n🔍 Performance Comparisons and Scoring");
    println!("Scoring: +1 for better Avg Time (lower), Throughput (higher), Memory (lower) per data size.");

    for op in operations {
        println!("\n{} Comparison:", op);
        println!("{:<10} | {:<12} | {:<12} | {:<15} | {:<15} | {:<12} | {:<12} | {:<15}",
                 "Data Size", "TMT Time", "Merkle Time", "TMT TPut", "Merkle TPut", "TMT Mem", "Merkle Mem", "Winner");
        println!("{}", "-".repeat(110));

        let sizes = if op == "Stress_Build" { vec![100000, 500000] } else { vec![100, 1000, 10000, 50000] };
        for &size in &sizes {
            let tmt = results.iter().find(|r| r.operation == op && r.tree_type == "TMT" && r.data_size == size);
            let merkle = results.iter().find(|r| r.operation == op && r.tree_type == "Merkle" && r.data_size == size);

            if let (Some(t), Some(m)) = (tmt, merkle) {
                let time_winner = if t.avg_time_ms < m.avg_time_ms { "TMT" } else { "Merkle" };
                let tput_winner = if t.throughput_ops_per_sec > m.throughput_ops_per_sec { "TMT" } else { "Merkle" };
                let mem_winner = if t.memory_usage_mb < m.memory_usage_mb { "TMT" } else { "Merkle" };

                // Score
                if time_winner == "TMT" { tmt_score += 1; } else { merkle_score += 1; }
                if tput_winner == "TMT" { tmt_score += 1; } else { merkle_score += 1; }
                if mem_winner == "TMT" { tmt_score += 1; } else { merkle_score += 1; }

                // Overall winner for this round (majority)
                let round_score = (if time_winner == "TMT" {1} else {0}) + (if tput_winner == "TMT" {1} else {0}) + (if mem_winner == "TMT" {1} else {0});
                let overall_winner = if round_score >= 2 { "TMT" } else { "Merkle" };

                println!("{:<10} | {:<12.3} | {:<12.3} | {:<15.2} | {:<15.2} | {:<12.2} | {:<12.2} | {:<15}",
                         size, t.avg_time_ms, m.avg_time_ms, t.throughput_ops_per_sec, m.throughput_ops_per_sec,
                         t.memory_usage_mb, m.memory_usage_mb, overall_winner);
            }
        }
    }

    println!("\n🏆 Overall Scores");
    println!("TMT: {} points", tmt_score);
    println!("Merkle: {} points", merkle_score);
    let grand_winner = if tmt_score > merkle_score { "TMT" } else if merkle_score > tmt_score { "Merkle" } else { "Tie" };
    println!("Grand Winner: {}", grand_winner);
}

// Main function to run all benchmarks
fn main() {
    let config = BenchmarkConfig::default();
    let mut all_results = Vec::new();

    println!("==================================================");
    println!("      TMT vs Merkle Tree Benchmark Suite");
    println!("==================================================");
    println!("Configuration:");
    println!("  - Data Sizes: {:?}", config.data_sizes);
    println!("  - Iterations per test: {}", config.iterations);
    println!("  - Concurrent Threads: {:?}", config.thread_counts);
    println!("==================================================");

    // Run benchmarks
    all_results.extend(benchmark_tmt_operations(&config));
    all_results.extend(benchmark_merkle_operations(&config));
    all_results.extend(benchmark_concurrent_operations(&config));
    all_results.extend(stress_test(&config));

    // Print tables and comparisons
    print_tmt_table(&all_results);
    print_merkle_table(&all_results);
    print_comparisons(&all_results);
}