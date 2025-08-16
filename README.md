# Ternary Mesh Tree (TMT)

## Introduction

The Ternary Mesh Tree (TMT) is a specialized cryptographic hash tree designed for high-performance data integrity verification. It organizes data into a ternary (3-ary) tree structure, where each node is a hash of its children. The primary purpose of a TMT is to create a single, compact cryptographic fingerprint (the root hash) for a large dataset, allowing for efficient and secure verification of individual data blocks without needing to access the entire dataset.

Its primary use cases include blockchain systems, distributed storage networks, and any application requiring verifiable data integrity, where performance, concurrency, and compact proofs are critical.

## Core Concept

The TMT algorithm operates by constructing a tree from the bottom up.

1.  **Leaf Nodes**: The process begins with a list of data blocks. Each data block is hashed to create a "leaf node."
2.  **Internal Nodes**: The leaf nodes are then grouped into sets of three. The hashes of these three nodes are combined and hashed again to create a "parent" or "internal" node.
3.  **Tree Construction**: This process is repeated, moving up the tree level by level. The nodes of the current level are grouped by three and hashed to form the parents of the next level.
4.  **Padding**: If the number of nodes at any level is not a multiple of three, the level is padded with empty nodes until it is. This ensures the tree remains a balanced ternary tree.
5.  **Root Hash**: The process continues until only a single node remains: the **root**. This root hash serves as a unique and secure identifier for the entire dataset. Any change to any data block will result in a different root hash.

## Motivation

The TMT was designed to provide an efficient and flexible alternative to traditional binary Merkle trees. While binary trees are foundational, a ternary structure presents a different set of trade-offs that can be advantageous.

-   **Problem Solved**: It provides a mechanism to prove that a specific piece of data is part of a larger set without needing to have the entire set. This is crucial for decentralized and distributed systems where bandwidth and storage are constraints.
-   **Advantages over Binary Trees**: A ternary tree is wider and shallower than a binary tree for the same number of leaves (`height ≈ log3(N)` vs. `log2(N)`). This can lead to shorter verification paths and more compact proofs, potentially reducing the amount of data needed for verification.

## Key Features

-   **High Efficiency**: Utilizes the `blake3` hash function for high-speed hashing and a bottom-up build algorithm for optimized tree construction.
-   **Thread-Safety**: Designed for concurrent applications. The core data structures are protected by `Arc<RwLock>`, allowing multiple threads to read data and generate proofs in parallel while ensuring safe, exclusive access for updates.
-   **Compact Proofs**: The ternary structure can result in shallower trees, leading to more compact verification proofs compared to binary trees for the same amount of data.
-   **Configurability**: Allows users to tune performance through configuration parameters, including caching, metrics collection, and thresholds for parallel execution.
-   **Batch Operations**: Supports efficient batch updates, which apply multiple changes and recompute the necessary parts of the tree in a single, optimized pass.

## Mathematical Foundation

TMT is built on fundamental cryptographic principles:

-   **Cryptographic Hash Functions**: It uses `blake3`, a one-way function that maps data of any size to a fixed-size output (a hash). It is computationally infeasible to find two different inputs that produce the same hash (collision resistance) or to derive the input from its hash (pre-image resistance).
-   **Tree Balancing**: To maintain a consistent structure, the tree is always balanced. If a level does not have a number of nodes divisible by three, it is padded with zero-value nodes. This ensures that the path from any leaf to the root has a predictable length.
-   **Proof Generation and Verification**: Verification relies on the "Merkle proof" concept. To verify a data block, a client needs only the block itself, the root hash, and a small set of "sibling" hashes along the path to the root. The client recomputes the hashes up the tree and checks if the final calculated root matches the known root hash. This confirms the data's integrity and inclusion in the tree.

## Algorithm Details

-   **Build**: Constructs the tree from a vector of data blocks in a bottom-up fashion. It first creates all leaf nodes, then iteratively hashes groups of three nodes to create parent levels until a single root is achieved.
-   **Verify**: To verify a data block, its hash is recomputed. Then, using a `VerificationProof` (which contains the necessary sibling hashes), the root hash is recalculated. If it matches the tree's root hash, the data is valid.
-   **Update**: When a leaf node's data is changed, its hash is recomputed. This change is then propagated up the tree by re-hashing all direct ancestors up to the root.
-   **Batch Update**: An optimized method to update multiple leaves at once. It first updates all specified leaves, then identifies the set of unique internal nodes that are affected and recomputes their hashes in a single pass, which is more efficient than performing individual updates.
-   **Serialize/Deserialize**: The entire tree can be serialized into a compact binary format using `bincode` for network transmission or persistent storage, and then deserialized back into a functional TMT instance.

## Performance Optimizations

-   **Caching**: An optional hash cache stores the results of hash computations. If the same data block is processed multiple times, its hash can be retrieved from the cache instead of being recomputed.
-   **Parallel Processing**: The design includes a `parallel_threshold` configuration, enabling the potential for parallelizing operations (like the build process) on large datasets to leverage multi-core processors.
-   **Memory Efficiency**: The tree's nodes are stored in a contiguous `Vec`, which is more memory-efficient than pointer-based node structures, leading to better cache locality and lower overhead.

## Comparison

| Feature            | Ternary Mesh Tree (TMT)                               | Binary Merkle Tree                                    |
| ------------------ | ----------------------------------------------------- | ----------------------------------------------------- |
| **Fanout**         | 3 (each internal node has up to 3 children)           | 2 (each internal node has up to 2 children)           |
| **Tree Height**    | Shallower (`~log3(N)`)                                | Deeper (`~log2(N)`)                                   |
| **Proof Size**     | Potentially smaller (fewer levels)                    | Potentially larger (more levels)                      |
| **Verification**   | Combines 3 hashes at each step                        | Combines 2 hashes at each step                        |
| **Flexibility**    | Offers a different performance profile for specific loads | The standard, widely-used model                       |

## Use Cases

TMT is well-suited for applications where data integrity and performance are paramount:

-   **Blockchain Technology**: Verifying the inclusion of transactions in a block. The root hash is stored in the block header.
-   **Data Integrity Verification**: Ensuring that large files or datasets (e.g., software distributions, backups) have not been corrupted or tampered with.
-   **Distributed Systems**: Verifying the consistency of data chunks across different nodes in a distributed storage system like IPFS.

## Configuration Options

The TMT's behavior can be fine-tuned via the `TmtConfig` struct:

-   `enable_caching`: Enables or disables the hash cache.
-   `max_cache_size`: Sets the maximum number of entries in the hash cache.
-   `enable_metrics`: Enables or disables the collection of performance metrics.
-   `parallel_threshold`: Defines the minimum number of items required to justify spawning parallel threads for processing.

## Error Handling

The algorithm implements a robust `TmtError` enum to handle potential issues, including:
-   `EmptyData`: Attempting to build a tree from no data.
-   `InvalidIndex`: Accessing a leaf with an out-of-bounds index.
-   `UninitializedTree`: Performing operations on a tree that has not been built.
-   `SerializationError`: An error occurred during serialization or deserialization.

## Thread Safety

Thread safety is a core design feature, achieved using `Arc<RwLock<T>>`. This allows any number of threads to read from the tree concurrently (e.g., performing verifications). When a write operation (e.g., `update`) is needed, the lock ensures exclusive access, preventing data races and ensuring consistency.

## Serialization

The `serialize` and `deserialize` methods allow the TMT to be easily saved and loaded. The process uses `bincode` to convert the tree's nodes, leaf data, and metadata into a single, compact byte vector, which is ideal for storage or sending over a network.

## Metrics and Monitoring

When enabled, the TMT collects valuable performance metrics, including:
-   Build time.
-   Verification and update times.
-   Total number of operations performed.
-   Estimated memory usage.
These metrics are essential for debugging, performance tuning, and monitoring the health of the system.

## Limitations

-   **Memory Usage**: The current implementation holds the entire tree in memory, which may be a constraint for extremely large datasets that exceed available RAM.
-   **Update Cost**: While updates are efficient, each update requires re-hashing all ancestors up to the root. For write-heavy workloads, this can become a bottleneck.
-   **Padding Overhead**: The need to pad levels to a multiple of three adds a small amount of computational and memory overhead.

## Future Improvements

-   **Disk-Backed Storage**: A version of TMT that uses a disk-backed key-value store to hold nodes would allow for trees that are much larger than the available system memory.
-   **Pruning**: Implementing logic to prune historical or unnecessary branches of the tree to reduce its memory footprint over time.
-   **Advanced Parallelism**: Further leveraging libraries like `rayon` to more aggressively parallelize tree construction and updates.

## Getting Started

To use the TMT:
1.  Create a `TmtConfig` or use the default.
2.  Instantiate a `TernaryMeshTree` with the configuration.
3.  Prepare your data as a `Vec<Vec<u8>>`.
4.  Call the `.build()` method with your data to construct the tree.
5.  You can now get the root hash, verify data blocks, generate proofs, or update leaves.

## Dependencies

The implementation relies on the following external Rust crates:
-   `blake3`: For a fast, modern cryptographic hash function.
-   `serde`: For serialization and deserialization of data structures.
-e   `bincode`: For a compact binary encoding format.
-   `hex`: For converting hash bytes to hexadecimal strings.

## Testing

The TMT includes a suite of unit tests to ensure correctness and reliability. The tests cover core operations like building, updating, verifying, and serialization/deserialization.

# Benchmark
# 🔍 Performance Comparisons and Scoring

**Scoring**: +1 for better Avg Time (lower), Throughput (higher), Memory (lower) per data size.

## Build Comparison

| Data Size | TMT Time | Merkle Time | TMT TPut | Merkle TPut | TMT Mem | Merkle Mem | Winner |
|-----------|----------|-------------|----------|-------------|---------|------------|--------|
| 100       | 0.161    | 0.146       | 622742.56 | 683246.79  | 0.01    | 0.02       | Merkle |
| 1000      | 1.558    | 1.614       | 641782.62 | 619402.15  | 0.15    | 0.17       | TMT    |
| 10000     | 14.983   | 15.173      | 667442.68 | 659085.43  | 1.45    | 1.68       | TMT    |
| 50000     | 73.228   | 72.110      | 682803.41 | 693382.41  | 7.25    | 8.39       | Merkle |

## Verify Comparison

| Data Size | TMT Time | Merkle Time | TMT TPut | Merkle TPut | TMT Mem | Merkle Mem | Winner |
|-----------|----------|-------------|----------|-------------|---------|------------|--------|
| 100       | 0.005    | 0.013       | 207468.88 | 75757.58   | 0.01    | 0.02       | TMT    |
| 1000      | 0.006    | 0.153       | 167224.08 | 6533.39    | 0.15    | 0.17       | TMT    |
| 10000     | 0.008    | 2.106       | 130890.05 | 474.80     | 1.45    | 1.68       | TMT    |
| 50000     | 0.009    | 12.168      | 114678.90 | 82.19      | 7.25    | 8.39       | TMT    |

## Update Comparison

| Data Size | TMT Time | Merkle Time | TMT TPut | Merkle TPut | TMT Mem | Merkle Mem | Winner |
|-----------|----------|-------------|----------|-------------|---------|------------|--------|
| 100       | 0.004    | 0.014       | 245098.04 | 73964.50   | 0.01    | 0.02       | TMT    |
| 1000      | 0.006    | 0.152       | 177935.94 | 6598.05    | 0.15    | 0.17       | TMT    |
| 10000     | 0.007    | 2.048       | 142857.14 | 488.16     | 1.45    | 1.68       | TMT    |
| 50000     | 0.008    | 12.025      | 126262.63 | 83.16      | 7.25    | 8.39       | TMT    |

## Stress_Build Comparison

| Data Size | TMT Time | Merkle Time | TMT TPut | Merkle TPut | TMT Mem | Merkle Mem | Winner |
|-----------|----------|-------------|----------|-------------|---------|------------|--------|
| 100000    | 155.454  | 149.977     | 643277.11 | 666770.68  | 14.50   | 16.79      | Merkle |
| 500000    | 569.794  | 531.099     | 877509.81 | 941443.53  | 72.48   | 83.92      | Merkle |

## 🏆 Overall Scores

- **TMT**: 34 points
- **Merkle**: 8 points
- **Grand Winner**: TMT
## Conclusion

The Ternary Mesh Tree is a powerful, high-performance, and thread-safe data structure for ensuring data integrity. By using a ternary structure, it offers a compelling alternative to traditional binary Merkle trees, with potential advantages in proof compactness and verification speed. Its configurability and robust design make it an excellent choice for demanding applications in blockchain, distributed storage, and beyond.
