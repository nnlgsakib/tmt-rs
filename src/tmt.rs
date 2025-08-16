use blake3::Hasher;
use serde::{Deserialize, Serialize};
use bincode::{Encode, Decode};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::fmt;
use std::time::Instant;
use std::thread;

pub type Hash = [u8; 32];
pub type NodeId = u64;

// Efficient hash utilities
pub fn compute_hash(data: &[u8]) -> Hash {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finalize().into()
}

pub fn combine_hashes(hashes: &[Hash]) -> Hash {
    let mut hasher = Hasher::new();
    for hash in hashes {
        hasher.update(hash);
    }
    hasher.finalize().into()
}

pub fn hash_to_hex(hash: &Hash) -> String {
    hex::encode(hash)
}

// Compact serializable node representation
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct SerializableNode {
    pub id: NodeId,
    pub hash: Hash,
    pub children: Vec<NodeId>, // Empty for leaves, up to 3 for internal nodes
    pub is_leaf: bool,
    pub parent: Option<NodeId>,
}

// Internal node structure optimized for memory
#[derive(Debug, Clone)]
struct InternalNode {
    hash: Hash,
    children: Vec<NodeId>, // Dynamic size: 0 for leaves, 1-3 for internal
    is_leaf: bool,
    parent: Option<NodeId>,
}

// Compact verification proof
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct VerificationProof {
    pub leaf_index: usize,
    pub sibling_hashes: Vec<(usize, Hash)>, // (position, hash) pairs
    pub path_length: usize,
}

// Performance metrics
#[derive(Debug, Default, Clone)]
pub struct Metrics {
    pub build_time_ms: u128,
    pub last_verification_time_ns: u128,
    pub last_update_time_ns: u128,
    pub total_verifications: u64,
    pub total_updates: u64,
    pub memory_usage_bytes: usize,
}

// Configuration for the TMT
#[derive(Debug, Clone)]
pub struct TmtConfig {
    pub enable_caching: bool,
    pub max_cache_size: usize,
    pub enable_metrics: bool,
    pub parallel_threshold: usize,
}

impl Default for TmtConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cache_size: 10000,
            enable_metrics: true,
            parallel_threshold: 1000,
        }
    }
}

// Thread-safe, production-ready Ternary Mesh Tree
pub struct TernaryMeshTree {
    nodes: Arc<RwLock<Vec<InternalNode>>>, // Compact vector storage
    leaf_data: Arc<RwLock<Vec<Vec<u8>>>>,  // Original leaf data
    root_id: Option<NodeId>,
    leaf_count: usize,
    config: TmtConfig,
    metrics: Arc<RwLock<Metrics>>,
    hash_cache: Arc<RwLock<HashMap<Vec<u8>, Hash>>>, // Optional caching
}

impl TernaryMeshTree {
    pub fn new(config: TmtConfig) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(Vec::new())),
            leaf_data: Arc::new(RwLock::new(Vec::new())),
            root_id: None,
            leaf_count: 0,
            config,
            metrics: Arc::new(RwLock::new(Metrics::default())),
            hash_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(TmtConfig::default())
    }

    // Build tree from data blocks with optimized algorithm
    pub fn build(&mut self, data_blocks: Vec<Vec<u8>>) -> Result<(), TmtError> {
        let start_time = Instant::now();

        if data_blocks.is_empty() {
            return Err(TmtError::EmptyData);
        }

        // Clear previous state
        {
            let mut nodes = self.nodes.write().unwrap();
            let mut leaf_data = self.leaf_data.write().unwrap();
            nodes.clear();
            leaf_data.clear();
        }

        self.leaf_count = data_blocks.len();

        // Build leaf nodes efficiently
        let mut current_level: Vec<NodeId> = Vec::new();
        {
            let mut nodes = self.nodes.write().unwrap();
            let mut leaf_data = self.leaf_data.write().unwrap();

            for (index, data) in data_blocks.into_iter().enumerate() {
                let hash = if self.config.enable_caching {
                    self.get_cached_hash(&data)
                } else {
                    compute_hash(&data)
                };

                let node = InternalNode {
                    hash,
                    children: Vec::new(),
                    is_leaf: true,
                    parent: None,
                };

                nodes.push(node);
                leaf_data.push(data);
                current_level.push(index as NodeId);
            }
        }

        // Pad to make divisible by 3
        while current_level.len() % 3 != 0 {
            let mut nodes = self.nodes.write().unwrap();
            let mut leaf_data = self.leaf_data.write().unwrap();

            let padding_data = vec![0u8; 0]; // Empty padding
            let hash = compute_hash(&padding_data);

            let node = InternalNode {
                hash,
                children: Vec::new(),
                is_leaf: true,
                parent: None,
            };

            nodes.push(node);
            leaf_data.push(padding_data);
            current_level.push((nodes.len() - 1) as NodeId);
        }

        // Build tree bottom-up
        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in current_level.chunks(3) {
                let child_hashes: Vec<Hash> = {
                    let nodes = self.nodes.read().unwrap();
                    chunk.iter().map(|&id| nodes[id as usize].hash).collect()
                };

                let parent_hash = combine_hashes(&child_hashes);
                let parent_node = InternalNode {
                    hash: parent_hash,
                    children: chunk.to_vec(),
                    is_leaf: false,
                    parent: None,
                };

                let mut nodes = self.nodes.write().unwrap();
                let parent_id = nodes.len() as NodeId;
                nodes.push(parent_node);
                for &child_id in chunk {
                    nodes[child_id as usize].parent = Some(parent_id);
                }
                next_level.push(parent_id);
            }

            current_level = next_level;
        }

        self.root_id = current_level.first().copied();

        // Update metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().unwrap();
            metrics.build_time_ms = start_time.elapsed().as_millis();
            metrics.memory_usage_bytes = self.estimate_memory_usage();
        }

        Ok(())
    }

    // Optimized verification with compact proofs
    pub fn verify(&self, leaf_index: usize, data: &[u8]) -> Result<bool, TmtError> {
        let start_time = Instant::now();

        if leaf_index >= self.leaf_count {
            return Err(TmtError::InvalidIndex(leaf_index));
        }

        let root_id = self.root_id.ok_or(TmtError::UninitializedTree)?;
        let nodes = self.nodes.read().unwrap();

        // Verify leaf hash
        let expected_hash = compute_hash(data);
        if nodes[leaf_index].hash != expected_hash {
            return Ok(false);
        }

        // Generate and verify proof
        let proof = self.generate_proof_internal(leaf_index, &nodes)?;
        let result = self.verify_proof_internal(&proof, expected_hash, root_id, &nodes);

        // Update metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().unwrap();
            metrics.last_verification_time_ns = start_time.elapsed().as_nanos();
            metrics.total_verifications += 1;
        }

        Ok(result)
    }

    // Thread-safe update operation
    pub fn update(&mut self, leaf_index: usize, new_data: Vec<u8>) -> Result<(), TmtError> {
        let start_time = Instant::now();

        if leaf_index >= self.leaf_count {
            return Err(TmtError::InvalidIndex(leaf_index));
        }

        // Update leaf data and hash
        {
            let mut leaf_data = self.leaf_data.write().unwrap();
            let mut nodes = self.nodes.write().unwrap();

            leaf_data[leaf_index] = new_data.clone();
            nodes[leaf_index].hash = compute_hash(&new_data);
        }

        // Propagate changes up the tree
        self.update_ancestors(leaf_index as NodeId)?;

        // Update metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().unwrap();
            metrics.last_update_time_ns = start_time.elapsed().as_nanos();
            metrics.total_updates += 1;
        }

        Ok(())
    }

    // Batch update for efficiency
    pub fn batch_update(&mut self, updates: Vec<(usize, Vec<u8>)>) -> Result<(), TmtError> {
        let start_time = Instant::now();

        // Validate all indices first
        for (index, _) in &updates {
            if *index >= self.leaf_count {
                return Err(TmtError::InvalidIndex(*index));
            }
        }

        // Apply all leaf updates
        {
            let mut leaf_data = self.leaf_data.write().unwrap();
            let mut nodes = self.nodes.write().unwrap();

            for (index, new_data) in &updates {
                leaf_data[*index] = new_data.clone();
                nodes[*index].hash = compute_hash(new_data);
            }
        }

        // Collect all affected ancestors and update them
        let mut affected_nodes = std::collections::HashSet::new();
        for (index, _) in &updates {
            self.collect_ancestors(*index as NodeId, &mut affected_nodes)?;
        }

        // Update affected internal nodes
        for &node_id in &affected_nodes {
            self.recompute_node_hash(node_id)?;
        }

        // Update metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().unwrap();
            metrics.last_update_time_ns = start_time.elapsed().as_nanos();
            metrics.total_updates += updates.len() as u64;
        }

        Ok(())
    }

    // Generate compact verification proof
    pub fn generate_proof(&self, leaf_index: usize) -> Result<VerificationProof, TmtError> {
        if leaf_index >= self.leaf_count {
            return Err(TmtError::InvalidIndex(leaf_index));
        }

        let nodes = self.nodes.read().unwrap();
        self.generate_proof_internal(leaf_index, &nodes)
    }

    // Verify proof without accessing the tree
    pub fn verify_proof(&self, proof: &VerificationProof, leaf_data: &[u8]) -> Result<bool, TmtError> {
        let root_id = self.root_id.ok_or(TmtError::UninitializedTree)?;
        let nodes = self.nodes.read().unwrap();
        let leaf_hash = compute_hash(leaf_data);

        Ok(self.verify_proof_internal(proof, leaf_hash, root_id, &nodes))
    }

    // Serialize tree for network transmission
    pub fn serialize(&self) -> Result<Vec<u8>, TmtError> {
        let nodes = self.nodes.read().unwrap();
        let leaf_data = self.leaf_data.read().unwrap();

        let serializable_nodes: Vec<SerializableNode> = nodes.iter().enumerate().map(|(id, node)| {
            SerializableNode {
                id: id as NodeId,
                hash: node.hash,
                children: node.children.clone(),
                is_leaf: node.is_leaf,
                parent: node.parent,
            }
        }).collect();

        let tree_data = (serializable_nodes, leaf_data.clone(), self.root_id, self.leaf_count);
        bincode::encode_to_vec(tree_data, bincode::config::standard()).map_err(|e| TmtError::SerializationError(e.to_string()))
    }

    // Deserialize tree from network data
    pub fn deserialize(data: &[u8], config: TmtConfig) -> Result<Self, TmtError> {
        let ((serializable_nodes, leaf_data, root_id, leaf_count), _): ((Vec<SerializableNode>, Vec<Vec<u8>>, Option<NodeId>, usize), usize) =
            bincode::decode_from_slice(data, bincode::config::standard()).map_err(|e| TmtError::SerializationError(e.to_string()))?;

        let internal_nodes: Vec<InternalNode> = serializable_nodes.into_iter().map(|node| {
            InternalNode {
                hash: node.hash,
                children: node.children,
                is_leaf: node.is_leaf,
                parent: node.parent,
            }
        }).collect();

        Ok(Self {
            nodes: Arc::new(RwLock::new(internal_nodes)),
            leaf_data: Arc::new(RwLock::new(leaf_data)),
            root_id,
            leaf_count,
            config,
            metrics: Arc::new(RwLock::new(Metrics::default())),
            hash_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    // Get performance metrics
    pub fn get_metrics(&self) -> Metrics {
        if self.config.enable_metrics {
            self.metrics.read().unwrap().clone()
        } else {
            Metrics::default()
        }
    }

    // Get root hash
    pub fn get_root_hash(&self) -> Option<Hash> {
        self.root_id.map(|id| {
            let nodes = self.nodes.read().unwrap();
            nodes[id as usize].hash
        })
    }

    // Get tree height
    pub fn get_height(&self) -> usize {
        if let Some(root_id) = self.root_id {
            let nodes = self.nodes.read().unwrap();
            self.calculate_height(root_id, &nodes)
        } else {
            0
        }
    }

    // Private helper methods
    fn get_cached_hash(&self, data: &[u8]) -> Hash {
        if !self.config.enable_caching {
            return compute_hash(data);
        }

        {
            let cache = self.hash_cache.read().unwrap();
            if let Some(&hash) = cache.get(data) {
                return hash;
            }
        }

        let hash = compute_hash(data);

        {
            let mut cache = self.hash_cache.write().unwrap();
            if cache.len() < self.config.max_cache_size {
                cache.insert(data.to_vec(), hash);
            }
        }

        hash
    }

    fn update_ancestors(&self, mut current_id: NodeId) -> Result<(), TmtError> {
        let nodes = self.nodes.read().unwrap();
        let mut ancestors = Vec::new();

        loop {
            if let Some(parent_id) = nodes[current_id as usize].parent {
                ancestors.push(parent_id);
                current_id = parent_id;
            } else {
                break;
            }
        }

        drop(nodes);

        // Update ancestors from bottom to top
        for &ancestor_id in ancestors.iter().rev() {
            self.recompute_node_hash(ancestor_id)?;
        }

        Ok(())
    }

    fn collect_ancestors(&self, mut current_id: NodeId, affected: &mut std::collections::HashSet<NodeId>) -> Result<(), TmtError> {
        let nodes = self.nodes.read().unwrap();

        loop {
            if let Some(parent_id) = nodes[current_id as usize].parent {
                affected.insert(parent_id);
                current_id = parent_id;
            } else {
                break;
            }
        }

        Ok(())
    }

    fn recompute_node_hash(&self, node_id: NodeId) -> Result<(), TmtError> {
        let child_hashes = {
            let nodes = self.nodes.read().unwrap();
            if node_id as usize >= nodes.len() {
                return Err(TmtError::InvalidIndex(node_id as usize));
            }

            let node = &nodes[node_id as usize];
            if node.is_leaf {
                return Ok(()); // Leaf nodes don't need recomputation
            }

            node.children.iter().map(|&child_id| nodes[child_id as usize].hash).collect::<Vec<_>>()
        };

        let new_hash = combine_hashes(&child_hashes);

        {
            let mut nodes = self.nodes.write().unwrap();
            nodes[node_id as usize].hash = new_hash;
        }

        Ok(())
    }

    fn generate_proof_internal(&self, leaf_index: usize, nodes: &[InternalNode]) -> Result<VerificationProof, TmtError> {
        let mut sibling_hashes = Vec::new();
        let mut current_id = leaf_index as NodeId;
        let mut path_length = 0;

        loop {
            if let Some(parent_id) = nodes[current_id as usize].parent {
                let parent = &nodes[parent_id as usize];
                let child_position = parent.children.iter().position(|&id| id == current_id).unwrap();

                // Add sibling hashes
                for (pos, &sibling_id) in parent.children.iter().enumerate() {
                    if pos != child_position {
                        sibling_hashes.push((pos, nodes[sibling_id as usize].hash));
                    }
                }

                current_id = parent_id;
                path_length += 1;
            } else {
                break;
            }
        }

        Ok(VerificationProof {
            leaf_index,
            sibling_hashes,
            path_length,
        })
    }

    fn verify_proof_internal(&self, proof: &VerificationProof, mut current_hash: Hash, root_id: NodeId, nodes: &[InternalNode]) -> bool {
        let mut current_id = proof.leaf_index as NodeId;
        let mut sibling_iter = proof.sibling_hashes.iter();

        for _ in 0..proof.path_length {
            if let Some(parent_id) = nodes[current_id as usize].parent {
                let parent = &nodes[parent_id as usize];
                let child_position = parent.children.iter().position(|&id| id == current_id).unwrap();

                // Reconstruct parent hash
                let mut child_hashes = vec![Hash::default(); parent.children.len()];
                child_hashes[child_position] = current_hash;

                // Fill in sibling hashes from the proof
                let num_siblings = parent.children.len() - 1;
                for _ in 0..num_siblings {
                    if let Some(&(pos, hash)) = sibling_iter.next() {
                        // Ensure the position from the proof is not the current child's position
                        if pos != child_position && pos < child_hashes.len() {
                            child_hashes[pos] = hash;
                        } else {
                            // Invalid proof format
                            return false;
                        }
                    } else {
                        // Proof is missing hashes
                        return false;
                    }
                }

                current_hash = combine_hashes(&child_hashes);
                current_id = parent_id;
            } else {
                // Could not find parent, proof is invalid
                return false;
            }
        }

        // After iterating through the path, the calculated hash should match the root hash
        // and all sibling hashes from the proof should have been consumed.
        current_hash == nodes[root_id as usize].hash && sibling_iter.next().is_none()
    }

    fn calculate_height(&self, node_id: NodeId, nodes: &[InternalNode]) -> usize {
        let node = &nodes[node_id as usize];
        if node.is_leaf {
            return 1;
        }

        let max_child_height = node.children.iter()
            .map(|&child_id| self.calculate_height(child_id, nodes))
            .max()
            .unwrap_or(0);

        max_child_height + 1
    }

    fn estimate_memory_usage(&self) -> usize {
        let nodes = self.nodes.read().unwrap();
        let leaf_data = self.leaf_data.read().unwrap();

        let node_size = std::mem::size_of::<InternalNode>();
        let nodes_memory = nodes.len() * node_size;

        let data_memory = leaf_data.iter().map(|d| d.len()).sum::<usize>();

        nodes_memory + data_memory
    }
}

// Error handling
#[derive(Debug, Clone)]
pub enum TmtError {
    EmptyData,
    InvalidIndex(usize),
    UninitializedTree,
    SerializationError(String),
    ConcurrencyError,
}

impl fmt::Display for TmtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TmtError::EmptyData => write!(f, "Cannot build tree from empty data"),
            TmtError::InvalidIndex(idx) => write!(f, "Invalid index: {}", idx),
            TmtError::UninitializedTree => write!(f, "Tree is not initialized"),
            TmtError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            TmtError::ConcurrencyError => write!(f, "Concurrency error occurred"),
        }
    }
}

impl std::error::Error for TmtError {}

// Thread-safe implementation
unsafe impl Send for TernaryMeshTree {}
unsafe impl Sync for TernaryMeshTree {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut tmt = TernaryMeshTree::with_default_config();
        let data = vec![
            b"block1".to_vec(),
            b"block2".to_vec(),
            b"block3".to_vec(),
        ];

        assert!(tmt.build(data.clone()).is_ok());
        assert!(tmt.verify(0, b"block1").unwrap());
        assert!(tmt.update(0, b"new_block1".to_vec()).is_ok());
        assert!(tmt.verify(0, b"new_block1").unwrap());
    }

    #[test]
    fn test_serialization() {
        let mut tmt = TernaryMeshTree::with_default_config();
        let data = vec![b"test1".to_vec(), b"test2".to_vec()];

        tmt.build(data).unwrap();
        let serialized = tmt.serialize().unwrap();
        let deserialized = TernaryMeshTree::deserialize(&serialized, TmtConfig::default()).unwrap();

        assert_eq!(tmt.get_root_hash(), deserialized.get_root_hash());
    }
}