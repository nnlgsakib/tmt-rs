use blake3::Hasher;
use serde::{Deserialize, Serialize};
use bincode::{Encode, Decode};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

pub type Hash = [u8; 32];
pub type NodeId = u64;

// Hash utilities
pub fn compute_hash(data: &[u8]) -> Hash {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finalize().into()
}

pub fn combine_two_hashes(left: &Hash, right: &Hash) -> Hash {
    let mut hasher = Hasher::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

pub fn hash_to_hex(hash: &Hash) -> String {
    hex::encode(hash)
}

// Merkle node
#[derive(Debug, Clone)]
struct MerkleNode {
    hash: Hash,
    left: Option<NodeId>,
    right: Option<NodeId>,
    is_leaf: bool,
}

// Merkle proof
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct MerkleProof {
    pub leaf_index: usize,
    pub sibling_hashes: Vec<(bool, Hash)>, // (is_right_sibling, hash)
    pub root_hash: Hash,
}

// Performance metrics
#[derive(Debug, Default, Clone)]
pub struct MerkleMetrics {
    pub build_time_ms: u128,
    pub last_verification_time_ns: u128,
    pub last_update_time_ns: u128,
    pub total_verifications: u64,
    pub total_updates: u64,
    pub memory_usage_bytes: usize,
}

// Configuration
#[derive(Debug, Clone)]
pub struct MerkleConfig {
    pub enable_caching: bool,
    pub max_cache_size: usize,
    pub enable_metrics: bool,
}

impl Default for MerkleConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cache_size: 10000,
            enable_metrics: true,
        }
    }
}

// Standard binary Merkle tree
pub struct MerkleTree {
    nodes: Arc<RwLock<Vec<MerkleNode>>>,
    leaf_data: Arc<RwLock<Vec<Vec<u8>>>>,
    root_id: Option<NodeId>,
    leaf_count: usize,
    config: MerkleConfig,
    metrics: Arc<RwLock<MerkleMetrics>>,
    hash_cache: Arc<RwLock<HashMap<Vec<u8>, Hash>>>,
}

impl MerkleTree {
    pub fn new(config: MerkleConfig) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(Vec::new())),
            leaf_data: Arc::new(RwLock::new(Vec::new())),
            root_id: None,
            leaf_count: 0,
            config,
            metrics: Arc::new(RwLock::new(MerkleMetrics::default())),
            hash_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(MerkleConfig::default())
    }

    // Build tree from data blocks
    pub fn build(&mut self, data_blocks: Vec<Vec<u8>>) -> Result<(), MerkleError> {
        let start_time = Instant::now();

        if data_blocks.is_empty() {
            return Err(MerkleError::EmptyData);
        }

        // Clear previous state
        {
            let mut nodes = self.nodes.write().unwrap();
            let mut leaf_data = self.leaf_data.write().unwrap();
            nodes.clear();
            leaf_data.clear();
        }

        self.leaf_count = data_blocks.len();

        // Create leaf nodes
        let mut current_level: Vec<NodeId> = Vec::new();
        {
            let mut nodes = self.nodes.write().unwrap();
            let mut leaf_data = self.leaf_data.write().unwrap();

            for data in data_blocks {
                let hash = if self.config.enable_caching {
                    self.get_cached_hash(&data)
                } else {
                    compute_hash(&data)
                };

                let node = MerkleNode {
                    hash,
                    left: None,
                    right: None,
                    is_leaf: true,
                };

                let node_id = nodes.len() as NodeId;
                nodes.push(node);
                leaf_data.push(data);
                current_level.push(node_id);
            }
        }

        // Build tree bottom-up
        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            // Process pairs
            for pair in current_level.chunks(2) {
                let left_id = pair[0];
                let right_id = if pair.len() > 1 { Some(pair[1]) } else { None };

                let (left_hash, right_hash) = {
                    let nodes = self.nodes.read().unwrap();
                    let left_hash = nodes[left_id as usize].hash;
                    let right_hash = if let Some(right_id) = right_id {
                        nodes[right_id as usize].hash
                    } else {
                        left_hash // Duplicate for odd number of nodes
                    };
                    (left_hash, right_hash)
                };

                let parent_hash = combine_two_hashes(&left_hash, &right_hash);
                let parent_node = MerkleNode {
                    hash: parent_hash,
                    left: Some(left_id),
                    right: right_id,
                    is_leaf: false,
                };

                let mut nodes = self.nodes.write().unwrap();
                let parent_id = nodes.len() as NodeId;
                nodes.push(parent_node);
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

    // Verify data with proof
    pub fn verify(&self, leaf_index: usize, data: &[u8]) -> Result<bool, MerkleError> {
        let start_time = Instant::now();

        if leaf_index >= self.leaf_count {
            return Err(MerkleError::InvalidIndex(leaf_index));
        }

        let nodes = self.nodes.read().unwrap();
        let expected_hash = compute_hash(data);

        // Check if leaf hash matches
        if nodes[leaf_index].hash != expected_hash {
            return Ok(false);
        }

        // Generate and verify proof
        let proof = self.generate_proof_internal(leaf_index, &nodes)?;
        let result = Self::verify_proof_static(&proof, data);

        // Update metrics
        if self.config.enable_metrics {
            let mut metrics = self.metrics.write().unwrap();
            metrics.last_verification_time_ns = start_time.elapsed().as_nanos();
            metrics.total_verifications += 1;
        }

        Ok(result)
    }

    // Update leaf data
    pub fn update(&mut self, leaf_index: usize, new_data: Vec<u8>) -> Result<(), MerkleError> {
        let start_time = Instant::now();

        if leaf_index >= self.leaf_count {
            return Err(MerkleError::InvalidIndex(leaf_index));
        }

        // Update leaf data and hash
        {
            let mut leaf_data = self.leaf_data.write().unwrap();
            let mut nodes = self.nodes.write().unwrap();

            leaf_data[leaf_index] = new_data.clone();
            nodes[leaf_index].hash = compute_hash(&new_data);
        }

        // Update ancestors
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
    pub fn batch_update(&mut self, updates: Vec<(usize, Vec<u8>)>) -> Result<(), MerkleError> {
        let start_time = Instant::now();

        // Validate indices
        for (index, _) in &updates {
            if *index >= self.leaf_count {
                return Err(MerkleError::InvalidIndex(*index));
            }
        }

        // Apply leaf updates
        {
            let mut leaf_data = self.leaf_data.write().unwrap();
            let mut nodes = self.nodes.write().unwrap();

            for (index, new_data) in &updates {
                leaf_data[*index] = new_data.clone();
                nodes[*index].hash = compute_hash(new_data);
            }
        }

        // Collect and update affected ancestors
        let mut affected_nodes = std::collections::HashSet::new();
        for (index, _) in &updates {
            self.collect_ancestors(*index as NodeId, &mut affected_nodes);
        }

        // Update internal nodes
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

    // Generate proof for a leaf
    pub fn generate_proof(&self, leaf_index: usize) -> Result<MerkleProof, MerkleError> {
        if leaf_index >= self.leaf_count {
            return Err(MerkleError::InvalidIndex(leaf_index));
        }

        let nodes = self.nodes.read().unwrap();
        self.generate_proof_internal(leaf_index, &nodes)
    }

    // Static proof verification (can be used without tree instance)
    pub fn verify_proof_static(proof: &MerkleProof, data: &[u8]) -> bool {
        let mut current_hash = compute_hash(data);

        for &(is_right, sibling_hash) in &proof.sibling_hashes {
            current_hash = if is_right {
                combine_two_hashes(&current_hash, &sibling_hash)
            } else {
                combine_two_hashes(&sibling_hash, &current_hash)
            };
        }

        current_hash == proof.root_hash
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
        if self.leaf_count == 0 {
            return 0;
        }
        (self.leaf_count as f64).log2().ceil() as usize + 1
    }

    // Get performance metrics
    pub fn get_metrics(&self) -> MerkleMetrics {
        if self.config.enable_metrics {
            self.metrics.read().unwrap().clone()
        } else {
            MerkleMetrics::default()
        }
    }

    // Serialize tree
    pub fn serialize(&self) -> Result<Vec<u8>, MerkleError> {
        let nodes = self.nodes.read().unwrap();
        let leaf_data = self.leaf_data.read().unwrap();

        let serializable_nodes: Vec<SerializableMerkleNode> = nodes.iter().enumerate().map(|(id, node)| {
            SerializableMerkleNode {
                id: id as NodeId,
                hash: node.hash,
                left: node.left,
                right: node.right,
                is_leaf: node.is_leaf,
            }
        }).collect();

        let tree_data = (serializable_nodes, leaf_data.clone(), self.root_id, self.leaf_count);
        bincode::encode_to_vec(tree_data, bincode::config::standard()).map_err(|e| MerkleError::SerializationError(e.to_string()))
    }

    // Deserialize tree
    pub fn deserialize(data: &[u8], config: MerkleConfig) -> Result<Self, MerkleError> {
        let ((serializable_nodes, leaf_data, root_id, leaf_count), _): ((Vec<SerializableMerkleNode>, Vec<Vec<u8>>, Option<NodeId>, usize), usize) =
            bincode::decode_from_slice(data, bincode::config::standard()).map_err(|e| MerkleError::SerializationError(e.to_string()))?;

        let internal_nodes: Vec<MerkleNode> = serializable_nodes.into_iter().map(|node| {
            MerkleNode {
                hash: node.hash,
                left: node.left,
                right: node.right,
                is_leaf: node.is_leaf,
            }
        }).collect();

        Ok(Self {
            nodes: Arc::new(RwLock::new(internal_nodes)),
            leaf_data: Arc::new(RwLock::new(leaf_data)),
            root_id,
            leaf_count,
            config,
            metrics: Arc::new(RwLock::new(MerkleMetrics::default())),
            hash_cache: Arc::new(RwLock::new(HashMap::new())),
        })
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

    fn find_parent(&self, child_id: NodeId, nodes: &[MerkleNode]) -> Option<NodeId> {
        for (id, node) in nodes.iter().enumerate() {
            if !node.is_leaf {
                if node.left == Some(child_id) || node.right == Some(child_id) {
                    return Some(id as NodeId);
                }
            }
        }
        None
    }

    fn update_ancestors(&self, mut current_id: NodeId) -> Result<(), MerkleError> {
        let nodes = self.nodes.read().unwrap();
        let mut ancestors = Vec::new();

        while let Some(parent_id) = self.find_parent(current_id, &nodes) {
            ancestors.push(parent_id);
            current_id = parent_id;
        }

        drop(nodes);

        for &ancestor_id in &ancestors {
            self.recompute_node_hash(ancestor_id)?;
        }

        Ok(())
    }

    fn collect_ancestors(&self, mut current_id: NodeId, affected: &mut std::collections::HashSet<NodeId>) {
        let nodes = self.nodes.read().unwrap();

        while let Some(parent_id) = self.find_parent(current_id, &nodes) {
            affected.insert(parent_id);
            current_id = parent_id;
        }
    }

    fn recompute_node_hash(&self, node_id: NodeId) -> Result<(), MerkleError> {
        let (left_hash, right_hash) = {
            let nodes = self.nodes.read().unwrap();
            let node = &nodes[node_id as usize];

            if node.is_leaf {
                return Ok(());
            }

            let left_hash = node.left.map(|id| nodes[id as usize].hash).unwrap_or_default();
            let right_hash = node.right.map(|id| nodes[id as usize].hash).unwrap_or(left_hash);

            (left_hash, right_hash)
        };

        let new_hash = combine_two_hashes(&left_hash, &right_hash);

        {
            let mut nodes = self.nodes.write().unwrap();
            nodes[node_id as usize].hash = new_hash;
        }

        Ok(())
    }

    fn generate_proof_internal(&self, leaf_index: usize, nodes: &[MerkleNode]) -> Result<MerkleProof, MerkleError> {
        let mut sibling_hashes = Vec::new();
        let mut current_id = leaf_index as NodeId;

        while let Some(parent_id) = self.find_parent(current_id, nodes) {
            let parent = &nodes[parent_id as usize];

            if parent.left == Some(current_id) {
                // Current node is left child, sibling is right
                if let Some(right_id) = parent.right {
                    sibling_hashes.push((true, nodes[right_id as usize].hash));
                }
            } else {
                // Current node is right child, sibling is left
                if let Some(left_id) = parent.left {
                    sibling_hashes.push((false, nodes[left_id as usize].hash));
                }
            }

            current_id = parent_id;
        }

        let root_hash = self.get_root_hash().ok_or(MerkleError::UninitializedTree)?;

        Ok(MerkleProof {
            leaf_index,
            sibling_hashes,
            root_hash,
        })
    }

    fn estimate_memory_usage(&self) -> usize {
        let nodes = self.nodes.read().unwrap();
        let leaf_data = self.leaf_data.read().unwrap();

        let node_size = std::mem::size_of::<MerkleNode>();
        let nodes_memory = nodes.len() * node_size;
        let data_memory = leaf_data.iter().map(|d| d.len()).sum::<usize>();

        nodes_memory + data_memory
    }
}

// Serializable node for network transmission
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct SerializableMerkleNode {
    pub id: NodeId,
    pub hash: Hash,
    pub left: Option<NodeId>,
    pub right: Option<NodeId>,
    pub is_leaf: bool,
}

// Error types
#[derive(Debug, Clone)]
pub enum MerkleError {
    EmptyData,
    InvalidIndex(usize),
    UninitializedTree,
    SerializationError(String),
}

impl std::fmt::Display for MerkleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MerkleError::EmptyData => write!(f, "Cannot build tree from empty data"),
            MerkleError::InvalidIndex(idx) => write!(f, "Invalid index: {}", idx),
            MerkleError::UninitializedTree => write!(f, "Tree is not initialized"),
            MerkleError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for MerkleError {}

// Thread safety
unsafe impl Send for MerkleTree {}
unsafe impl Sync for MerkleTree {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_basic_operations() {
        let mut merkle = MerkleTree::with_default_config();
        let data = vec![
            b"block1".to_vec(),
            b"block2".to_vec(),
            b"block3".to_vec(),
            b"block4".to_vec(),
        ];

        assert!(merkle.build(data.clone()).is_ok());
        assert!(merkle.verify(0, b"block1").unwrap());
        assert!(merkle.update(0, b"new_block1".to_vec()).is_ok());
        assert!(merkle.verify(0, b"new_block1").unwrap());
    }

    #[test]
    fn test_merkle_proof_verification() {
        let mut merkle = MerkleTree::with_default_config();
        let data = vec![b"test1".to_vec(), b"test2".to_vec(), b"test3".to_vec()];

        merkle.build(data).unwrap();
        let proof = merkle.generate_proof(0).unwrap();
        assert!(MerkleTree::verify_proof_static(&proof, b"test1"));
        assert!(!MerkleTree::verify_proof_static(&proof, b"wrong_data"));
    }

    #[test]
    fn test_merkle_serialization() {
        let mut merkle = MerkleTree::with_default_config();
        let data = vec![b"test1".to_vec(), b"test2".to_vec()];

        merkle.build(data).unwrap();
        let serialized = merkle.serialize().unwrap();
        let deserialized = MerkleTree::deserialize(&serialized, MerkleConfig::default()).unwrap();

        assert_eq!(merkle.get_root_hash(), deserialized.get_root_hash());
    }
}