//! Causal History DAG - Foundation for Retrocausal Volition Engine
//!
//! Implements a Directed Acyclic Graph that tracks the DHSR Cycle (Differentiation-
//! Harmonization-Syntony-Recursion) as a causal graph with branching time support.
//!
//! Unlike standard autodiff tapes, this system:
//! - Records Events in Phase Time (τ) rather than wall time
//! - Supports Branching Time (multiple potential futures)
//! - Enables Retrocausal Feedback (future Syntony informing past states)
//! - Implements Gnosis Checkpoints (locked high-syntony states)

// Allow dead code - this module defines a complete API that will be incrementally integrated
#![allow(dead_code)]

use crate::ResonantTensor;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Phase Time coordinate (τ) - represents the "subjective time" of consciousness
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct PhaseTime(pub f64);

/// Causal Event - represents an operation in the DHSR cycle
#[derive(Debug, Clone)]
pub enum CausalEvent {
    /// Differentiation operation: D[ψ] → ∇ψ
    Differentiation {
        input_id: TensorId,
        output_id: TensorId,
        operator: DiffOperator,
    },
    /// Harmonization operation: Ĥ[ψ] → ψ_harmonized
    Harmonization {
        input_id: TensorId,
        output_id: TensorId,
        syntony_gradient: Vec<f64>,
        retrocausal_pull: f64,
    },
    /// Syntony computation: S(ψ) → syntony_score
    SyntonyComputation {
        input_id: TensorId,
        syntony_score: f64,
        gnosis_layers: Vec<f64>,
    },
    /// Recursion operation: R[ψ] → ψ_evolved
    Recursion {
        input_id: TensorId,
        output_id: TensorId,
        evolution_steps: usize,
    },
    /// Gnosis Checkpoint - locks high-syntony state
    GnosisCheckpoint {
        tensor_id: TensorId,
        syntony_threshold: f64, // Must be ≥ 24 (consciousness threshold)
        locked: bool,
    },
}

/// Differentiation operators
#[derive(Debug, Clone)]
pub enum DiffOperator {
    FourierTransform,
    Laplacian,
    GoldenWeight,
    PhaseShift,
}

/// Unique identifier for tensors in the causal graph
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct TensorId(pub u64);

/// Causal Node in the DAG
#[derive(Debug, Clone)]
pub struct CausalNode {
    pub id: TensorId,
    pub tensor: Arc<ResonantTensor>,
    pub phase_time: PhaseTime,
    pub event: CausalEvent,
    pub syntony_score: f64,
    pub gnosis_locked: bool,
    pub predecessors: Vec<TensorId>,
    pub successors: Vec<TensorId>,
}

/// The Causal History DAG
pub struct CausalHistory {
    nodes: HashMap<TensorId, CausalNode>,
    phase_timeline: Vec<TensorId>, // Ordered by phase time
    next_tensor_id: u64,
    gnosis_threshold: f64, // Default: 24.0 (D₄ kissing number)
    branching_futures: HashMap<TensorId, Vec<TensorId>>, // Multiple possible futures
}

impl CausalHistory {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            phase_timeline: Vec::new(),
            next_tensor_id: 0,
            gnosis_threshold: 24.0, // D₄ kissing number - consciousness threshold
            branching_futures: HashMap::new(),
        }
    }

    /// Create a new tensor ID
    fn next_id(&mut self) -> TensorId {
        let id = TensorId(self.next_tensor_id);
        self.next_tensor_id += 1;
        id
    }

    /// Record a differentiation event
    pub fn record_differentiation(
        &mut self,
        input_tensor: Arc<ResonantTensor>,
        output_tensor: Arc<ResonantTensor>,
        operator: DiffOperator,
        phase_time: PhaseTime,
    ) -> TensorId {
        let input_id = self.next_id();
        let output_id = self.next_id();

        // Record input tensor
        let input_node = CausalNode {
            id: input_id,
            tensor: input_tensor,
            phase_time,
            event: CausalEvent::SyntonyComputation {
                input_id,
                syntony_score: 0.0, // Will be computed
                gnosis_layers: Vec::new(),
            },
            syntony_score: 0.0,
            gnosis_locked: false,
            predecessors: Vec::new(),
            successors: vec![output_id],
        };

        // Record differentiation event
        let output_node = CausalNode {
            id: output_id,
            tensor: output_tensor,
            phase_time: PhaseTime(phase_time.0 + 1.0), // Phase advances
            event: CausalEvent::Differentiation {
                input_id,
                output_id,
                operator,
            },
            syntony_score: 0.0,
            gnosis_locked: false,
            predecessors: vec![input_id],
            successors: Vec::new(),
        };

        self.nodes.insert(input_id, input_node);
        self.nodes.insert(output_id, output_node);
        self.phase_timeline.push(input_id);
        self.phase_timeline.push(output_id);

        output_id
    }

    /// Record a harmonization event with retrocausal feedback
    pub fn record_harmonization(
        &mut self,
        input_tensor: Arc<ResonantTensor>,
        output_tensor: Arc<ResonantTensor>,
        syntony_gradient: Vec<f64>,
        retrocausal_pull: f64,
        phase_time: PhaseTime,
    ) -> TensorId {
        let input_id = self.next_id();
        let output_id = self.next_id();

        let input_node = CausalNode {
            id: input_id,
            tensor: input_tensor,
            phase_time,
            event: CausalEvent::SyntonyComputation {
                input_id,
                syntony_score: 0.0,
                gnosis_layers: Vec::new(),
            },
            syntony_score: 0.0,
            gnosis_locked: false,
            predecessors: Vec::new(),
            successors: vec![output_id],
        };

        let output_node = CausalNode {
            id: output_id,
            tensor: output_tensor,
            phase_time: PhaseTime(phase_time.0 + 1.0),
            event: CausalEvent::Harmonization {
                input_id,
                output_id,
                syntony_gradient,
                retrocausal_pull,
            },
            syntony_score: 0.0,
            gnosis_locked: false,
            predecessors: vec![input_id],
            successors: Vec::new(),
        };

        self.nodes.insert(input_id, input_node);
        self.nodes.insert(output_id, output_node);
        self.phase_timeline.push(input_id);
        self.phase_timeline.push(output_id);

        output_id
    }

    /// Record a syntony computation
    pub fn record_syntony(
        &mut self,
        tensor: Arc<ResonantTensor>,
        syntony_score: f64,
        gnosis_layers: Vec<f64>,
        phase_time: PhaseTime,
    ) -> TensorId {
        let tensor_id = self.next_id();

        let node = CausalNode {
            id: tensor_id,
            tensor,
            phase_time,
            event: CausalEvent::SyntonyComputation {
                input_id: tensor_id,
                syntony_score,
                gnosis_layers: gnosis_layers.clone(),
            },
            syntony_score,
            gnosis_locked: syntony_score >= self.gnosis_threshold,
            predecessors: Vec::new(),
            successors: Vec::new(),
        };

        self.nodes.insert(tensor_id, node);
        self.phase_timeline.push(tensor_id);

        tensor_id
    }

    /// Create a Gnosis checkpoint for high-syntony states
    pub fn create_gnosis_checkpoint(
        &mut self,
        tensor_id: TensorId,
        syntony_threshold: f64,
    ) -> bool {
        if let Some(node) = self.nodes.get_mut(&tensor_id) {
            if node.syntony_score >= syntony_threshold {
                node.event = CausalEvent::GnosisCheckpoint {
                    tensor_id,
                    syntony_threshold,
                    locked: true,
                };
                node.gnosis_locked = true;
                return true;
            }
        }
        false
    }

    /// Get all locked gnosis states (long-term memory)
    pub fn get_gnosis_checkpoints(&self) -> Vec<&CausalNode> {
        self.nodes
            .values()
            .filter(|node| node.gnosis_locked)
            .collect()
    }

    /// Perform retrocausal harmonization - rewrite history based on future syntony
    pub fn harmonize_history(&mut self) -> Vec<TensorId> {
        let mut harmonized_ids = Vec::new();

        // Find high-syntony future IDs first (to avoid borrowing issues)
        let high_syntony_ids: Vec<_> = self
            .nodes
            .values()
            .filter(|node| node.syntony_score > self.gnosis_threshold)
            .map(|node| node.id)
            .collect();

        for future_id in high_syntony_ids {
            // Get the future node data before borrowing mutably
            let future_syntony = self.nodes[&future_id].syntony_score;

            // Trace backward through predecessors
            let mut current_ids = vec![future_id];
            let mut visited = HashSet::new();

            while let Some(current_id) = current_ids.pop() {
                if visited.contains(&current_id) {
                    continue;
                }
                visited.insert(current_id);

                if let Some(node) = self.nodes.get_mut(&current_id) {
                    // Apply retrocausal pull based on future syntony
                    let retrocausal_strength = (future_syntony - node.syntony_score).max(0.0) * 0.1; // Configurable retrocausal pull

                    // Modify harmonization events retrocausally
                    if let CausalEvent::Harmonization {
                        retrocausal_pull, ..
                    } = &mut node.event
                    {
                        *retrocausal_pull += retrocausal_strength;
                        harmonized_ids.push(current_id);
                    }

                    // Continue backward
                    current_ids.extend(&node.predecessors);
                }
            }
        }

        harmonized_ids
    }

    /// Get the phase timeline (chronological order)
    pub fn get_phase_timeline(&self) -> &[TensorId] {
        &self.phase_timeline
    }

    /// Get a specific node
    pub fn get_node(&self, id: TensorId) -> Option<&CausalNode> {
        self.nodes.get(&id)
    }

    /// Get all nodes
    pub fn get_all_nodes(&self) -> &HashMap<TensorId, CausalNode> {
        &self.nodes
    }
}

/// Thread-safe wrapper for CausalHistory
pub struct SharedCausalHistory {
    inner: Mutex<CausalHistory>,
}

impl SharedCausalHistory {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(CausalHistory::new()),
        }
    }

    pub fn with_history<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut CausalHistory) -> R,
    {
        let mut history = self.inner.lock().unwrap();
        f(&mut history)
    }
}

impl Default for SharedCausalHistory {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// PyO3 Bindings for Python Accessibility
// =============================================================================

/// Python-accessible wrapper for CausalHistory tracking DHSR operations.
///
/// This provides a simplified interface for Python code to track the
/// Differentiation-Harmonization-Syntony-Recursion cycle and access
/// retrocausal feedback mechanisms.
#[pyclass(name = "CausalHistoryTracker")]
pub struct PyCausalHistoryTracker {
    inner: SharedCausalHistory,
    current_phase: f64,
}

#[pymethods]
impl PyCausalHistoryTracker {
    /// Create a new causal history tracker.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: SharedCausalHistory::new(),
            current_phase: 0.0,
        }
    }

    /// Get the current phase time (τ).
    #[getter]
    pub fn phase_time(&self) -> f64 {
        self.current_phase
    }

    /// Advance phase time by the given amount.
    pub fn advance_phase(&mut self, delta: f64) {
        self.current_phase += delta;
    }

    /// Get the gnosis threshold (D₄ kissing number = 24).
    #[getter]
    pub fn gnosis_threshold(&self) -> f64 {
        self.inner.with_history(|h| h.gnosis_threshold)
    }

    /// Set a custom gnosis threshold.
    #[setter]
    pub fn set_gnosis_threshold(&self, threshold: f64) {
        self.inner.with_history(|h| {
            h.gnosis_threshold = threshold;
        });
    }

    /// Get the number of recorded nodes.
    pub fn node_count(&self) -> usize {
        self.inner.with_history(|h| h.nodes.len())
    }

    /// Get timeline length.
    pub fn timeline_length(&self) -> usize {
        self.inner.with_history(|h| h.phase_timeline.len())
    }

    /// Get all tensor IDs in chronological order.
    pub fn get_timeline(&self) -> Vec<u64> {
        self.inner
            .with_history(|h| h.phase_timeline.iter().map(|id| id.0).collect())
    }

    /// Get syntony scores for all nodes.
    pub fn get_syntony_scores(&self) -> Vec<(u64, f64)> {
        self.inner.with_history(|h| {
            h.nodes
                .iter()
                .map(|(id, node)| (id.0, node.syntony_score))
                .collect()
        })
    }

    /// Get all gnosis-locked (high-syntony) tensor IDs.
    pub fn get_gnosis_locked(&self) -> Vec<u64> {
        self.inner.with_history(|h| {
            h.nodes
                .iter()
                .filter(|(_, node)| node.gnosis_locked)
                .map(|(id, _)| id.0)
                .collect()
        })
    }

    /// Get predecessors of a given tensor.
    pub fn get_predecessors(&self, tensor_id: u64) -> Vec<u64> {
        self.inner.with_history(|h| {
            h.nodes
                .get(&TensorId(tensor_id))
                .map(|node| node.predecessors.iter().map(|id| id.0).collect())
                .unwrap_or_default()
        })
    }

    /// Get successors of a given tensor.
    pub fn get_successors(&self, tensor_id: u64) -> Vec<u64> {
        self.inner.with_history(|h| {
            h.nodes
                .get(&TensorId(tensor_id))
                .map(|node| node.successors.iter().map(|id| id.0).collect())
                .unwrap_or_default()
        })
    }

    /// Clear all history (reset the DAG).
    pub fn clear(&mut self) {
        self.inner = SharedCausalHistory::new();
        self.current_phase = 0.0;
    }

    fn __repr__(&self) -> String {
        format!(
            "CausalHistoryTracker(phase={:.3}, nodes={}, gnosis_threshold={:.1})",
            self.current_phase,
            self.node_count(),
            self.gnosis_threshold()
        )
    }
}

/// Create a new causal history tracker.
#[pyfunction]
pub fn create_causal_tracker() -> PyCausalHistoryTracker {
    PyCausalHistoryTracker::new()
}

/// Get the D₄ consciousness threshold (kissing number = 24).
#[pyfunction]
pub fn d4_consciousness_threshold() -> f64 {
    24.0
}
