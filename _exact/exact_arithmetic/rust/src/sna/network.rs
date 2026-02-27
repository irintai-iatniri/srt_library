use pyo3::prelude::*;
use std::collections::VecDeque;
use crate::sna::resonant_oscillator::ResonantOscillator;

/// The Time Tunnel (Axon).
/// Carries signals from Source -> Target with a specific temporal delay.
/// Not exposed to Python directly; managed by the Network.
#[derive(Clone)]
struct Axon {
    source_id: usize,
    target_id: usize,
    target_sensor_idx: usize,
    // Buffer Head = Arriving Now. Tail = Just Sent.
    buffer: VecDeque<i64>, 
}

impl Axon {
    fn new(source: usize, target: usize, sensor: usize, delay: usize) -> Self {
        // Causality: Signal cannot arrive before it is sent. Min delay = 1 tick.
        let safe_delay = std::cmp::max(1, delay);
        
        let mut buffer = VecDeque::with_capacity(safe_delay);
        // Pre-fill with Silence (The Void) to establish causality
        for _ in 0..safe_delay {
            buffer.push_back(0);
        }
        
        Self {
            source_id: source,
            target_id: target,
            target_sensor_idx: sensor,
            buffer,
        }
    }

    /// The Clock Tick.
    /// Pushes new spike into the "Past". Pops old spike into the "Present".
    fn tick(&mut self, new_spike: i64) -> i64 {
        self.buffer.push_back(new_spike);
        self.buffer.pop_front().unwrap_or(0)
    }
    
    /// Peek at the value arriving this tick (without consuming it yet)
    fn peek_arrival(&self) -> i64 {
        *self.buffer.front().unwrap_or(&0)
    }
    
    fn reset(&mut self) {
        let len = self.buffer.len();
        self.buffer.clear();
        for _ in 0..len {
            self.buffer.push_back(0);
        }
    }
}

/// The Holographic Connectome.
/// Manages a sparse graph of neurons connected by temporal delay lines.
#[pyclass]
pub struct SyntonicNetwork {
    neurons: Vec<ResonantOscillator>,
    axons: Vec<Axon>,
    // Optimization: Map target_id -> List of Axon indices feeding it
    incoming_map: Vec<Vec<usize>>, 
    input_indices: Vec<usize>,
    output_indices: Vec<usize>,
}

#[pymethods]
impl SyntonicNetwork {
    #[new]
    pub fn new() -> Self {
        Self {
            neurons: Vec::new(),
            axons: Vec::new(),
            incoming_map: Vec::new(),
            input_indices: Vec::new(),
            output_indices: Vec::new(),
        }
    }

    /// Returns the number of neurons in the network.
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Returns the number of synapses (axons) in the network.
    pub fn synapse_count(&self) -> usize {
        self.axons.len()
    }

    /// Adds a neuron to the network.
    /// Returns: The new neuron's ID.
    pub fn add_neuron(&mut self, neuron: ResonantOscillator) -> usize {
        let id = self.neurons.len();
        self.neurons.push(neuron);
        self.incoming_map.push(Vec::new());
        id
    }

    /// Connects two neurons with a delay line.
    /// source: ID of firing neuron
    /// target: ID of receiving neuron
    /// sensor_idx: Which input port on the target to connect to
    /// delay: Propagation time in ticks (min 1)
    pub fn add_synapse(&mut self, source: usize, target: usize, sensor_idx: usize, delay: usize) -> PyResult<()> {
        if source >= self.neurons.len() || target >= self.neurons.len() {
             return Err(pyo3::exceptions::PyValueError::new_err("Invalid neuron ID"));
        }
        
        let target_neuron = &self.neurons[target];
        if sensor_idx >= target_neuron.sensor_count() {
             return Err(pyo3::exceptions::PyValueError::new_err("Invalid sensor index"));
        }

        let axon = Axon::new(source, target, sensor_idx, delay);
        let axon_idx = self.axons.len();
        self.axons.push(axon);
        
        // Register for fast lookup during tick
        self.incoming_map[target].push(axon_idx);
        
        Ok(())
    }
    
    /// Registers a neuron as an entry point for external signals.
    /// External input N maps to input_indices[N].
    pub fn register_input(&mut self, neuron_id: usize) -> PyResult<()> {
        if neuron_id >= self.neurons.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid neuron ID for input"));
        }
        if !self.input_indices.contains(&neuron_id) {
            self.input_indices.push(neuron_id);
        }
        Ok(())
    }
    
    /// Registers a neuron as an exit point.
    /// Output N maps to output_indices[N].
    pub fn register_output(&mut self, neuron_id: usize) -> PyResult<()> {
        if neuron_id >= self.neurons.len() {
             return Err(pyo3::exceptions::PyValueError::new_err("Invalid neuron ID for output"));
        }
        if !self.output_indices.contains(&neuron_id) {
            self.output_indices.push(neuron_id);
        }
        Ok(())
    }

    /// Hard reset of all network state (buffers and neurons).
    /// Use this between independent sequences.
    pub fn reset_state(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        for axon in &mut self.axons {
            axon.reset();
        }
    }

    /// Advances the network by one time step.
    /// external_inputs: Vector of values matching the registered inputs.
    pub fn tick(&mut self, external_inputs: Vec<i64>) -> PyResult<Vec<i8>> {
        self.internal_tick(&external_inputs)
    }

    /// Advances the network by one time step and returns ALL neuron states.
    /// external_inputs: Vector of values matching the registered inputs.
    /// Returns: Vec<i8> of all neuron states (for reservoir observation)
    #[pyo3(signature = (external_inputs))]
    pub fn tick_all(&mut self, external_inputs: Vec<i64>) -> PyResult<Vec<i8>> {
        // 1. Collect inputs for each neuron
        let mut network_inputs = Vec::with_capacity(self.neurons.len());
        
        for n_id in 0..self.neurons.len() {
            let sensor_count = self.neurons[n_id].sensor_count();
            let mut inputs = vec![0; sensor_count];

            // A. Axonal Inputs (Internal Superposition)
            for &axon_idx in &self.incoming_map[n_id] {
                let axon = &self.axons[axon_idx];
                inputs[axon.target_sensor_idx] += axon.peek_arrival();
            }

            // B. External Inputs (To Sensor 0)
            if let Some(ext_idx) = self.input_indices.iter().position(|&id| id == n_id) {
                if sensor_count > 0 {
                    inputs[0] += external_inputs[ext_idx];
                }
            }
            
            network_inputs.push(inputs);
        }

        // 2. Neural Firing (Parallel Physics)
        let mut spikes = Vec::with_capacity(self.neurons.len());
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let spike = neuron.step(network_inputs[i].clone())?;
            spikes.push(spike as i64);
        }

        // 3. Axonal Transport (Propagation)
        for axon in self.axons.iter_mut() {
            let source_spike = spikes[axon.source_id];
            axon.tick(source_spike);
        }

        // 4. Return ALL neuron states (not just outputs)
        let all_states: Vec<i8> = spikes.iter().map(|&s| s as i8).collect();
        Ok(all_states)
    }

    /// Processes an entire sequence of inputs for batch efficiency.
    /// input_sequence: List of input vectors [Time][InputChannel]
    /// Returns: List of output vectors [Time][OutputChannel]
    pub fn process_sequence(&mut self, input_sequence: Vec<Vec<i64>>) -> PyResult<Vec<Vec<i8>>> {
        let mut output_sequence = Vec::with_capacity(input_sequence.len());
        
        for inputs in input_sequence {
            let outputs = self.internal_tick(&inputs)?;
            output_sequence.push(outputs);
        }
        
        Ok(output_sequence)
    }

    /// Introspects the network topology.
    /// 
    /// Returns a list of dictionaries, one per axon connection:
    /// - 'source': source neuron index
    /// - 'target': target neuron index
    /// - 'sensor': target sensor index (which input to target neuron)
    /// - 'delay': axonal delay in ticks
    ///
    /// # Example (Python)
    /// ```python
    /// net = SyntonicNetwork()
    /// net.add_neurons(3)
    /// net.add_axon(source=0, target=1, sensor=0, delay=5)
    /// net.add_axon(source=1, target=2, sensor=0, delay=10)
    /// topology = net.inspect_topology()
    /// # Returns: [
    /// #   {'source': 0, 'target': 1, 'sensor': 0, 'delay': 5},
    /// #   {'source': 1, 'target': 2, 'sensor': 0, 'delay': 10}
    /// # ]
    /// ```
    pub fn inspect_topology(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let mut topology = Vec::with_capacity(self.axons.len());
        
        for axon in &self.axons {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("source", axon.source_id)?;
            dict.set_item("target", axon.target_id)?;
            dict.set_item("sensor", axon.target_sensor_idx)?;
            dict.set_item("delay", axon.buffer.len())?;
            topology.push(dict.into());
        }
        
        Ok(topology)
    }

    /// Introspects the network's input/output configuration.
    /// 
    /// Returns a tuple (input_indices, output_indices):
    /// - input_indices: List of neuron indices designated as inputs
    /// - output_indices: List of neuron indices designated as outputs
    ///
    /// # Example (Python)
    /// ```python
    /// net = SyntonicNetwork()
    /// net.add_neurons(5)
    /// net.register_io(inputs=[0, 1], outputs=[3, 4])
    /// inputs, outputs = net.inspect_io()
    /// # Returns: ([0, 1], [3, 4])
    /// ```
    pub fn inspect_io(&self, _py: Python) -> PyResult<(Vec<usize>, Vec<usize>)> {
        Ok((self.input_indices.clone(), self.output_indices.clone()))
    }
}

// Internal Logic
impl SyntonicNetwork {
    fn internal_tick(&mut self, external_inputs: &[i64]) -> PyResult<Vec<i8>> {
        if external_inputs.len() != self.input_indices.len() {
             return Err(pyo3::exceptions::PyValueError::new_err(format!(
                 "Input count mismatch: Expected {}, got {}", 
                 self.input_indices.len(), external_inputs.len()
             )));
        }

        // 1. Synaptic Integration (Gathering)
        let mut network_inputs: Vec<Vec<i64>> = Vec::with_capacity(self.neurons.len());

        for (n_id, neuron) in self.neurons.iter().enumerate() {
            let sensor_count = neuron.sensor_count();
            let mut inputs = vec![0; sensor_count];

            // A. Axonal Inputs (Internal Superposition)
            for &axon_idx in &self.incoming_map[n_id] {
                let axon = &self.axons[axon_idx];
                inputs[axon.target_sensor_idx] += axon.peek_arrival();
            }

            // B. External Inputs (To Sensor 0)
            if let Some(ext_idx) = self.input_indices.iter().position(|&id| id == n_id) {
                if sensor_count > 0 {
                    inputs[0] += external_inputs[ext_idx];
                }
            }
            
            network_inputs.push(inputs);
        }

        // 2. Neural Firing (Parallel Physics)
        let mut spikes = Vec::with_capacity(self.neurons.len());
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            // Note: .clone() here copies small Vec<i64> (inputs).
            // Acceptable cost for Prototype 1.
            let spike = neuron.step(network_inputs[i].clone())?;
            spikes.push(spike as i64);
        }

        // 3. Axonal Transport (Propagation)
        for axon in self.axons.iter_mut() {
            let source_spike = spikes[axon.source_id];
            axon.tick(source_spike);
        }

        // 4. Output Collection
        let mut output_vals = Vec::new();
        for &id in &self.output_indices {
            output_vals.push(spikes[id] as i8);
        }

        Ok(output_vals)
    }
}
