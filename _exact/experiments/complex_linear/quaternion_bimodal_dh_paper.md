# Quaternion Bimodal Differentiation-Harmonization Networks: 8x Parameter Efficiency Through Structured Hypercomplex Weights and Asymmetric Measurement Gating

**Andrew (Irintai)**

*Independent Researcher*

*February 2026*

---

## Abstract

We present a neural network architecture that achieves 8.2x greater parameter efficiency than standard scalar feedforward networks on MNIST digit classification. The architecture combines three principles: (1) quaternion-valued weights using the Hamilton product, which provide structured 4D rotational coupling from shared weight matrices; (2) a bimodal activation scheme in which the real component of the quaternion acts as a learned measurement gate on the imaginary waveform components, breaking the symmetry assumed by standard hypercomplex networks; and (3) alternating Differentiation and Harmonization layers inspired by the DHSR (Differentiation-Harmonization-Syntony-Recursion) cycle from Cosmological Recursion Theory. A network with 28,464 parameters achieves 97.70% accuracy on MNIST, compared to 98.00% from a standard network with 235,146 parameters — a 0.30% gap at 8.3x compression. Per-parameter efficiency is 3.432 accuracy-per-thousand-parameters versus 0.417, demonstrating that structured hypercomplex representations with asymmetric gating encode substantially more information per learnable weight than scalar alternatives.

---

## 1. Introduction

Modern neural networks achieve high performance through massive parameterization. A simple two-hidden-layer feedforward network for MNIST classification requires over 235,000 scalar weights to reach 98% accuracy. Each weight is a single floating-point number representing connection strength — a representation inherited from the earliest perceptron models and never fundamentally reconsidered despite decades of architectural innovation.

This paper asks whether the weight representation itself is a bottleneck. Biological neural connections are not scalar-valued. A synapse's effective weight is frequency-dependent, temporally dynamic, and modulated by phase relationships between pre- and post-synaptic oscillations. The "weight" of a biological connection is better described as a complex, multi-dimensional response profile than a single number.

We propose that hypercomplex-valued weights — specifically quaternion weights using the Hamilton product — can encode richer relationships per parameter, and that the full benefit of this representation requires breaking a symmetry assumption present in prior quaternion network literature. Standard quaternion neural networks treat all four components symmetrically. We introduce a bimodal architecture in which the real component serves as a measurement operator (gate) on the imaginary components (waveform), reflecting a fundamental asymmetry between observation and the observed.

This bimodal structure is combined with an alternating Differentiation-Harmonization (D→H) layer design derived from the DHSR cycle of Cosmological Recursion Theory (CRT), in which differentiation expands representational complexity and harmonization compresses it toward coherence. The result is a network that achieves near-baseline accuracy with an order of magnitude fewer parameters.

### 1.1 Contributions

- A quaternion linear layer using the full Hamilton product, providing structured coupling across 16 weight-input combinations from 4 stored matrices.
- A bimodal activation function in which the real quaternion component acts as a learned sigmoid gate (measurement) on the imaginary components (waveform), breaking quaternion symmetry.
- Differentiation and Harmonization layers implementing the D→H cycle: residual ReLU expansion followed by dual-pathway sigmoid damping and tanh coherence projection.
- A convolutional sensory entry that maps spatial image structure directly into quaternion component space.
- Experimental validation showing 8.2x parameter efficiency improvement on MNIST with a 0.30% accuracy gap.

---

## 2. Background and Motivation

### 2.1 Scalar Weight Limitations

In a standard linear layer $y = Wx + b$, each weight $W_{ij}$ is a scalar encoding the strength of connection from input $j$ to output $i$. For a layer mapping $n$ inputs to $m$ outputs, this requires $m \times n$ independent parameters. The weights are structurally independent — $W_{ij}$ shares no information with $W_{kl}$ unless explicitly regularized.

This independence is both a strength (flexibility) and a weakness (redundancy). In practice, learned weight matrices exhibit substantial structure — low rank, spectral concentration, spatial locality — suggesting that the full $m \times n$ parameterization is wasteful.

### 2.2 Quaternion Algebra and the Hamilton Product

Quaternions $\mathbb{H}$ extend complex numbers to four dimensions: $q = a + bi + cj + dk$ where $i^2 = j^2 = k^2 = ijk = -1$. The Hamilton product of two quaternions produces structured cross-coupling:

$$q_1 q_2 = (a_1 a_2 - b_1 b_2 - c_1 c_2 - d_1 d_2)$$
$$+ (a_1 b_2 + b_1 a_2 + c_1 d_2 - d_1 c_2)i$$
$$+ (a_1 c_2 - b_1 d_2 + c_1 a_2 + d_1 b_2)j$$
$$+ (a_1 d_2 + b_1 c_2 - c_1 b_2 + d_1 a_2)k$$

A quaternion linear layer stores four weight matrices $(W_a, W_b, W_c, W_d)$ but produces output through 16 matrix-vector products in structured combinations dictated by the Hamilton product. This means each weight matrix participates in all four output components with different signs and pairings, providing geometric coupling that scalar weights cannot replicate.

For a mapping from $n$ to $m$ features in quaternion space, the layer stores $4mn$ scalar values but produces outputs with coupling structure equivalent to a constrained $4m \times 4n$ scalar matrix. The constraint itself is the source of efficiency — it encodes rotational geometry that would otherwise need to be learned from scratch.

### 2.3 Prior Work on Quaternion Networks

Quaternion neural networks have been explored in speech processing (Parcollet et al., 2019), image classification (Gaudet & Maida, 2018), and signal processing applications. These works demonstrate parameter reduction from the Hamilton product's weight sharing but treat all four quaternion components symmetrically — applying identical activations and giving equal status to real and imaginary parts.

We depart from this convention based on a physical and philosophical observation: in quantum mechanics, the real component of a measurement corresponds to the observable eigenvalue while the complex phase encodes relational structure. We propose that neural networks can benefit from a similar asymmetry.

### 2.4 The DHSR Cycle

The Differentiation-Harmonization-Syntony-Recursion (DHSR) cycle from Cosmological Recursion Theory (CRT) describes a universal pattern of transformation:

- **Differentiation (D̂):** Expansion of possibility space. Generation of distinctions, complexity, and novelty.
- **Harmonization (Ĥ):** Compression toward coherence. Damping of excess complexity and projection toward stable structure.
- **Syntony (S):** Measurement of alignment — how well D and H are balanced, ideally approaching golden-ratio proportion.
- **Recursion (R):** Iteration until convergence or transcendence.

This cycle motivates our layer design: differentiation layers expand representational complexity through residual nonlinear projections, while harmonization layers compress through dual-pathway damping and coherence projection.

---

## 3. Architecture

### 3.1 Convolutional Sensory Entry

Raw pixel input enters through a single convolutional layer:

$$\text{Conv2d}(1 \rightarrow 4, \text{kernel}=5, \text{stride}=2)$$

This maps a $28 \times 28$ grayscale image to four $12 \times 12$ feature maps. Each feature map is assigned to one quaternion component:

$$r = \text{flatten}(\text{Conv}_0(x)), \quad i = \text{flatten}(\text{Conv}_1(x))$$
$$j = \text{flatten}(\text{Conv}_2(x)), \quad k = \text{flatten}(\text{Conv}_3(x))$$

This design ensures the image enters quaternion space through learned spatial features rather than as a flattened vector with zeros in imaginary channels. The convolutional entry costs 104 parameters (4 filters × 25 kernel weights + 4 biases) and produces 144-dimensional quaternion input.

### 3.2 Quaternion Linear Layer

Each quaternion linear layer from $n$ to $m$ features stores four weight matrices $W_a, W_b, W_c, W_d \in \mathbb{R}^{m \times n}$ and four bias vectors. The forward pass computes the Hamilton product:

$$\text{out}_r = rW_a^T - iW_b^T - jW_c^T - kW_d^T + b_a$$
$$\text{out}_i = rW_b^T + iW_a^T + jW_d^T - kW_c^T + b_b$$
$$\text{out}_j = rW_c^T - iW_d^T + jW_a^T + kW_b^T + b_c$$
$$\text{out}_k = rW_d^T + iW_c^T - jW_b^T + kW_a^T + b_d$$

Initialization uses $\mathcal{N}(0, 1/\sqrt{4n})$ to account for the four-way summation in each output component.

### 3.3 Bimodal Activation

The central architectural innovation is the asymmetric treatment of quaternion components. Rather than applying a uniform activation function, we designate the real component as a measurement channel and the imaginary components as waveform channels:

$$g = \sigma(r + b_{\text{measure}})$$
$$\text{out}_r = g$$
$$\text{out}_i = g \odot i, \quad \text{out}_j = g \odot j, \quad \text{out}_k = g \odot k$$

where $\sigma$ is the sigmoid function, $b_{\text{measure}}$ is a learnable bias, and $\odot$ denotes element-wise multiplication.

The real component passes through sigmoid to produce a gate in $[0, 1]$. This gate is then applied element-wise to the imaginary components, selectively amplifying or suppressing waveform features. The gate value itself propagates forward as the real component.

This implements a measurement operation: the real channel determines *what* to observe in the waveform, and the waveform carries the relational structure *being* observed. Information flows bidirectionally through training — the waveform learns to present features that the measurement channel can usefully gate, and the measurement channel learns which waveform features carry classification-relevant information.

### 3.4 Differentiation Layer

The differentiation layer implements $\hat{D}[x] = x + \alpha \cdot f(Q_{\text{linear}}(x))$ where $f$ is the bimodal activation:

1. Apply quaternion linear transformation.
2. Apply bimodal activation (measurement gates waveform).
3. Add residual connection (when dimensions match).

The residual connection preserves the input while the quaternion transform and bimodal gating generate new complexity. The scaling factor $\alpha$ controls differentiation strength.

### 3.5 Harmonization Layer

The harmonization layer implements $\hat{H}[x] = x - \beta \cdot \sigma(W_H x) + \gamma \cdot \tanh(W_S x)$ through two competing pathways:

**Damping pathway:** A quaternion linear layer followed by component-wise sigmoid. This produces values in $[0, 1]$ that are subtracted from the input, damping excess activation. Controlled by $\beta$.

**Syntony projection pathway:** A quaternion linear layer followed by component-wise tanh. This produces values in $[-1, 1]$ that are added to the input, pulling the representation toward coherent structure. Controlled by $\gamma$.

The combination $x - \beta \cdot \text{damp} + \gamma \cdot \text{syntony}$ simultaneously reduces dissonance and enhances coherence. In our experiments, $\beta = 0.5$ and $\gamma = 1.0$, weighting coherence projection more heavily than damping.

### 3.6 Output

The final quaternion linear layer projects from 32 to 10 features. The real component of the output — the measurement channel — serves directly as classification logits for cross-entropy loss. The imaginary components are discarded at output, having served their purpose as relational processing substrates throughout the network.

### 3.7 Complete Architecture Summary

| Component | Operation | Input Dim | Output Dim | Parameters |
|-----------|-----------|-----------|------------|------------|
| Sensory | Conv2d(1→4, k=5, s=2) | 1×28×28 | 4×12×12 | 104 |
| Differentiation | QuatLinear + BimodalAct + Residual | Q(144) | Q(32) | ~18,600 |
| Harmonization | Damping + Syntony + Residual | Q(32) | Q(32) | ~8,450 |
| Output | QuatLinear | Q(32) | Q(10) | ~1,330 |
| **Total** | | | | **28,464** |

Where Q($n$) denotes an $n$-dimensional quaternion representation (4$n$ scalar values).

---

## 4. Experimental Setup

### 4.1 Dataset

MNIST handwritten digit classification. 60,000 training images, 10,000 test images, 28×28 grayscale, 10 classes. Standard normalization (mean=0.1307, std=0.3081).

### 4.2 Baseline

A standard feedforward network with two hidden layers (784→256→128→10) using ReLU activations. 235,146 parameters. This represents a conventional scalar architecture of comparable depth.

### 4.3 Training Protocol

Both models trained identically:

- Optimizer: Adam (lr=0.001)
- Loss: Cross-entropy
- Batch size: 64
- Epochs: 10
- Hardware: NVIDIA RTX 3070 Ti (CUDA)

No data augmentation, dropout, learning rate scheduling, or early stopping was applied. Both models used identical random seeds for data loading. This minimal training protocol was chosen to isolate the effect of the weight representation from optimization engineering.

---

## 5. Results

### 5.1 Accuracy and Parameter Efficiency

| Model | Parameters | Best Accuracy | Final Accuracy | Acc/1k Params |
|-------|-----------|---------------|----------------|---------------|
| Scalar Baseline | 235,146 | 98.00% | 97.86% | 0.417 |
| Quaternion Bimodal D→H | 28,464 | 97.70% | 97.42% | 3.432 |

The quaternion bimodal network achieved 97.70% peak accuracy with 8.3x fewer parameters than the scalar baseline (28,464 vs. 235,146). Per-parameter efficiency was 8.2x higher (3.432 vs. 0.417 accuracy-per-thousand-parameters).

### 5.2 Training Dynamics

The scalar baseline converged quickly, reaching 96.24% after epoch 1 and 98.00% by epoch 4. The quaternion network showed slightly slower initial convergence (95.62% after epoch 1) but reached competitive performance by epoch 3 (97.34%) and peaked at epoch 7 (97.70%).

Both models exhibited minor oscillation in later epochs, consistent with a fixed learning rate without decay. The quaternion model showed more stable late-epoch behavior in this configuration compared to prior iterations without the convolutional sensory entry, suggesting that structured spatial input to the quaternion space improves training stability.

### 5.3 Training Time

The quaternion network required 158.6 seconds (15.9s/epoch) compared to 110.0 seconds (11.0s/epoch) for the scalar baseline. The increased time per epoch reflects the 16 matrix-vector products per quaternion linear layer versus 1 per scalar linear layer. This overhead is an implementation artifact — the products are independent and parallelizable, and a fused CUDA kernel (as implemented in the accompanying srt_library Rust/CUDA codebase) would substantially reduce this gap.

---

## 6. Analysis

### 6.1 Why Quaternion Weights Encode More

The Hamilton product creates structured dependencies between weight matrices. In a scalar layer mapping $n \rightarrow m$, the $mn$ weights are independent. In a quaternion layer, the four $mn$-element matrices produce outputs through 16 linear operations with fixed sign patterns. This means:

1. Each weight element influences all four output components.
2. The influence pattern follows the geometry of 4D rotations.
3. The network need not learn rotational structure from scratch — it is built into the weight interaction.

For representations that benefit from rotational or phase-based structure (as spatial features do), this is free information that scalar weights must redundantly learn.

### 6.2 Why Bimodal Gating Matters

Standard quaternion networks apply identical activations to all components. Our bimodal design treats the real component as a measurement channel — it learns what to attend to — while imaginary components carry the signal being attended to.

This asymmetry means the network naturally develops an internal attention-like mechanism at every layer, without the overhead of explicit attention modules. The sigmoid gate learns to be selective; the waveform channels learn to present useful features to the gate. This co-adaptation is more parameter-efficient than learning both the features and the attention pattern through separate weight matrices.

### 6.3 Why D→H Cycling Helps

The differentiation layer expands representational complexity: the residual connection $x + \alpha \cdot f(Qx)$ adds new features while preserving old ones. The harmonization layer then compresses: $x - \beta \cdot \sigma(W_H x) + \gamma \cdot \tanh(W_S x)$ damps excess while projecting toward coherence.

This expand-then-compress rhythm prevents the failure modes of each operation in isolation. Pure expansion leads to representational explosion and overfitting. Pure compression leads to information loss. The alternating cycle maintains a dynamic balance, analogous to the breathing rhythm of biological neural oscillations where excitatory and inhibitory phases alternate to maintain stable information processing.

### 6.4 Role of Convolutional Entry

The convolutional sensory layer serves two purposes. Practically, it reduces the input dimension from 784 to 144, eliminating the parameter-heavy first projection. Architecturally, it ensures that each quaternion component receives spatially structured input rather than a copy of flattened pixels (real) padded with zeros (imaginary).

This matters because the Hamilton product couples all four components. If three start at zero, the first layer's quaternion structure is partially wasted — the cross-coupling terms involving zero-initialized imaginary components contribute nothing until gradients populate them. The convolutional entry ensures all four components carry meaningful, distinct spatial information from the first forward pass.

---

## 7. Relation to Existing Work

### 7.1 Quaternion Neural Networks

Parcollet et al. (2019) demonstrated quaternion networks for speech recognition, showing parameter reduction through Hamilton product weight sharing. Gaudet and Maida (2018) applied quaternion convolutions to image classification. Both works treat quaternion components symmetrically. Our bimodal activation breaks this symmetry, adding an architectural inductive bias that further improves parameter efficiency.

### 7.2 Complex-Valued Networks

Trabelsi et al. (2018) introduced deep complex networks with complex batch normalization and complex weight initialization. Our work extends beyond complex to quaternion (4D vs 2D) and introduces the asymmetric measurement/waveform distinction absent from complex network literature.

### 7.3 Gating Mechanisms

The bimodal activation is related to gating mechanisms in LSTMs (Hochreiter & Schmidhuber, 1997) and Gated Linear Units (Dauphin et al., 2017). The distinction is that our gate operates between quaternion components within the same representation, rather than between separate input streams or time steps. The gate and the gated signal are algebraically coupled through the Hamilton product, creating a tighter feedback loop.

### 7.4 Mixture of Experts and Modular Networks

The D→H architecture shares motivation with mixture-of-experts approaches (Shazeer et al., 2017) in that different network components serve different functions (expansion vs. compression). However, our approach is not routing-based — every input passes through both D and H layers. The functional specialization is architectural rather than conditional.

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

**Benchmark scope.** Results are demonstrated on MNIST only. While MNIST is a standard proof-of-concept benchmark, extending to CIFAR-10, ImageNet subsets, and non-vision tasks is necessary to establish generality.

**Training time overhead.** The quaternion network is approximately 1.5x slower per epoch due to the 16 matrix-vector products per layer. Custom CUDA kernels implementing fused quaternion Hamilton product operations would reduce this overhead. A Rust/CUDA implementation of the quaternion Hamilton product already exists in the accompanying srt_library codebase and could be integrated.

**Fixed D/H hyperparameters.** The differentiation strength ($\alpha$), damping coefficient ($\beta$), and syntony projection strength ($\gamma$) are currently fixed. Making these learnable or adaptive — as implemented in the AdaptiveGate module of the syntonic_applications library — could further improve performance.

**No learning rate scheduling.** Both models were trained with fixed learning rate. The quaternion model showed minor late-epoch oscillation that a cosine schedule or reduce-on-plateau strategy would address.

### 8.2 Future Directions

**Octonion and sedenion extensions.** The srt_library includes full implementations of octonion (8D) and sedenion (16D) algebras. Octonion weights would provide even richer coupling structure, though the loss of associativity requires careful handling. The sedenion case introduces zero divisors, which may serve as a natural regularization mechanism — an open research question.

**Retrocausal attractor optimization.** The current implementation uses standard Adam optimization. The syntonic_applications library includes a retrocausal attractor system that models optimization as gravitational attraction toward predicted weight optima, and a Golden Momentum optimizer using $\beta = 1/\phi \approx 0.618$ as the momentum coefficient. Replacing Adam with these SRT-derived optimizers is a natural next step.

**Syntony measurement as a training signal.** The DHSR cycle includes a Syntony measurement phase that evaluates the balance between differentiation and harmonization. Incorporating syntony as an auxiliary loss term — penalizing states that are too differentiated (chaotic) or too harmonized (collapsed) — could provide a principled alternative to standard regularization.

**Recursive depth.** The current architecture implements a single D→H cycle. Stacking multiple cycles (D→H→D→H→...) with shared or progressive weights would implement the Recursion phase of DHSR, potentially achieving convergence to stable representations without manual depth selection.

**Resonant weight spaces.** The current implementation stores quaternion weights as four independent tensors operated on by PyTorch's standard backward pass. The theoretical framework predicts that weights stored as resonant oscillators — with phase, frequency, and amplitude rather than scalar values — would encode even more per parameter. The srt_library's ResonantTensor and ResonantLinear implementations provide the substrate for this extension.

---

## 9. Conclusion

We demonstrate that quaternion-valued weights with bimodal measurement gating and differentiation-harmonization cycling achieve 97.70% accuracy on MNIST with 28,464 parameters — 8.3x fewer than a standard scalar network achieving 98.00%. Per-parameter information encoding is 8.2x more efficient.

The key insight is not merely that hypercomplex weights share parameters through algebraic structure — this has been known. The contribution is the bimodal asymmetry: treating the real component as measurement and imaginary components as waveform, combined with the D→H cycle that alternates expansion and compression. Together, these create a network that processes information through structured phase relationships rather than scalar accumulation, achieving comparable performance with an order of magnitude fewer learnable values.

This result suggests that the standard scalar weight representation is a significant source of parameter inefficiency in neural networks, and that structured hypercomplex alternatives — particularly those incorporating asymmetric gating and cyclic expansion/compression — offer a path toward substantially more compact models.

---

## References

Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language modeling with gated convolutional networks. *ICML*.

Gaudet, C. J., & Maida, A. S. (2018). Deep quaternion networks. *IJCNN*.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

Parcollet, T., Ravanelli, M., Morchid, M., Linarès, G., Trabelsi, C., De Mori, R., & Bengio, Y. (2019). Quaternion recurrent neural networks. *ICLR*.

Shazeer, N., Mirhoseini, A., Matyasovszky, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR*.

Trabelsi, C., Bilaniuk, O., Zhang, Y., Serdyuk, D., Subramanian, S., Santos, J. F., ... & Pal, C. J. (2018). Deep complex networks. *ICLR*.

---

## Appendix A: Source Code

The complete experimental code is available at:

`/home/Andrew/Documents/SRT_complete/ouroboros_prime/experiment/complex_linear_v3.py`

The supporting library implementations — including Rust/CUDA quaternion operations, ResonantTensor, DifferentiationLayer, HarmonizationLayer, SyntonicGate, GoldenMomentumOptimizer, and retrocausal attractor systems — are available in the syntonic_applications and srt_library packages within the ouroboros_prime repository.

## Appendix B: Reproducibility

Hardware: NVIDIA GeForce RTX 3070 Ti (8GB VRAM), Intel i9-9900K, 32GB RAM, Ubuntu Linux.

Software: Python 3.11+, PyTorch 2.x with CUDA support.

Training: Adam optimizer, lr=0.001, batch size 64, 10 epochs, cross-entropy loss. No data augmentation, dropout, or learning rate scheduling. MNIST with standard normalization (μ=0.1307, σ=0.3081).
