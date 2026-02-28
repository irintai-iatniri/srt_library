# srt_library — Directory Tree

*Last updated: 2026-02-28*

```
srt_library/
├── CLAUDE.md
├── README.md
├── _exact/
│   ├── CLAUDE.md
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── LICENSE-COMMERCIAL.md
│   ├── LICENSE-RESEARCH.md
│   ├── README.md
│   ├── REFACTORING_PLAN.md
│   ├── RUST_CODE_STRUCTURE.md
│   ├── __init__.py
│   ├── _core/
│   │   └── __init__.py
│   ├── compile_exact_kernels.py
│   ├── compile_exact_kernels.sh
│   ├── compile_kernels.py
│   ├── compile_kernels.sh
│   ├── core/
│   │   ├── __init__.py
│   │   ├── backend.py
│   │   ├── constants.py
│   │   ├── device.py
│   │   ├── dtype.py
│   │   ├── state.py
│   │   └── types.py
│   ├── debug/
│   │   ├── debug_cuda.py
│   │   ├── debug_softmax.py
│   │   └── debug_test_env.py
│   ├── docs/
│   │   ├── README.md
│   │   ├── exact_arithmetic_api.md
│   │   ├── exact_arithmetic_architecture.md
│   │   └── migrating_to_exact.md
│   ├── exact_arithmetic/
│   │   ├── Cargo.lock
│   │   ├── Cargo.toml
│   │   ├── README.md
│   │   ├── RUST_CODE_STRUCTURE.md
│   │   ├── __init__.py
│   │   ├── pyproject.toml
│   │   ├── quick_start.md
│   │   └── rust/
│   │       ├── Cargo.toml
│   │       ├── bin/
│   │       │   └── generate_constants.rs
│   │       ├── build.rs
│   │       ├── kernels/
│   │       │   ├── attention.cu
│   │       │   ├── attractor.cu
│   │       │   ├── autograd.cu
│   │       │   ├── complex_ops.cu
│   │       │   ├── conv_ops.cu
│   │       │   ├── core_ops.cu
│   │       │   ├── corrections.cu
│   │       │   ├── dgemm_native.cu
│   │       │   ├── dhsr.cu
│   │       │   ├── dhsr_fp64.cu
│   │       │   ├── e8_projection.cu
│   │       │   ├── elementwise.cu
│   │       │   ├── fixed_point.cuh
│   │       │   ├── gnosis.cu
│   │       │   ├── golden_batch_norm.cu
│   │       │   ├── golden_gelu.cu
│   │       │   ├── golden_ops.cu
│   │       │   ├── heat_kernel.cu
│   │       │   ├── hierarchy.cu
│   │       │   ├── host_wrappers.cu
│   │       │   ├── matmul.cu
│   │       │   ├── phi_residual.cu
│   │       │   ├── prime_ops.cu
│   │       │   ├── prime_selection.cu
│   │       │   ├── ptx/
│   │       │   ├── reduction.cu
│   │       │   ├── resonant_d.cu
│   │       │   ├── scatter_gather_srt.cu
│   │       │   ├── sedenion_ops.cu
│   │       │   ├── sgemm_native.cu
│   │       │   ├── srt_constants.cu
│   │       │   ├── srt_constants.cuh
│   │       │   ├── syntonic_softmax.cu
│   │       │   ├── trilinear.cu
│   │       │   ├── winding_ops.cu
│   │       │   └── wmma_syntonic.cu
│   │       ├── scripts/
│   │       │   └── generate_srt_constants.rs
│   │       └── src/
│   │           ├── autograd/
│   │           │   ├── backward.rs
│   │           │   └── mod.rs
│   │           ├── bin/
│   │           │   └── generate_srt_constants.rs
│   │           ├── build.rs
│   │           ├── constants.rs
│   │           ├── exact/
│   │           │   ├── constants.rs
│   │           │   ├── dual.rs
│   │           │   ├── fixed.rs
│   │           │   ├── golden.rs
│   │           │   ├── mod.rs
│   │           │   ├── pythagorean.rs
│   │           │   ├── rational.rs
│   │           │   ├── rotator.rs
│   │           │   ├── symexpr.rs
│   │           │   ├── syntonic.rs
│   │           │   ├── ternary_solver.rs
│   │           │   ├── traits.rs
│   │           │   └── transcendental.rs
│   │           ├── gnosis.rs
│   │           ├── golden_gelu.rs
│   │           ├── hierarchy.rs
│   │           ├── hypercomplex/
│   │           │   ├── mod.rs
│   │           │   ├── octonion.rs
│   │           │   ├── quaternion.rs
│   │           │   └── sedenion.rs
│   │           ├── lib.rs
│   │           ├── lib.rs.math_additions
│   │           ├── linalg/
│   │           │   ├── matmul.rs
│   │           │   └── mod.rs
│   │           ├── math_utils.rs
│   │           ├── prime_selection.rs
│   │           ├── resonance_test.rs
│   │           ├── resonant/
│   │           │   ├── attractor.rs
│   │           │   ├── crystallize.rs
│   │           │   ├── e8_lattice.rs
│   │           │   ├── e8_lattice_nn.rs
│   │           │   ├── evolver.rs
│   │           │   ├── golden_norm.rs
│   │           │   ├── loss.rs
│   │           │   ├── mod.rs
│   │           │   ├── number_theory.rs
│   │           │   ├── phi_ops.rs
│   │           │   ├── py_wrappers.rs
│   │           │   ├── retrocausal.rs
│   │           │   ├── syntonic_softmax.rs
│   │           │   ├── syntony.rs
│   │           │   └── tensor.rs
│   │           ├── sna/
│   │           │   ├── mod.rs
│   │           │   ├── network.rs
│   │           │   └── resonant_oscillator.rs
│   │           ├── spectral.rs
│   │           ├── tensor/
│   │           │   ├── broadcast.rs
│   │           │   ├── causal_history.rs
│   │           │   ├── conv.rs
│   │           │   ├── cuda/
│   │           │   │   ├── async_transfer.rs
│   │           │   │   ├── device_manager.rs
│   │           │   │   ├── memory_pool.rs
│   │           │   │   ├── mod.rs
│   │           │   │   ├── multi_gpu.rs
│   │           │   │   └── srt_memory_protocol.rs
│   │           │   ├── data_loading.rs
│   │           │   ├── mod.rs
│   │           │   ├── precision_policy.rs
│   │           │   ├── py_data_loading.rs
│   │           │   ├── py_srt_cuda_ops.rs
│   │           │   ├── reduction.rs
│   │           │   ├── srt_kernels.rs
│   │           │   ├── srt_optimization.rs
│   │           │   └── storage.rs
│   │           ├── transcendence.rs
│   │           ├── vibe.rs
│   │           └── winding.rs
│   ├── experiments/
│   │   ├── complex_conv.py
│   │   ├── complex_linear/
│   │   │   ├── complex_linear.py
│   │   │   └── quaternion_bimodal_dh_paper.md
│   │   ├── theory_of_mind_experiment.py
│   │   └── tree.md
│   ├── generate_api_index.py
│   ├── pyproject.toml
│   ├── python/
│   │   ├── __init__.py
│   │   ├── consciousness/
│   │   │   ├── __init__.py
│   │   │   └── gnosis.py
│   │   ├── constants.py
│   │   ├── corrections/
│   │   │   ├── __init__.py
│   │   │   └── factors.py
│   │   ├── crt/
│   │   │   ├── __init__.py
│   │   │   ├── dhsr_fused/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── dhsr_evolution.py
│   │   │   │   ├── dhsr_loop.py
│   │   │   │   └── dhsr_reference.py
│   │   │   ├── extended_hierarchy.py
│   │   │   └── operators/
│   │   │       ├── __init__.py
│   │   │       ├── base.py
│   │   │       ├── differentiation.py
│   │   │       ├── gnosis.py
│   │   │       ├── harmonization.py
│   │   │       ├── mobius.py
│   │   │       ├── projectors.py
│   │   │       ├── recursion.py
│   │   │       └── syntony.py
│   │   ├── exact/
│   │   │   └── __init__.py
│   │   ├── geometry/
│   │   │   ├── __init__.py
│   │   │   ├── torus.py
│   │   │   └── winding.py
│   │   ├── golden/
│   │   │   ├── __init__.py
│   │   │   ├── measure.py
│   │   │   └── recursion.py
│   │   ├── golden_random.py
│   │   ├── hypercomplex/
│   │   │   └── __init__.py
│   │   ├── lattice/
│   │   │   ├── __init__.py
│   │   │   ├── d4.py
│   │   │   ├── e8.py
│   │   │   ├── golden_cone.py
│   │   │   └── quadratic_form.py
│   │   ├── linalg/
│   │   │   └── __init__.py
│   │   ├── nn/
│   │   │   ├── __init__.py
│   │   │   └── resonant_tensor.py
│   │   ├── resonant/
│   │   │   ├── __init__.py
│   │   │   ├── resonant_dhsr_block.py
│   │   │   ├── resonant_embedding.py
│   │   │   ├── resonant_engine_net.py
│   │   │   ├── resonant_transformer.py
│   │   │   └── retrocausal.py
│   │   ├── sn/
│   │   │   └── __init__.py
│   │   └── spectral/
│   │       ├── __init__.py
│   │       ├── heat_kernel.py
│   │       ├── knot_laplacian.py
│   │       ├── mobius.py
│   │       └── theta_series.py
│   ├── rust/
│   │   ├── Cargo.toml
│   │   ├── bin/
│   │   │   └── generate_constants.rs
│   │   ├── build.rs
│   │   ├── kernels/
│   │   │   ├── attention.cu
│   │   │   ├── attractor.cu
│   │   │   ├── autograd.cu
│   │   │   ├── complex_ops.cu
│   │   │   ├── conv_ops.cu
│   │   │   ├── core_ops.cu
│   │   │   ├── corrections.cu
│   │   │   ├── dgemm_native.cu
│   │   │   ├── dhsr.cu
│   │   │   ├── dhsr_fp64.cu
│   │   │   ├── e8_projection.cu
│   │   │   ├── elementwise.cu
│   │   │   ├── embedding_ops.cu
│   │   │   ├── fixed_point.cuh
│   │   │   ├── functional_ops.cu
│   │   │   ├── gnosis.cu
│   │   │   ├── golden_batch_norm.cu
│   │   │   ├── golden_gelu.cu
│   │   │   ├── golden_ops.cu
│   │   │   ├── heat_kernel.cu
│   │   │   ├── hierarchy.cu
│   │   │   ├── host_wrappers.cu
│   │   │   ├── matmul.cu
│   │   │   ├── nn_ops.cu
│   │   │   ├── phase_state_ops.cu
│   │   │   ├── phi_residual.cu
│   │   │   ├── prime_ops.cu
│   │   │   ├── prime_selection.cu
│   │   │   ├── ptx/
│   │   │   ├── recurrent_ops.cu
│   │   │   ├── reduction.cu
│   │   │   ├── resonant_d.cu
│   │   │   ├── scatter_gather_srt.cu
│   │   │   ├── sedenion_ops.cu
│   │   │   ├── sgemm_native.cu
│   │   │   ├── srt_constants.cu
│   │   │   ├── srt_constants.cuh
│   │   │   ├── syntonic_softmax.cu
│   │   │   ├── trilinear.cu
│   │   │   ├── winding_ops.cu
│   │   │   └── wmma_syntonic.cu
│   │   ├── scripts/
│   │   │   └── generate_srt_constants.rs
│   │   ├── src/
│   │   │   ├── autograd/
│   │   │   │   ├── backward.rs
│   │   │   │   └── mod.rs
│   │   │   ├── bin/
│   │   │   │   └── generate_srt_constants.rs
│   │   │   ├── build.rs
│   │   │   ├── constants.rs
│   │   │   ├── exact/
│   │   │   │   ├── constants.rs
│   │   │   │   ├── dual.rs
│   │   │   │   ├── fixed.rs
│   │   │   │   ├── golden.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── pythagorean.rs
│   │   │   │   ├── rational.rs
│   │   │   │   ├── rotator.rs
│   │   │   │   ├── symexpr.rs
│   │   │   │   ├── syntonic.rs
│   │   │   │   ├── ternary_solver.rs
│   │   │   │   ├── traits.rs
│   │   │   │   └── transcendental.rs
│   │   │   ├── gnosis.rs
│   │   │   ├── golden_gelu.rs
│   │   │   ├── hierarchy.rs
│   │   │   ├── hypercomplex/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── octonion.rs
│   │   │   │   ├── quaternion.rs
│   │   │   │   └── sedenion.rs
│   │   │   ├── lib.rs
│   │   │   ├── lib.rs.math_additions
│   │   │   ├── linalg/
│   │   │   │   ├── matmul.rs
│   │   │   │   └── mod.rs
│   │   │   ├── math_utils.rs
│   │   │   ├── prime_selection.rs
│   │   │   ├── resonance_test.rs
│   │   │   ├── resonant/
│   │   │   │   ├── attractor.rs
│   │   │   │   ├── crystallize.rs
│   │   │   │   ├── e8_lattice.rs
│   │   │   │   ├── e8_lattice_nn.rs
│   │   │   │   ├── evolver.rs
│   │   │   │   ├── golden_norm.rs
│   │   │   │   ├── loss.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── number_theory.rs
│   │   │   │   ├── phi_ops.rs
│   │   │   │   ├── py_wrappers.rs
│   │   │   │   ├── retrocausal.rs
│   │   │   │   ├── syntonic_softmax.rs
│   │   │   │   ├── syntony.rs
│   │   │   │   └── tensor.rs
│   │   │   ├── sna/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── network.rs
│   │   │   │   └── resonant_oscillator.rs
│   │   │   ├── spectral.rs
│   │   │   ├── tensor/
│   │   │   │   ├── broadcast.rs
│   │   │   │   ├── causal_history.rs
│   │   │   │   ├── conv.rs
│   │   │   │   ├── cuda/
│   │   │   │   │   ├── async_transfer.rs
│   │   │   │   │   ├── device_manager.rs
│   │   │   │   │   ├── memory_pool.rs
│   │   │   │   │   ├── mod.rs
│   │   │   │   │   ├── multi_gpu.rs
│   │   │   │   │   └── srt_memory_protocol.rs
│   │   │   │   ├── data_loading.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── nn_ops.rs
│   │   │   │   ├── precision_policy.rs
│   │   │   │   ├── py_data_loading.rs
│   │   │   │   ├── py_srt_cuda_ops.rs
│   │   │   │   ├── reduction.rs
│   │   │   │   ├── srt_kernels.rs
│   │   │   │   ├── srt_optimization.rs
│   │   │   │   └── storage.rs
│   │   │   ├── transcendence.rs
│   │   │   ├── vibe.rs
│   │   │   └── winding.rs
│   │   └── tests/
│   │       └── verify_exact_math.rs
│   ├── src/
│   │   └── vibe.rs
│   ├── syntonic/
│   │   ├── __init__.py
│   │   └── crt/
│   │       └── operators/
│   │           └── harmonization.py
│   ├── test_all_kernels.py
│   ├── tests/
│   │   ├── benchmark_exact_performance.py
│   │   ├── simple_test_convergence.py
│   │   ├── test_convergence_benchmark.py
│   │   ├── test_dhsr_minimal.py
│   │   ├── test_float_isolation.py
│   │   ├── test_gnosis.py
│   │   ├── test_gnostic_shapes.py
│   │   ├── test_gradient_logic.py
│   │   ├── test_grand_synthesis.py
│   │   ├── test_pisano_transcendence.py
│   │   ├── test_prime_selection.py
│   │   ├── test_resonant_matmul.py
│   │   ├── test_sedenion.py
│   │   ├── test_sgc_reconstruction.py
│   │   ├── test_sna_genesis.py
│   │   ├── test_sna_imports.py
│   │   ├── test_sna_plasticity.py
│   │   ├── test_srt_hierarchy_integration.py
│   │   ├── test_srt_implementation.py
│   │   ├── test_state_statistics.py
│   │   ├── test_syntonic_basic.py
│   │   ├── test_syntonic_network.py
│   │   ├── test_trft.py
│   │   ├── test_trft_debug.py
│   │   └── test_viz_demo.py
│   └── tree.md
├── _float/
│   ├── CLAUDE.md
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── LICENSE-COMMERCIAL.md
│   ├── LICENSE-RESEARCH.md
│   ├── README.md
│   ├── __init__.py
│   ├── compile_kernels.py
│   ├── compile_kernels.sh
│   ├── docs/
│   │   ├── README.md
│   │   ├── exact_arithmetic_api.md
│   │   ├── exact_arithmetic_architecture.md
│   │   └── migrating_to_exact.md
│   ├── float_arithmetic/
│   │   ├── Cargo.lock
│   │   ├── Cargo.toml
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── pyproject.toml
│   │   └── rust/
│   │       ├── Cargo.toml
│   │       ├── bin/
│   │       │   └── generate_constants.rs
│   │       ├── build.rs
│   │       ├── kernels/
│   │       │   ├── attention.cu
│   │       │   ├── attractor.cu
│   │       │   ├── autograd.cu
│   │       │   ├── complex_ops.cu
│   │       │   ├── conv_ops.cu
│   │       │   ├── core_ops.cu
│   │       │   ├── corrections.cu
│   │       │   ├── dgemm_native.cu
│   │       │   ├── dhsr.cu
│   │       │   ├── dhsr_fp64.cu
│   │       │   ├── e8_projection.cu
│   │       │   ├── elementwise.cu
│   │       │   ├── fixed_point.cuh
│   │       │   ├── gnosis.cu
│   │       │   ├── golden_batch_norm.cu
│   │       │   ├── golden_gelu.cu
│   │       │   ├── golden_ops.cu
│   │       │   ├── heat_kernel.cu
│   │       │   ├── hierarchy.cu
│   │       │   ├── host_wrappers.cu
│   │       │   ├── matmul.cu
│   │       │   ├── phi_residual.cu
│   │       │   ├── prime_ops.cu
│   │       │   ├── prime_selection.cu
│   │       │   ├── ptx/
│   │       │   ├── reduction.cu
│   │       │   ├── resonant_d.cu
│   │       │   ├── scatter_gather_srt.cu
│   │       │   ├── sedenion_ops.cu
│   │       │   ├── sgemm_native.cu
│   │       │   ├── srt_constants.cu
│   │       │   ├── srt_constants.cuh
│   │       │   ├── syntonic_softmax.cu
│   │       │   ├── trilinear.cu
│   │       │   ├── winding_ops.cu
│   │       │   └── wmma_syntonic.cu
│   │       ├── scripts/
│   │       │   └── generate_srt_constants.rs
│   │       └── src/
│   │           ├── autograd/
│   │           │   ├── backward.rs
│   │           │   └── mod.rs
│   │           ├── bin/
│   │           │   └── generate_srt_constants.rs
│   │           ├── build.rs
│   │           ├── exact/
│   │           │   ├── constants.rs
│   │           │   ├── dual.rs
│   │           │   ├── fixed.rs
│   │           │   ├── golden.rs
│   │           │   ├── mod.rs
│   │           │   ├── pythagorean.rs
│   │           │   ├── rational.rs
│   │           │   ├── rotator.rs
│   │           │   ├── symexpr.rs
│   │           │   ├── syntonic.rs
│   │           │   ├── ternary_solver.rs
│   │           │   ├── traits.rs
│   │           │   └── transcendental.rs
│   │           ├── gnosis.rs
│   │           ├── golden_gelu.rs
│   │           ├── hierarchy.rs
│   │           ├── hypercomplex/
│   │           │   ├── mod.rs
│   │           │   ├── octonion.rs
│   │           │   ├── quaternion.rs
│   │           │   └── sedenion.rs
│   │           ├── lib.rs
│   │           ├── lib.rs.math_additions
│   │           ├── linalg/
│   │           │   ├── matmul.rs
│   │           │   └── mod.rs
│   │           ├── math_utils.rs
│   │           ├── prime_selection.rs
│   │           ├── resonant/
│   │           │   ├── attractor.rs
│   │           │   ├── crystallize.rs
│   │           │   ├── e8_lattice.rs
│   │           │   ├── e8_lattice_nn.rs
│   │           │   ├── evolver.rs
│   │           │   ├── golden_norm.rs
│   │           │   ├── loss.rs
│   │           │   ├── mod.rs
│   │           │   ├── number_theory.rs
│   │           │   ├── phi_ops.rs
│   │           │   ├── py_wrappers.rs
│   │           │   ├── retrocausal.rs
│   │           │   ├── syntonic_softmax.rs
│   │           │   ├── syntony.rs
│   │           │   └── tensor.rs
│   │           ├── sna/
│   │           │   ├── mod.rs
│   │           │   ├── network.rs
│   │           │   └── resonant_oscillator.rs
│   │           ├── spectral.rs
│   │           ├── tensor/
│   │           │   ├── broadcast.rs
│   │           │   ├── causal_history.rs
│   │           │   ├── conv.rs
│   │           │   ├── cuda/
│   │           │   │   ├── async_transfer.rs
│   │           │   │   ├── device_manager.rs
│   │           │   │   ├── memory_pool.rs
│   │           │   │   ├── mod.rs
│   │           │   │   ├── multi_gpu.rs
│   │           │   │   └── srt_memory_protocol.rs
│   │           │   ├── data_loading.rs
│   │           │   ├── mod.rs
│   │           │   ├── precision_policy.rs
│   │           │   ├── py_data_loading.rs
│   │           │   ├── py_srt_cuda_ops.rs
│   │           │   ├── reduction.rs
│   │           │   ├── srt_kernels.rs
│   │           │   ├── srt_optimization.rs
│   │           │   └── storage.rs
│   │           ├── transcendence.rs
│   │           ├── vibe.rs
│   │           └── winding.rs
│   ├── nn/
│   │   ├── __init__.py
│   │   └── resonant_tensor.py
│   ├── pyproject.toml
│   ├── rust/
│   │   ├── Cargo.toml
│   │   ├── bin/
│   │   │   └── generate_constants.rs
│   │   ├── build.rs
│   │   ├── kernels/
│   │   │   ├── attention.cu
│   │   │   ├── attractor.cu
│   │   │   ├── autograd.cu
│   │   │   ├── complex_ops.cu
│   │   │   ├── conv_ops.cu
│   │   │   ├── core_ops.cu
│   │   │   ├── corrections.cu
│   │   │   ├── dgemm_native.cu
│   │   │   ├── dhsr.cu
│   │   │   ├── dhsr_fp64.cu
│   │   │   ├── e8_projection.cu
│   │   │   ├── elementwise.cu
│   │   │   ├── fixed_point.cuh
│   │   │   ├── gnosis.cu
│   │   │   ├── golden_batch_norm.cu
│   │   │   ├── golden_gelu.cu
│   │   │   ├── golden_ops.cu
│   │   │   ├── heat_kernel.cu
│   │   │   ├── hierarchy.cu
│   │   │   ├── host_wrappers.cu
│   │   │   ├── matmul.cu
│   │   │   ├── phi_residual.cu
│   │   │   ├── prime_ops.cu
│   │   │   ├── prime_selection.cu
│   │   │   ├── ptx/
│   │   │   ├── reduction.cu
│   │   │   ├── resonant_d.cu
│   │   │   ├── scatter_gather_srt.cu
│   │   │   ├── sedenion_ops.cu
│   │   │   ├── sgemm_native.cu
│   │   │   ├── srt_constants.cu
│   │   │   ├── srt_constants.cuh
│   │   │   ├── syntonic_softmax.cu
│   │   │   ├── trilinear.cu
│   │   │   ├── winding_ops.cu
│   │   │   └── wmma_syntonic.cu
│   │   ├── scripts/
│   │   │   └── generate_srt_constants.rs
│   │   ├── src/
│   │   │   ├── autograd/
│   │   │   │   ├── backward.rs
│   │   │   │   └── mod.rs
│   │   │   ├── bin/
│   │   │   │   └── generate_srt_constants.rs
│   │   │   ├── build.rs
│   │   │   ├── exact/
│   │   │   │   ├── constants.rs
│   │   │   │   ├── dual.rs
│   │   │   │   ├── fixed.rs
│   │   │   │   ├── golden.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── pythagorean.rs
│   │   │   │   ├── rational.rs
│   │   │   │   ├── rotator.rs
│   │   │   │   ├── symexpr.rs
│   │   │   │   ├── syntonic.rs
│   │   │   │   ├── ternary_solver.rs
│   │   │   │   ├── traits.rs
│   │   │   │   └── transcendental.rs
│   │   │   ├── gnosis.rs
│   │   │   ├── golden_gelu.rs
│   │   │   ├── hierarchy.rs
│   │   │   ├── hypercomplex/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── octonion.rs
│   │   │   │   ├── quaternion.rs
│   │   │   │   └── sedenion.rs
│   │   │   ├── lib.rs
│   │   │   ├── lib.rs.math_additions
│   │   │   ├── linalg/
│   │   │   │   ├── matmul.rs
│   │   │   │   └── mod.rs
│   │   │   ├── math_utils.rs
│   │   │   ├── prime_selection.rs
│   │   │   ├── resonant/
│   │   │   │   ├── attractor.rs
│   │   │   │   ├── crystallize.rs
│   │   │   │   ├── e8_lattice.rs
│   │   │   │   ├── e8_lattice_nn.rs
│   │   │   │   ├── evolver.rs
│   │   │   │   ├── golden_norm.rs
│   │   │   │   ├── loss.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── number_theory.rs
│   │   │   │   ├── phi_ops.rs
│   │   │   │   ├── py_wrappers.rs
│   │   │   │   ├── retrocausal.rs
│   │   │   │   ├── syntonic_softmax.rs
│   │   │   │   ├── syntony.rs
│   │   │   │   └── tensor.rs
│   │   │   ├── sna/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── network.rs
│   │   │   │   └── resonant_oscillator.rs
│   │   │   ├── spectral.rs
│   │   │   ├── tensor/
│   │   │   │   ├── broadcast.rs
│   │   │   │   ├── causal_history.rs
│   │   │   │   ├── conv.rs
│   │   │   │   ├── cuda/
│   │   │   │   │   ├── async_transfer.rs
│   │   │   │   │   ├── device_manager.rs
│   │   │   │   │   ├── memory_pool.rs
│   │   │   │   │   ├── mod.rs
│   │   │   │   │   ├── multi_gpu.rs
│   │   │   │   │   └── srt_memory_protocol.rs
│   │   │   │   ├── data_loading.rs
│   │   │   │   ├── mod.rs
│   │   │   │   ├── precision_policy.rs
│   │   │   │   ├── py_data_loading.rs
│   │   │   │   ├── py_srt_cuda_ops.rs
│   │   │   │   ├── reduction.rs
│   │   │   │   ├── srt_kernels.rs
│   │   │   │   ├── srt_optimization.rs
│   │   │   │   └── storage.rs
│   │   │   ├── transcendence.rs
│   │   │   └── winding.rs
│   │   └── tests/
│   │       └── verify_exact_math.rs
│   └── tests/
│       ├── benchmark_exact_performance.py
│       ├── simple_test_convergence.py
│       ├── test_convergence_benchmark.py
│       ├── test_dhsr_minimal.py
│       ├── test_float_isolation.py
│       ├── test_gnosis.py
│       ├── test_gnostic_shapes.py
│       ├── test_gradient_logic.py
│       ├── test_grand_synthesis.py
│       ├── test_pisano_transcendence.py
│       ├── test_prime_selection.py
│       ├── test_resonant_matmul.py
│       ├── test_sedenion.py
│       ├── test_sgc_reconstruction.py
│       ├── test_sna_genesis.py
│       ├── test_sna_imports.py
│       ├── test_sna_plasticity.py
│       ├── test_srt_hierarchy_integration.py
│       ├── test_srt_implementation.py
│       ├── test_state_statistics.py
│       ├── test_syntonic_basic.py
│       ├── test_syntonic_network.py
│       ├── test_trft.py
│       ├── test_trft_debug.py
│       └── test_viz_demo.py
├── core/
│   ├── __init__.py
│   ├── _core/
│   │   └── __init__.py
│   ├── _version.py
│   ├── backend.py
│   ├── constants.py
│   ├── device.py
│   ├── distributed.py
│   ├── domains.py
│   ├── dtype.py
│   ├── exact.py
│   ├── exceptions.py
│   ├── fft.py
│   ├── golden.py
│   ├── hypercomplex.py
│   ├── jit.py
│   ├── linalg.py
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── activations/
│   │   │   └── gnosis_gelu.py
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── archonic_detector.py
│   │   │   ├── escape.py
│   │   │   ├── health.py
│   │   │   └── visualization.py
│   │   ├── functional.py
│   │   ├── golden_gelu.py
│   │   ├── layers/
│   │   │   ├── __init__.py
│   │   │   ├── attention.py
│   │   │   ├── conv.py
│   │   │   ├── differentiation.py
│   │   │   ├── embedding.py
│   │   │   ├── gnosis.py
│   │   │   ├── harmonization.py
│   │   │   ├── normalization.py
│   │   │   ├── pixel_ops.py
│   │   │   ├── prime_gate.py
│   │   │   ├── prime_syntony_gate.py
│   │   │   ├── recurrent.py
│   │   │   ├── recursion.py
│   │   │   ├── resonant_linear.py
│   │   │   ├── resonant_parameter.py
│   │   │   └── syntonic_gate.py
│   │   ├── loss/
│   │   │   ├── __init__.py
│   │   │   ├── phase_alignment.py
│   │   │   ├── regularization.py
│   │   │   ├── syntonic_loss.py
│   │   │   └── syntony_metrics.py
│   │   ├── optim/
│   │   │   ├── __init__.py
│   │   │   └── golden_momentum.py
│   │   ├── resonant_tensor.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── callbacks.py
│   │   │   ├── gradient_trainer.py
│   │   │   ├── metrics.py
│   │   │   └── trainer.py
│   │   └── winding/
│   │       ├── __init__.py
│   │       ├── dhsr_block.py
│   │       ├── embedding.py
│   │       ├── fermat.py
│   │       ├── fibonacci_hierarchy.py
│   │       ├── lucas.py
│   │       ├── mersenne.py
│   │       ├── prime_selection.py
│   │       ├── syntony.py
│   │       ├── winding_encoder.py
│   │       └── winding_net.py
│   ├── prompt_core/
│   │   └── templates/
│   ├── server.py
│   ├── sn.py
│   ├── srt_math.py
│   ├── srt_random.py
│   ├── state.py
│   └── types.py
├── generate_tree.py
├── phase_state_benchmarks.py
├── phase_state_benchmarks2.py
├── phase_state_vibes_compiler.py
├── theory_unique_components/
│   ├── GnosticOuroboros/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   └── ouroboros_daemon.py
│   │   ├── g_comms.py
│   │   ├── gnostic_ouroboros.py
│   │   ├── helpers.py
│   │   ├── io/
│   │   │   └── flux_bridge.py
│   │   └── winding_chain.py
│   ├── __init__.py
│   ├── cosmological_block.py
│   ├── crt/
│   │   ├── __init__.py
│   │   ├── dhsr_fused/
│   │   │   ├── __init__.py
│   │   │   ├── dhsr_evolution.py
│   │   │   ├── dhsr_loop.py
│   │   │   └── dhsr_reference.py
│   │   ├── extended_hierarchy.py
│   │   └── operators/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── differentiation.py
│   │       ├── gnosis.py
│   │       ├── harmonization.py
│   │       ├── mobius.py
│   │       ├── projectors.py
│   │       ├── recursion.py
│   │       └── syntony.py
│   ├── embeddings.py
│   ├── prime_syntony_gate.py
│   ├── resonant/
│   │   ├── __init__.py
│   │   ├── resonant_dhsr_block.py
│   │   ├── resonant_embedding.py
│   │   ├── resonant_engine_net.py
│   │   ├── resonant_transformer.py
│   │   └── retrocausal.py
│   ├── srt/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── corrections/
│   │   │   ├── __init__.py
│   │   │   └── factors.py
│   │   ├── fermat_forces.py
│   │   ├── functional/
│   │   │   ├── __init__.py
│   │   │   └── syntony.py
│   │   ├── geometry/
│   │   │   ├── __init__.py
│   │   │   ├── torus.py
│   │   │   └── winding.py
│   │   ├── golden/
│   │   │   ├── __init__.py
│   │   │   ├── measure.py
│   │   │   └── recursion.py
│   │   ├── lattice/
│   │   │   ├── __init__.py
│   │   │   ├── d4.py
│   │   │   ├── e8.py
│   │   │   ├── golden_cone.py
│   │   │   └── quadratic_form.py
│   │   ├── lucas_shadow.py
│   │   ├── mersenne_matter.py
│   │   ├── prime_selection.py
│   │   ├── spectral/
│   │   │   ├── __init__.py
│   │   │   ├── heat_kernel.py
│   │   │   ├── knot_laplacian.py
│   │   │   ├── mobius.py
│   │   │   └── theta_series.py
│   │   └── transcendence.py
│   ├── syntonic_attention.py
│   ├── syntonic_cnn.py
│   ├── syntonic_mlp.py
│   └── syntonic_transformer.py
└── tree.md
```
