# FlagGems: Accelerate LLM Training and Inference with High-Performance Operators

**Supercharge your LLM performance with FlagGems, a high-performance operator library built on OpenAI Triton, offering seamless PyTorch integration and broad hardware support.** ([See the original repo](https://github.com/FlagOpen/FlagGems))

![FlagGems Overview](https://github.com/user-attachments/assets/97950fc6-62bb-4b6a-b8d5-5751c14492fa)

## Key Features of FlagGems

*   **PyTorch Compatibility:** Seamlessly integrates with PyTorch using familiar APIs.
*   **Optimized Performance:** Hand-optimized operators deliver significant speedups.
*   **Eager Mode Ready:** Operates independently of `torch.compile`.
*   **Automatic Codegen:** Generates pointwise and fused operators efficiently.
*   **Fast Kernel Dispatch:** Optimized runtime kernel dispatching for speed.
*   **Multi-Backend Support:** Enables support for diverse hardware platforms, maximizing hardware utilization.
*   **Wide Operator Coverage:** Supports a growing list of operators for various LLM tasks.
*   **C++ Triton Function Dispatcher:** Boosts end-to-end performance.

## Why Choose FlagGems?

FlagGems provides a compelling solution for accelerating LLM workloads:

*   **Performance:** Achieve significant speedups compared to standard PyTorch.
*   **Ease of Use:** Leverage your existing PyTorch knowledge with minimal code changes.
*   **Flexibility:** Supports a wide range of hardware platforms, including those from AI chip vendors.
*   **Developer-Friendly:** Triton language offers readability and ease of development.

## Core Components & Technologies

FlagGems leverages key technologies to achieve high performance and flexibility:

*   **OpenAI Triton:**  A high-performance language and compiler for GPU programming.
*   **ATen Backend:** Integrates with PyTorch's ATen backend, allowing for a smooth transition.
*   **LibEntry:** A mechanism for managing kernel cache and bypassing the runtime of Autotuner, Heuristics, and JitFunction to improve performance.
*   **C++ Runtime:** Improves end-to-end performance.

## Supported Operators & Hardware

### Supported Operators (v3.0)
FlagGems supports a comprehensive set of operators, constantly expanding to meet the needs of modern LLMs. Here is a partial list, see the full details in the original repository:

*   184 Operators in total
*   Basic Math Operators: allclose, isclose, isfinite, floor_divide, trunc_divide, maximum, minimum
*   BLAS Operators: addmm, bmm, mm, mv, outer
*   Pointwise Operators: abs, add, div, dropout, exp, gelu, mul, pow, reciprocal, relu, rsqrt, silu, sub, triu, bitwise_and, bitwise_not, bitwise_or, cos, clamp, eq, ge, gt, isinf, isnan, le, lt, ne, neg, or, sin, tanh, sigmoid
*   Reduction Operators: all, any, amax, argmax, max, min, prod, sum, var_mean, vector_norm, cross_entropy_loss, group_norm, log_softmax, rms_norm
*   Fused Operators: fused_add_rms_norm, skip_layer_norm, gelu_and_mul, silu_and_mul, apply_rotary_position_embedding
*   Distribution Operators: normal, uniform\_, exponential\_, multinomial, nonzero, topk, rand, randn, rand_like, randn_like
*   Science Operators: erf, resolve_conj, resolve_neg
*   Tensor Operators: where, arange, repeat, masked_fill, tile, unique, index_select, masked_select, ones, ones_like, zeros, zeros_like, full, full_like, flip, pad
*   Neural Network Operators: embedding

### Supported Platforms

FlagGems offers broad hardware support. The table below shows supported platforms and their current state of support.

| Vendor      | State                      | float16 | float32 | bfloat16 |
| ----------- | -------------------------- | ------- | ------- | -------- |
| aipu        | ✅ （Partial support）     | ✅      | ✅      | ✅       |
| ascend      | ✅ （Partial support）     | ✅      | ✅      | ✅       |
| cambricon   | ✅                         | ✅      | ✅      | ✅       |
| hygon       | ✅                         | ✅      | ✅      | ✅       |
| iluvatar    | ✅                         | ✅      | ✅      | ✅       |
| kunlunxin   | ✅                         | ✅      | ✅      | ✅       |
| metax       | ✅                         | ✅      | ✅      | ✅       |
| mthreads    | ✅                         | ✅      | ✅      | ✅       |
| nvidia      | ✅                         | ✅      | ✅      | ✅       |
| arm(cpu)    | 🚧                         |         |         |          |
| tsingmicro  | 🚧                         |         |         |          |

## Performance Gains

FlagGems delivers significant performance improvements compared to the PyTorch ATen library.

![Operator Speedup](./docs/assets/speedup-20250423.png)

## Get Started

To begin using FlagGems, refer to the documentation [GetStart](docs/get_start_with_flaggems.md) for installation and usage instructions.

## Contribute

We welcome contributions to FlagGems! For guidelines, see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Cite Us

If you use FlagGems in your research, please cite our project:

```bibtex
@misc{flaggems2024,
    title={FlagOpen/FlagGems: FlagGems is an operator library for large language models implemented in the Triton language.},
    url={https://github.com/FlagOpen/FlagGems},
    journal={GitHub},
    author={BAAI FlagOpen team},
    year={2024}
}
```

## Contact Us

For questions or feedback, please submit an issue or contact us at <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.