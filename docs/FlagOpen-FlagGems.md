# FlagGems: Accelerate LLM Training & Inference with High-Performance Operators

**FlagGems unlocks significant performance gains for LLM training and inference by leveraging the power of OpenAI Triton and PyTorch, enabling developers to seamlessly accelerate their models across diverse hardware platforms.**  [View the original repository](https://github.com/FlagOpen/FlagGems)

## Key Features of FlagGems

*   **Extensive Operator Library:** Provides a comprehensive collection of PyTorch-compatible operators.
*   **Optimized Performance:** Hand-optimized kernels for select operators, maximizing efficiency.
*   **Eager Mode Ready:** Works seamlessly with eager mode, independent of `torch.compile`.
*   **Automated Code Generation:**  Generates pointwise and fused operators with automatic support for various data types and layouts.
*   **Fast Kernel Dispatching:**  Employs a rapid per-function runtime kernel dispatching mechanism.
*   **Multi-Backend Support:**  Enables hardware acceleration across numerous platforms, offering flexibility.
*   **C++ Triton Function Dispatcher:** (Under development) Aims to reduce Python runtime overhead, boosting performance.

## Why Choose FlagGems?

FlagGems empowers developers to accelerate LLM training and inference without complex code changes. It bridges the gap between familiar PyTorch APIs and the performance advantages of Triton, offering a user-friendly experience and efficient hardware utilization.

## Enhanced Features & Capabilities

### Multi-Backend Hardware Support

FlagGems excels in supporting a broad spectrum of hardware platforms. The library has undergone extensive testing and is validated across multiple configurations.

### Automatic Codegen

FlagGems streamlines operator development by providing automatic code generation capabilities for pointwise and fused operators.  This system supports diverse requirements, including standard element-wise calculations and non-tensor parameters.  For more details, explore the [pointwise_dynamic documentation](docs/pointwise_dynamic.md).

### LibEntry

FlagGems introduces `LibEntry`, a mechanism to manage the kernel cache and bypass the runtime of `Autotuner`, `Heuristics`, and `JitFunction`. This approach boosts efficiency and reduces runtime overhead.

`LibEntry` also provides options to directly wrap `Autotuner`, `Heuristics`, and `JitFunction`, maintaining full tuning capabilities. This means a simplified cache key format and reduced key computation.

### C++ Runtime

FlagGems offers a C++ runtime option to eliminate Python runtime overhead, enhancing overall performance.

## Changelog

### v3.0

*   Support for a total of 184 operators, including custom operators used in large model inference.
*   Expanded hardware platform support, including Ascend and AIPU.
*   Compatibility with the vLLM framework, verified through DeepSeek model inference testing.

### v2.1

*   Support for various Tensor operators: `where`, `arange`, `repeat`, `masked_fill`, `tile`, `unique`, `index_select`, `masked_select`, `ones`, `ones_like`, `zeros`, `zeros_like`, `full`, `full_like`, `flip`, `pad`.
*   Support for neural network operator: `embedding`.
*   Support for basic math operators: `allclose`, `isclose`, `isfinite`, `floor_divide`, `trunc_divide`, `maximum`, `minimum`.
*   Support for distribution operators: `normal`, `uniform_`, `exponential_`, `multinomial`, `nonzero`, `topk`, `rand`, `randn`, `rand_like`, `randn_like`.
*   Support for science operators: `erf`, `resolve_conj`, `resolve_neg`.

### v2.0

*   Support for BLAS operators: `mv`, `outer`.
*   Support for pointwise operators: `bitwise_and`, `bitwise_not`, `bitwise_or`, `cos`, `clamp`, `eq`, `ge`, `gt`, `isinf`, `isnan`, `le`, `lt`, `ne`, `neg`, `or`, `sin`, `tanh`, `sigmoid`.
*   Support for reduction operators: `all`, `any`, `amax`, `argmax`, `max`, `min`, `prod`, `sum`, `var_mean`, `vector_norm`, `cross_entropy_loss`, `group_norm`, `log_softmax`, `rms_norm`.
*   Support for fused operators: `fused_add_rms_norm`, `skip_layer_norm`, `gelu_and_mul`, `silu_and_mul`, `apply_rotary_position_embedding`.

### v1.0

*   Support for BLAS operators: `addmm`, `bmm`, `mm`.
*   Support for pointwise operators: `abs`, `add`, `div`, `dropout`, `exp`, `gelu`, `mul`, `pow`, `reciprocal`, `relu`, `rsqrt`, `silu`, `sub`, `triu`.
*   Support for reduction operators: `cumsum`, `layernorm`, `mean`, `softmax`.

## Getting Started

Get up and running with FlagGems quickly! Check out the documentation at [GetStart](docs/get_start_with_flaggems.md) for installation and usage instructions.

## Supported Operators

View the planned operator implementations in [OperatorList](docs/operator_list.md).

## Example Models

*   Bert-base-uncased
*   Llama-2-7b
*   Llava-1.5-7b

## Supported Platforms

| Vendor       | State                 | float16 | float32 | bfloat16 |
| ------------ | --------------------- | ------- | ------- | -------- |
| aipu         | ✅ (Partial support)  | ✅      | ✅      | ✅       |
| ascend       | ✅ (Partial support)  | ✅      | ✅      | ✅       |
| cambricon    | ✅                    | ✅      | ✅      | ✅       |
| hygon        | ✅                    | ✅      | ✅      | ✅       |
| iluvatar     | ✅                    | ✅      | ✅      | ✅       |
| kunlunxin    | ✅                    | ✅      | ✅      | ✅       |
| metax        | ✅                    | ✅      | ✅      | ✅       |
| mthreads     | ✅                    | ✅      | ✅      | ✅       |
| nvidia       | ✅                    | ✅      | ✅      | ✅       |
| arm(cpu)     | 🚧                   |         |         |          |
| tsingmicro   | 🚧                   |         |         |          |

## Performance

![Operator Speedup](./docs/assets/speedup-20250423.png)

The provided chart illustrates the speedup achieved by FlagGems compared to the PyTorch ATen library when operating in eager mode. Speedup values are derived by averaging performance gains across various shapes, providing a comprehensive overview of operator performance.

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](./CONTRIBUTING.md) for details on how to get involved.

## Citation

If you utilize FlagGems in your work, please cite our project:

```bibtex
@misc{flaggems2024,
    title={FlagOpen/FlagGems: FlagGems is an operator library for large language models implemented in the Triton language.},
    url={https://github.com/FlagOpen/FlagGems},
    journal={GitHub},
    author={BAAI FlagOpen team},
    year={2024}
}
```

## Contact

For questions, please submit an issue, or contact us at <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

FlagGems is licensed under the [Apache 2.0](./LICENSE) license.