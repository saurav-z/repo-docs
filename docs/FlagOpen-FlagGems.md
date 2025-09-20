# FlagGems: Accelerate LLM Training and Inference with High-Performance Operators

**FlagGems is a cutting-edge operator library built on OpenAI Triton, designed to supercharge large language model (LLM) training and inference across diverse hardware platforms.**  [View the original repository on GitHub](https://github.com/FlagOpen/FlagGems)

![FlagGems Logo](https://github.com/user-attachments/assets/97950fc6-62bb-4b6a-b8d5-5751c14492fa)

## Key Features & Benefits

*   **Extensive PyTorch Compatibility:** Seamlessly integrates with existing PyTorch workflows.
*   **Optimized Performance:** Hand-tuned performance for selected operators, achieving significant speedups.
*   **Eager Mode Ready:** Works independently of `torch.compile`, offering flexibility.
*   **Automatic Code Generation:** Simplifies operator development with support for various data types and layouts.
*   **Fast Kernel Dispatch:** Efficient per-function runtime kernel dispatching for rapid execution.
*   **Multi-Backend Support:** Enables support for a wide range of hardware platforms, including GPUs and specialized AI accelerators.
*   **C++ Runtime:** Provides a C++ runtime option to minimize overhead and improve end-to-end performance.

## Supported Operators

FlagGems offers a growing collection of PyTorch-compatible operators, categorized for easy navigation:

*   **BLAS Operators:** `addmm`, `bmm`, `mm`, `mv`, `outer`
*   **Pointwise Operators:** `abs`, `add`, `div`, `dropout`, `exp`, `gelu`, `mul`, `pow`, `reciprocal`, `relu`, `rsqrt`, `silu`, `sub`, `triu`, `bitwise_and`, `bitwise_not`, `bitwise_or`, `cos`, `clamp`, `eq`, `ge`, `gt`, `isinf`, `isnan`, `le`, `lt`, `ne`, `neg`, `or`, `sin`, `tanh`, `sigmoid`
*   **Reduction Operators:** `all`, `any`, `amax`, `argmax`, `max`, `min`, `prod`, `sum`, `var_mean`, `vector_norm`, `cross_entropy_loss`, `group_norm`, `log_softmax`, `rms_norm`
*   **Fused Operators:** `fused_add_rms_norm`, `skip_layer_norm`, `gelu_and_mul`, `silu_and_mul`, `apply_rotary_position_embedding`
*   **Tensor Operators:** `where`, `arange`, `repeat`, `masked_fill`, `tile`, `unique`, `index_select`, `masked_select`, `ones`, `ones_like`, `zeros`, `zeros_like`, `full`, `full_like`, `flip`, `pad`
*   **Neural Network Operator:** `embedding`
*   **Basic Math Operators:** `allclose`, `isclose`, `isfinite`, `floor_divide`, `trunc_divide`, `maximum`, `minimum`
*   **Distribution Operators:** `normal`, `uniform_`, `exponential_`, `multinomial`, `nonzero`, `topk`, `rand`, `randn`, `rand_like`, `randn_like`
*   **Science Operators:** `erf`, `resolve_conj`, `resolve_neg`

Refer to the [OperatorList](docs/operator_list.md) for a detailed list and future planned implementations.

## Core Technologies

### Multi-Backend Hardware Support

FlagGems' flexible architecture supports a broad spectrum of hardware, maximizing performance across different environments.

### Automatic Codegen

FlagGems features an automatic code generation system, simplifying the development of both pointwise and fused operators, supporting various needs. Learn more in [pointwise_dynamic](docs/pointwise_dynamic.md).

### LibEntry

`LibEntry` offers independent kernel cache management, bypassing runtime components for efficiency gains. It also supports direct wrapping of `Autotuner`, `Heuristics`, and `JitFunction`, minimizing redundant operations.

### C++ Runtime

The C++ runtime significantly boosts end-to-end performance by mitigating Python runtime overhead.

## Supported Platforms

FlagGems provides optimized performance across a variety of hardware platforms:

| Vendor      | State                     | float16 | float32 | bfloat16 |
| ----------- | ------------------------- | ------- | ------- | -------- |
| aipu        | ✅ (Partial support)      | ✅      | ✅      | ✅       |
| ascend      | ✅ (Partial support)      | ✅      | ✅      | ✅       |
| cambricon   | ✅                        | ✅      | ✅      | ✅       |
| hygon       | ✅                        | ✅      | ✅      | ✅       |
| iluvatar    | ✅                        | ✅      | ✅      | ✅       |
| kunlunxin   | ✅                        | ✅      | ✅      | ✅       |
| metax       | ✅                        | ✅      | ✅      | ✅       |
| mthreads    | ✅                        | ✅      | ✅      | ✅       |
| nvidia      | ✅                        | ✅      | ✅      | ✅       |
| arm(cpu)    | 🚧                        |         |         |          |
| tsingmicro  | 🚧                        |         |         |          |

## Performance

FlagGems delivers significant speedups compared to PyTorch ATen, particularly in eager mode.

![Operator Speedup](./docs/assets/speedup-20250423.png)

## Getting Started

For installation and usage instructions, please refer to the [GetStart](docs/get_start_with_flaggems.md) documentation.

## Example Models

*   Bert-base-uncased
*   Llama-2-7b
*   Llava-1.5-7b

## Contributing

Contributions to FlagGems are welcome! Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Citation

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

## Contact

For questions or inquiries, please submit an issue or contact us at <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

FlagGems is licensed under the [Apache 2.0](./LICENSE) license.