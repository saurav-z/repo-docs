# FlagGems: Accelerate LLM Training and Inference with High-Performance Operators

**Supercharge your Large Language Model (LLM) performance with FlagGems, a cutting-edge operator library built on OpenAI Triton, offering significant speedups across diverse hardware platforms.** ([Original Repo](https://github.com/FlagOpen/FlagGems))

![FlagGems Overview](https://github.com/user-attachments/assets/97950fc6-62bb-4b6a-b8d5-5751c14492fa)

## Key Features:

*   **PyTorch Compatibility:** Seamless integration with existing PyTorch workflows, allowing you to leverage Triton acceleration without code modification.
*   **Optimized Operators:** Hand-optimized performance for selective operators, maximizing efficiency for LLM workloads.
*   **Eager Mode Ready:** Operates effectively in eager mode, providing flexibility and compatibility independently of `torch.compile`.
*   **Automatic Codegen:**  Automated pointwise operator code generation that supports arbitrary input types and layouts, simplifying development.
*   **Fast Kernel Dispatch:** Efficient runtime kernel dispatching for optimal performance.
*   **Multi-Backend Support:**  Supports a wide range of hardware platforms, enabling broad accessibility and hardware flexibility.
*   **C++ Runtime:** Provides a C++ runtime to reduce Python overhead and boost end-to-end performance.
*   **LibEntry:** Manages kernel cache and bypasses runtime operations, and offers direct wrapping for functionality.

## Supported Operators & Features

FlagGems provides a comprehensive suite of operators, including:

*   **BLAS Operators:** `addmm`, `bmm`, `mm`, `mv`, `outer`
*   **Pointwise Operators:** `abs`, `add`, `div`, `dropout`, `exp`, `gelu`, `mul`, `pow`, `reciprocal`, `relu`, `rsqrt`, `silu`, `sub`, `triu`, `bitwise_and`, `bitwise_not`, `bitwise_or`, `cos`, `clamp`, `eq`, `ge`, `gt`, `isinf`, `isnan`, `le`, `lt`, `ne`, `neg`, `or`, `sin`, `tanh`, `sigmoid`, `allclose`, `isclose`, `isfinite`, `floor_divide`, `trunc_divide`, `maximum`, `minimum`
*   **Reduction Operators:** `all`, `any`, `amax`, `argmax`, `max`, `min`, `prod`, `sum`, `var_mean`, `vector_norm`, `cross_entropy_loss`, `group_norm`, `log_softmax`, `rms_norm`
*   **Fused Operators:** `fused_add_rms_norm`, `skip_layer_norm`, `gelu_and_mul`, `silu_and_mul`, `apply_rotary_position_embedding`
*   **Tensor Operators:** `where`, `arange`, `repeat`, `masked_fill`, `tile`, `unique`, `index_select`, `masked_select`, `ones`, `ones_like`, `zeros`, `zeros_like`, `full`, `full_like`, `flip`, `pad`
*   **Neural Network Operators:** `embedding`
*   **Distribution Operators:** `normal`, `uniform_`, `exponential_`, `multinomial`, `nonzero`, `topk`, `rand`, `randn`, `rand_like`, `randn_like`
*   **Science Operators:** `erf`, `resolve_conj`, `resolve_neg`

### Multi-Backend Hardware Support

FlagGems supports a wide array of hardware platforms:

| Vendor       | Status                   | float16 | float32 | bfloat16 |
|--------------|--------------------------|---------|---------|----------|
| aipu         | ✅ (Partial support)      | ✅      | ✅      | ✅        |
| ascend       | ✅ (Partial support)      | ✅      | ✅      | ✅        |
| cambricon    | ✅                      | ✅      | ✅      | ✅        |
| hygon        | ✅                      | ✅      | ✅      | ✅        |
| iluvatar     | ✅                      | ✅      | ✅      | ✅        |
| kunlunxin    | ✅                      | ✅      | ✅      | ✅        |
| metax        | ✅                      | ✅      | ✅      | ✅        |
| mthreads     | ✅                      | ✅      | ✅      | ✅        |
| nvidia       | ✅                      | ✅      | ✅      | ✅        |
| arm (CPU)    | 🚧                     |         |         |          |
| tsingmicro   | 🚧                     |         |         |          |

### Automatic Codegen

FlagGems' automatic code generation mechanism simplifies the creation of pointwise and fused operators.  For detailed information, see [pointwise\_dynamic](docs/pointwise_dynamic.md).

### LibEntry

FlagGems introduces `LibEntry` to efficiently manage kernel caches, and also supports direct wrapping of functionality.

## Performance

FlagGems provides significant speedups compared to PyTorch ATen, as demonstrated in the following chart:

![Operator Speedup](./docs/assets/speedup-20250423.png)

## Getting Started

Refer to the documentation for installation and usage: [GetStart](docs/get_start_with_flaggems.md).

## Example Models

FlagGems is compatible with several popular models:

*   Bert-base-uncased
*   Llama-2-7b
*   Llava-1.5-7b

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## Citation

If you find FlagGems helpful, please cite our project:

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

For any questions or inquiries, please submit an issue or contact us at <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

FlagGems is licensed under [Apache 2.0](./LICENSE).