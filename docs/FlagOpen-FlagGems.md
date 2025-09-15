# FlagGems: Accelerate LLM Training and Inference with High-Performance Operators

**FlagGems revolutionizes LLM performance by providing a library of high-performance operators implemented in OpenAI Triton, enabling faster training and inference across diverse hardware platforms.** ([Original Repository](https://github.com/FlagOpen/FlagGems))

[![FlagGems Overview](https://github.com/user-attachments/assets/97950fc6-62bb-4b6a-b8d5-5751c14492fa)](https://github.com/FlagOpen/FlagGems)

## Key Features

*   **PyTorch Compatibility:** Offers a wide range of PyTorch-compatible operators, enabling seamless integration with existing models.
*   **Optimized Performance:** Hand-optimized for select operators to ensure top-tier performance.
*   **Eager Mode Ready:** Supports eager mode operation, independent of `torch.compile`, for flexibility in development.
*   **Automated Codegen:** Features an automatic pointwise operator codegen supporting arbitrary input types and layout, boosting developer productivity.
*   **Fast Kernel Dispatch:** Provides efficient per-function runtime kernel dispatching for optimal performance.
*   **Multi-Backend Support:**  Offers a multi-backend interface, supporting diverse hardware platforms for broad applicability.
*   **C++ Runtime:** Optional C++ extensions to reduce Python runtime overhead and improve end-to-end performance.
*   **LibEntry:** Optimizes kernel cache management with LibEntry to bypass unnecessary runtime components and speed up execution.

## Why Choose FlagGems?

FlagGems provides model developers with a powerful solution to harness the performance benefits of Triton without requiring extensive code changes. Kernel developers will appreciate the readability and usability of the Triton language. Experience significantly faster LLM training and inference through the FlagGems operator library, which is designed for acceleration across a wide variety of hardware platforms.

## Key Advantages

*   **Ease of Use:** Utilize familiar PyTorch APIs and benefit from hardware acceleration without code modifications.
*   **High Performance:** Leverage hand-optimized and automatically generated operators for maximum efficiency.
*   **Hardware Agnostic:** Achieve hardware-agnostic performance, supporting multiple backends and a variety of hardware platforms.

## Changelog

### v3.0

*   Support for 184 operators, including custom operators for large model inference.
*   Enhanced hardware platform support, including Ascend and AIPU.
*   Compatibility with the vLLM framework, with successful inference verification on the DeepSeek model.

### v2.1

*   Added support for various tensor, neural network, basic math, and distribution operators.

### v2.0

*   Support for BLAS, pointwise, reduction, and fused operators.

### v1.0

*   Initial release supporting BLAS, pointwise, and reduction operators.

## Getting Started

For detailed installation and usage instructions, see the [GetStart](docs/get_start_with_flaggems.md) documentation.

## Supported Operators

Consult the [OperatorList](docs/operator_list.md) for a complete list of implemented operators.

## Example Models

*   Bert-base-uncased
*   Llama-2-7b
*   Llava-1.5-7b

## Supported Platforms

| Vendor       | State                  | float16 | float32 | bfloat16 |
|--------------|------------------------|---------|---------|----------|
| aipu         | ✅ (Partial support)  | ✅      | ✅      | ✅       |
| ascend       | ✅ (Partial support)  | ✅      | ✅      | ✅       |
| cambricon    | ✅                     | ✅      | ✅      | ✅       |
| hygon        | ✅                     | ✅      | ✅      | ✅       |
| iluvatar     | ✅                     | ✅      | ✅      | ✅       |
| kunlunxin    | ✅                     | ✅      | ✅      | ✅       |
| metax        | ✅                     | ✅      | ✅      | ✅       |
| mthreads     | ✅                     | ✅      | ✅      | ✅       |
| nvidia       | ✅                     | ✅      | ✅      | ✅       |
| arm(cpu)     | 🚧                     |         |         |          |
| tsingmicro   | 🚧                     |         |         |          |

## Performance

[Insert Image of Performance Graph -  Operator Speedup](./docs/assets/speedup-20250423.png)

## Contribute

We welcome contributions! Please review [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Citation

If you find FlagGems helpful in your work, please cite our project:

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

For any inquiries or issues, submit an issue or contact us at <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

FlagGems is licensed under the [Apache 2.0](./LICENSE) license.