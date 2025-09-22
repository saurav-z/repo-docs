# FlagGems: Accelerate LLM Training and Inference with High-Performance Operators

**FlagGems provides a high-performance operator library, built with OpenAI Triton, to supercharge your LLM training and inference across diverse hardware platforms.**

[Visit the original repository on GitHub](https://github.com/FlagOpen/FlagGems)

## Key Features

*   **Extensive PyTorch Compatibility:** Seamlessly integrates with PyTorch using familiar APIs.
*   **Optimized Performance:** Hand-optimized kernels for select operators, delivering significant speedups.
*   **Eager Mode Ready:** Works directly without requiring `torch.compile`.
*   **Automatic Codegen:** Pointwise operator code generation supporting various data types and layouts.
*   **Fast Kernel Dispatching:** Rapid per-function runtime kernel dispatching for optimal performance.
*   **Multi-Backend Support:** Supports a wide array of hardware platforms.
*   **C++ Runtime:** Includes a C++ runtime to minimize Python overhead and boost end-to-end performance.

## What is FlagGems?

FlagGems is designed to accelerate large language model (LLM) training and inference. It leverages the power of [OpenAI Triton](https://github.com/openai/triton) to provide a collection of high-performance operators, offering a significant performance boost over traditional PyTorch implementations.  It allows developers to leverage new hardware acceleration technologies without changing the low-level APIs, and making LLM development more efficient and accessible.

## Technical Highlights

### Multi-Backend Hardware Support

FlagGems is designed to run on a variety of hardware platforms, and has been thoroughly tested on multiple configurations. See the "Supported Platforms" section below for details.

### Automatic Codegen

FlagGems offers automated code generation for pointwise and fused operators, enabling easy creation of operators. The auto-generation system supports standard element-wise calculations, non-tensor parameters, and output types.

### LibEntry

FlagGems introduces `LibEntry`, which simplifies kernel cache management and avoids unnecessary runtime overhead. Decorating Triton kernels with `LibEntry` gives a more streamlined tuning and eliminates redundant parameter processing.

### C++ Runtime

The C++ runtime enhances the overall performance, reducing the overhead that exists in the Python runtime.

## Changelog

FlagGems is continuously evolving, with ongoing improvements and operator support. Notable updates include:

### v3.0

*   Support for 184 operators, including custom operators used in LLM inference.
*   Expanded hardware platform support, including Ascend and AIPU.
*   Compatibility with the vLLM framework.

### v2.1 & v2.0 & v1.0

*   Added support for various operators including Tensor, neural network, basic math, distribution, and science operators.
*   Expanded support for BLAS, pointwise, reduction, and fused operators.

## Getting Started

For detailed installation and usage instructions, please refer to the [Get Started with FlagGems documentation](docs/get_start_with_flaggems.md).

## Supported Operators

See the [Operator List](docs/operator_list.md) for the current list of supported operators.

## Example Models

*   Bert-base-uncased
*   Llama-2-7b
*   Llava-1.5-7b

## Supported Platforms

| Vendor     | State                  | float16 | float32 | bfloat16 |
| ---------- | ---------------------- | ------- | ------- | -------- |
| aipu       | ✅ （Partial support） | ✅      | ✅      | ✅       |
| ascend     | ✅ （Partial support） | ✅      | ✅      | ✅       |
| cambricon  | ✅                     | ✅      | ✅      | ✅       |
| hygon      | ✅                     | ✅      | ✅      | ✅       |
| iluvatar   | ✅                     | ✅      | ✅      | ✅       |
| kunlunxin  | ✅                     | ✅      | ✅      | ✅       |
| metax      | ✅                     | ✅      | ✅      | ✅       |
| mthreads   | ✅                     | ✅      | ✅      | ✅       |
| nvidia     | ✅                     | ✅      | ✅      | ✅       |
| arm(cpu)   | 🚧                     |         |         |          |
| tsingmicro | 🚧                     |         |         |          |

## Performance

FlagGems delivers impressive performance gains.  The following chart shows the speedup of FlagGems compared with PyTorch ATen library in eager mode. The speedup is calculated by averaging the speedup on each shape, representing the overall performance of the operator.

![Operator Speedup](./docs/assets/speedup-20250423.png)

## Contributing

We welcome contributions!  Please see the [CONTRIBUTING.md](./CONTRIBUTING.md) file for guidelines.

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

For questions or support, please submit an issue or contact us at <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

FlagGems is released under the [Apache 2.0](./LICENSE) license.