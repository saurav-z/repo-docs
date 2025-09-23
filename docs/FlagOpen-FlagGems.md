# FlagGems: Accelerate LLM Training and Inference with High-Performance Operators (Triton-Based)

**FlagGems is a high-performance operator library built on OpenAI Triton, designed to supercharge LLM training and inference across diverse hardware platforms.**  Explore the power of optimized kernels and seamless integration with PyTorch.  [Visit the original repository](https://github.com/FlagOpen/FlagGems).

[![FlagGems Logo](https://github.com/user-attachments/assets/97950fc6-62bb-4b6a-b8d5-5751c14492fa)](https://github.com/FlagOpen/FlagGems)

## Key Features

*   **Extensive PyTorch Compatibility:** Seamlessly integrates with your existing PyTorch workflows.
*   **Optimized Performance:** Hand-optimized kernels for selective operators, delivering significant speedups.
*   **Eager Mode Ready:** Operates independently of `torch.compile` for flexible usage.
*   **Automatic Code Generation:** Generates pointwise and fused operators, simplifying development.
*   **Fast Kernel Dispatching:** Efficient runtime kernel dispatching for optimized performance.
*   **Multi-Backend Support:** Enables acceleration across a wide range of hardware platforms.
*   **C++ Triton Function Dispatcher:** (Work in Progress) Further enhances performance.
*   **LibEntry:** Provides a dedicated interface for kernel caching and optimization.

## Why Choose FlagGems?

*   **Performance Boost:** Experience significant speedups compared to PyTorch ATen, as demonstrated in our performance benchmarks.
*   **Ease of Use:** Leverage familiar PyTorch APIs while benefiting from cutting-edge hardware acceleration.
*   **Developer-Friendly:** Triton language offers readability and ease of use, reducing the learning curve.
*   **Wide Hardware Support:** Benefit from optimized performance across diverse hardware configurations, including leading AI accelerators.

## What's New in Recent Releases

### v3.0

*   Expanded Operator Support: Now includes a total of 184 operators, encompassing custom operators utilized in large model inference.
*   Enhanced Hardware Compatibility: Added support for additional hardware platforms, such as Ascend and AIPU.
*   VLLM Framework Integration: Compatibility with the vLLM framework, demonstrated through successful inference verification of the DeepSeek model.

### v2.1 & Earlier Releases:

*   Comprehensive Operator Support: Includes a wide range of Tensor, neural network, basic math, and distribution operators.
*   BLAS and Fused Operator Support: Implementation of BLAS, pointwise, reduction, and fused operators.

## Getting Started

Get up and running quickly with FlagGems! Refer to the detailed documentation in [GetStart](docs/get_start_with_flaggems.md) for installation and usage instructions.

## Supported Operators & Example Models

Find a list of supported operators and the expected future implementations in [OperatorList](docs/operator_list.md). Examples include models like Bert-base-uncased, Llama-2-7b, and Llava-1.5-7b.

## Supported Hardware Platforms

FlagGems offers broad hardware support:

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

## Performance Benchmarks

[Image of Operator Speedup](./docs/assets/speedup-20250423.png) shows the performance improvement of FlagGems operators compared to PyTorch ATen library in eager mode.

## Contribute

We encourage contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

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

For questions or inquiries, please submit an issue or reach out to us at <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

FlagGems is licensed under the [Apache 2.0](./LICENSE) license.