# FlagGems: Accelerate LLM Training and Inference with High-Performance Operators (Powered by Triton)

**FlagGems dramatically boosts the performance of large language models by leveraging the power of OpenAI Triton for optimized operator implementations.** ([Original Repo](https://github.com/FlagOpen/FlagGems))

[![FlagGems](https://github.com/user-attachments/assets/97950fc6-62bb-4b6a-b8d5-5751c14492fa)](https://github.com/FlagOpen/FlagGems)

## Key Features

*   **PyTorch Compatibility:** Seamlessly integrates with PyTorch, allowing developers to use familiar APIs.
*   **Optimized Performance:** Hand-tuned kernels for select operators, delivering significant speedups.
*   **Eager Mode Ready:** Works independently of `torch.compile`, providing flexibility.
*   **Automatic Codegen:** Generates pointwise and fused operators automatically for efficiency and ease of development.
*   **Multi-Backend Support:** Supports a wide range of hardware platforms, including NVIDIA, AMD, and more, enabling acceleration across diverse environments.
*   **C++ Runtime (In Development):** Aims to further improve end-to-end performance by reducing Python overhead.
*   **Large Operator Library:** Provides a comprehensive collection of PyTorch-compatible operators.

## Why Choose FlagGems?

FlagGems is designed to accelerate LLM training and inference by providing optimized operator implementations leveraging the power of Triton. Its compatibility with PyTorch, automatic code generation, and multi-backend support, make it an ideal solution for developers looking to maximize performance across various hardware.

## Key Technologies

*   **OpenAI Triton:**  FlagGems utilizes the Triton language to create highly optimized kernels.
*   **LibEntry:** An independent system that manages kernel cache and bypasses the runtime of Autotuner, Heuristics, and JitFunction.
*   **Automatic Code Generation:**  Simplifies the creation of pointwise and fused operators.

## Supported Platforms

FlagGems provides support for various hardware platforms:

| Vendor      | Status                  | float16 | float32 | bfloat16 |
| ----------- | ----------------------- | ------- | ------- | -------- |
| aipu        | ✅ (Partial support)    | ✅      | ✅      | ✅       |
| ascend      | ✅ (Partial support)    | ✅      | ✅      | ✅       |
| cambricon   | ✅                      | ✅      | ✅      | ✅       |
| hygon       | ✅                      | ✅      | ✅      | ✅       |
| iluvatar    | ✅                      | ✅      | ✅      | ✅       |
| kunlunxin   | ✅                      | ✅      | ✅      | ✅       |
| metax       | ✅                      | ✅      | ✅      | ✅       |
| mthreads    | ✅                      | ✅      | ✅      | ✅       |
| nvidia      | ✅                      | ✅      | ✅      | ✅       |
| arm(cpu)    | 🚧                      |         |         |          |
| tsingmicro  | 🚧                      |         |         |          |

## Performance

The following chart shows the speedup of FlagGems compared with PyTorch ATen library in eager mode. The speedup is calculated by averaging the speedup on each shape, representing the overall performance of the operator.

![Operator Speedup](./docs/assets/speedup-20250423.png)

## Get Started

For detailed installation and usage instructions, please refer to the [Get Start Documentation](docs/get_start_with_flaggems.md).

## Contributing

We welcome contributions!  Please see our [CONTRIBUTING.md](./CONTRIBUTING.md) file for details on how to get involved.

## Citation

If you use FlagGems in your research, please cite us:

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

For any questions or suggestions, please submit an issue, or contact us at <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

FlagGems is released under the [Apache 2.0](./LICENSE) license.