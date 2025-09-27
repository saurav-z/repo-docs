# FlagGems: Accelerate LLM Training and Inference with High-Performance Operators

**FlagGems leverages the power of OpenAI Triton to deliver lightning-fast, backend-agnostic operators for LLMs, boosting performance across diverse hardware platforms.**  [See the original repository](https://github.com/FlagOpen/FlagGems).

## Key Features

*   **Extensive PyTorch Compatibility:** Seamless integration with existing PyTorch workflows.
*   **Optimized Performance:** Hand-optimized operators for superior speed.
*   **Eager Mode Ready:** Works directly without needing `torch.compile`.
*   **Automatic Codegen:** Generates operators for various data types and layouts.
*   **Fast Kernel Dispatching:** Efficient runtime for quick operator execution.
*   **Multi-Backend Support:** Supports a wide range of hardware for broad compatibility.
*   **C++ Runtime (In Progress):**  Enhances performance by reducing Python overhead.

## Feature Details

### Multi-Backend Hardware Support

FlagGems is designed to be versatile, supporting a wide array of hardware platforms. Extensive testing ensures reliable performance across different configurations.

### Automatic Codegen

Developers can easily generate pointwise and fused operators using FlagGems' automatic code generation. The system supports standard element-wise computations, non-tensor parameters, and output type specifications.  Learn more at [pointwise\_dynamic](docs/pointwise_dynamic.md).

### LibEntry

`LibEntry` enhances performance by independently managing the kernel cache, bypassing runtime overhead for `Autotuner`, `Heuristics`, and `JitFunction`. It also supports direct wrapping, preserving full tuning functionality while simplifying the cache key and reducing unnecessary computations.

### C++ Runtime

FlagGems offers a C++ runtime option, which optimizes performance by addressing the overhead of the Python runtime, resulting in faster end-to-end execution.

## Changelog

### v3.0

*   Expanded operator support to 184 operators, including custom operators for large model inference.
*   Extended hardware platform support, including Ascend, AIPU, and more.
*   Compatibility with the vLLM framework, with successful inference verification of the DeepSeek model.

### v2.1

*   Supported new Tensor, neural network, basic math, and distribution operators.

### v2.0

*   Supported BLAS, pointwise, reduction, and fused operators.

### v1.0

*   Initial support for BLAS, pointwise, and reduction operators.

## Get Started

For a quick start with installing and using FlagGems, please refer to the documentation [GetStart](docs/get_start_with_flaggems.md).

## Supported Operators

The development team will implement operators according to [OperatorList](docs/operator_list.md).

## Example Models

*   Bert-base-uncased
*   Llama-2-7b
*   Llava-1.5-7b

## Supported Platforms

| Vendor      | State                  | float16 | float32 | bfloat16 |
| ----------- | ---------------------- | ------- | ------- | -------- |
| aipu        | ✅ (Partial support)   | ✅      | ✅      | ✅       |
| ascend      | ✅ (Partial support)   | ✅      | ✅      | ✅       |
| cambricon   | ✅                     | ✅      | ✅      | ✅       |
| hygon       | ✅                     | ✅      | ✅      | ✅       |
| iluvatar    | ✅                     | ✅      | ✅      | ✅       |
| kunlunxin   | ✅                     | ✅      | ✅      | ✅       |
| metax       | ✅                     | ✅      | ✅      | ✅       |
| mthreads    | ✅                     | ✅      | ✅      | ✅       |
| nvidia      | ✅                     | ✅      | ✅      | ✅       |
| arm(cpu)    | 🚧                     |         |         |          |
| tsingmicro  | 🚧                     |         |         |          |

## Performance

![Operator Speedup](./docs/assets/speedup-20250423.png)

The chart showcases FlagGems' speedup compared to the PyTorch ATen library in eager mode. The speedup, calculated by averaging across various shapes, represents the overall operator performance.

## Contributions

We welcome contributions! Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Citation

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

FlagGems is licensed under [Apache 2.0](./LICENSE).