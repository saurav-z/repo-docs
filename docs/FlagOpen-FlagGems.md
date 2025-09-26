# FlagGems: Accelerate LLM Training and Inference with High-Performance Operators (Optimized for SEO)

**Supercharge your Large Language Model (LLM) performance with FlagGems, a cutting-edge operator library built with OpenAI Triton for unparalleled speed and efficiency.** ([See the original repository](https://github.com/FlagOpen/FlagGems))

FlagGems delivers significant performance gains by leveraging the power of the Triton language to optimize LLM training and inference across diverse hardware platforms, all while maintaining seamless compatibility with your existing PyTorch workflows.

## Key Features of FlagGems

*   **PyTorch Compatibility:** Easily integrate FlagGems with your PyTorch models without modifying low-level APIs.
*   **Optimized Operators:** Benefit from hand-optimized performance for a wide range of LLM operators.
*   **Eager Mode Ready:** Works directly in eager mode, independent of `torch.compile`.
*   **Automatic Codegen:** Generate pointwise and fused operators automatically, supporting various input types and layouts.
*   **Fast Kernel Dispatching:** Achieve high performance through fast per-function runtime kernel dispatching.
*   **Multi-Backend Support:** Runs on numerous hardware platforms, offering broad hardware compatibility.
*   **Extensive Backend Support:** Currently supports over 10 backends.
*   **C++ Triton Function Dispatcher:** (In progress) Further enhances performance with a C++ dispatcher.
*   **LibEntry:** Independently manages kernel cache and bypasses runtime overhead.

## More About FlagGems Features

### Multi-Backend Hardware Support

FlagGems offers broad hardware compatibility with extensive testing across various configurations.

### Automatic Codegen

Simplify operator generation with FlagGems' automatic code generation mechanism. Create both pointwise and fused operators effortlessly, with support for element-wise computations, non-tensor parameters, and output type specifications.  See more details in the [pointwise_dynamic](docs/pointwise_dynamic.md).

### LibEntry

`LibEntry` enhances performance by independently managing the kernel cache and bypassing the runtime of `Autotuner`, `Heuristics`, and `JitFunction`.  Decorate your Triton kernel with `LibEntry` for simplified usage.

`LibEntry` also provides direct wrapping of `Autotuner`, `Heuristics`, and `JitFunction` while avoiding nested runtime type invocations. This leads to reduced key computation and simplified cache key formats.

### C++ Runtime

FlagGems is available as a pure Python package or with C++ extensions. The C++ runtime optimizes performance, reducing the overhead associated with the Python runtime.

## Changelog Highlights

### v3.0

*   Support for 184 operators, including custom operators for large model inference.
*   Expanded hardware platform support, including Ascend and AIPU.
*   Compatibility with the vLLM framework, verified with the DeepSeek model.

### v2.1

*   Support for various Tensor, neural network, basic math, and distribution operators.

### v2.0

*   Support for BLAS, pointwise, reduction, and fused operators.

### v1.0

*   Initial support for BLAS, pointwise, and reduction operators.

## Get Started

Explore the documentation [GetStart](docs/get_start_with_flaggems.md) for installation and usage instructions.

## Supported Operators

The [OperatorList](docs/operator_list.md) details the operators that are and will be implemented.

## Example Models

*   Bert-base-uncased
*   Llama-2-7b
*   Llava-1.5-7b

## Supported Platforms

| Vendor       | State                  | float16 | float32 | bfloat16 |
| ------------ | ---------------------- | ------- | ------- | -------- |
| aipu         | ✅ （Partial support） | ✅      | ✅      | ✅       |
| ascend       | ✅ （Partial support） | ✅      | ✅      | ✅       |
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

FlagGems consistently outperforms PyTorch ATen in eager mode. The chart below illustrates the average speedup across various shapes:

![Operator Speedup](./docs/assets/speedup-20250423.png)

## Contributions

If you're interested in contributing, review [CONTRIBUTING.md](./CONTRIBUTING.md). Your contributions are welcome.

## Citation

If you find FlagGems useful, please cite our project:

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

For questions or support, submit an issue or contact us at <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

FlagGems is licensed under [Apache 2.0](./LICENSE).