# FlagGems: Accelerate LLM Training and Inference with High-Performance Operators

**FlagGems significantly boosts the performance of Large Language Models (LLMs) by leveraging the power of OpenAI Triton for optimized operator kernels.** [Explore the FlagGems repository](https://github.com/FlagOpen/FlagGems).

## Key Features of FlagGems

*   **Extensive Operator Library:** Provides a comprehensive set of PyTorch-compatible operators, essential for LLM tasks.
*   **Optimized Performance:** Features hand-optimized kernels for select operators, ensuring peak performance.
*   **Eager Mode Ready:** Works seamlessly in eager mode, independent of `torch.compile` for flexibility.
*   **Automatic Code Generation:** Includes an automatic pointwise operator codegen that supports arbitrary input types and layouts.
*   **Fast Kernel Dispatching:** Offers fast per-function runtime kernel dispatching for efficient execution.
*   **Multi-Backend Support:** Designed with a multi-backend interface, supporting diverse hardware platforms.
*   **C++ Triton Dispatcher:** Includes a C++ Triton function dispatcher to reduce Python overhead and increase end-to-end performance (in progress).

## What's New in FlagGems

*   **v3.0:** Supports 184 operators, including custom operators, and adds support for Ascend and AIPU hardware. Compatible with the vLLM framework.
*   **v2.1:** Expanded operator support, including tensor, neural network, basic math, and distribution operators.
*   **v2.0:** Added BLAS, pointwise, reduction, and fused operators, boosting LLM performance.
*   **v1.0:** Implemented essential BLAS, pointwise, and reduction operators for foundational LLM tasks.

## Benefits of Using FlagGems

*   **Seamless Integration:** Integrates with PyTorch, allowing developers to use familiar APIs while benefiting from hardware acceleration.
*   **Enhanced Performance:** Offers performance comparable to CUDA, leading to faster training and inference times.
*   **Simplified Development:** Triton language provides readability and ease of use, reducing the learning curve for developers.

## Hardware Support

FlagGems supports a wide range of hardware platforms, providing optimized performance across diverse environments:

| Vendor      | State                  | float16 | float32 | bfloat16 |
| ----------- | ---------------------- | ------- | ------- | -------- |
| aipu        | ✅ （Partial support） | ✅      | ✅      | ✅       |
| ascend      | ✅ （Partial support） | ✅      | ✅      | ✅       |
| cambricon   | ✅                     | ✅      | ✅      | ✅       |
| hygon       | ✅                     | ✅      | ✅      | ✅       |
| iluvatar    | ✅                     | ✅      | ✅      | ✅       |
| kunlunxin   | ✅                     | ✅      | ✅      | ✅       |
| metax       | ✅                     | ✅      | ✅      | ✅       |
| mthreads    | ✅                     | ✅      | ✅      | ✅       |
| nvidia      | ✅                     | ✅      | ✅      | ✅       |
| arm(cpu)    | 🚧                     |         |         |          |
| tsingmicro  | 🚧                     |         |         |          |

## Get Started

For installation and usage instructions, refer to the [Get Started Guide](docs/get_start_with_flaggems.md).

## Supported Operators

Refer to [Operator List](docs/operator_list.md) for the complete list of operators.

## Example Models

*   Bert-base-uncased
*   Llama-2-7b
*   Llava-1.5-7b

## Performance

(Include the Performance chart from the original README here, ideally with alt text)

## Contribute

Contributions to FlagGems are highly encouraged. Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Citation

If you use FlagGems, please cite our project:

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

For questions, submit an issue or contact us at <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

FlagGems is licensed under [Apache 2.0](./LICENSE).