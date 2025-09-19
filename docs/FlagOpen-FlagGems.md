# FlagGems: Accelerate LLM Training and Inference with High-Performance Operators

**Supercharge your Large Language Model (LLM) performance with FlagGems, a powerful operator library built on OpenAI Triton for blazing-fast training and inference.**  [View the original repository on GitHub](https://github.com/FlagOpen/FlagGems).

![FlagGems Overview](https://github.com/user-attachments/assets/97950fc6-62bb-4b6a-b8d5-5751c14492fa)

## Key Features

FlagGems offers a comprehensive suite of features designed to optimize LLM performance across diverse hardware platforms:

*   **Extensive Operator Library:** Provides a wide range of PyTorch-compatible operators, ensuring seamless integration with existing LLM workflows.
*   **Optimized Performance:**  Leverages hand-optimized kernels for select operators, delivering significant speedups.
*   **Eager Mode Ready:**  Works independently of `torch.compile`, offering flexibility in your development process.
*   **Automatic Code Generation:**  Supports automatic pointwise operator code generation for arbitrary input types and layouts, simplifying development.
*   **Fast Kernel Dispatching:**  Features rapid per-function runtime kernel dispatching for efficient execution.
*   **Multi-Backend Support:**  Enables support for diverse hardware platforms, expanding compatibility.
*   **C++ Triton Function Dispatcher:**  (Work in progress) Designed to reduce the overhead of the Python runtime and improve overall performance.
*   **LibEntry:** Independent kernel cache management that bypasses unnecessary runtime overhead

## What's New

### Changelog Highlights

*   **v3.0:** Support for 184 operators, including custom operators used in large model inference, and more hardware platforms, including Ascend and AIPU. Compatibility with the vLLM framework, verified with the DeepSeek model.
*   **v2.1:** Support for an array of Tensor, neural network, basic math, and distribution operators.
*   **v2.0:** Support for BLAS, pointwise, reduction, and fused operators.
*   **v1.0:** Initial support for BLAS, pointwise, and reduction operators.

## Supported Hardware Platforms

FlagGems delivers optimized performance across a broad range of hardware:

| Vendor      | State                    | float16 | float32 | bfloat16 |
| ----------- | ------------------------ | ------- | ------- | -------- |
| aipu        | ✅ (Partial support)     | ✅      | ✅      | ✅       |
| ascend      | ✅ (Partial support)     | ✅      | ✅      | ✅       |
| cambricon   | ✅                       | ✅      | ✅      | ✅       |
| hygon       | ✅                       | ✅      | ✅      | ✅       |
| iluvatar    | ✅                       | ✅      | ✅      | ✅       |
| kunlunxin   | ✅                       | ✅      | ✅      | ✅       |
| metax       | ✅                       | ✅      | ✅      | ✅       |
| mthreads    | ✅                       | ✅      | ✅      | ✅       |
| nvidia      | ✅                       | ✅      | ✅      | ✅       |
| arm(cpu)    | 🚧                       |         |         |          |
| tsingmicro  | 🚧                       |         |         |          |

## Performance Boost

[Insert image from ./docs/assets/speedup-20250423.png here.  Caption: FlagGems delivers significant speedups compared to PyTorch ATen library in eager mode.]

## Get Started

For detailed instructions on installation and usage, please refer to the [Get Started Guide](docs/get_start_with_flaggems.md).

## Supported Operators

The library's supported operators are listed in the [Operator List](docs/operator_list.md).

## Example Models

*   Bert-base-uncased
*   Llama-2-7b
*   Llava-1.5-7b

## Contribute

We welcome contributions!  Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

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

## Contact Us

For any questions or inquiries, please submit an issue or contact us at <a href="mailto:flaggems@baai.ac.cn">flaggems@baai.ac.cn</a>.

## License

FlagGems is licensed under the [Apache 2.0](./LICENSE) license.
```
Key improvements and SEO optimizations:

*   **Clear Headline:**  A concise and keyword-rich headline immediately grabs attention.
*   **One-Sentence Hook:** The opening sentence provides a compelling reason to use FlagGems.
*   **Structured Headings:**  Uses clear headings to organize information logically.
*   **Bulleted Key Features:**  Highlights the most important benefits in an easily digestible format.
*   **Keyword Optimization:** Uses relevant keywords like "LLM," "operator library," "Triton," and "performance" throughout.
*   **Call to Action:** Encourages users to "View the original repository on GitHub".
*   **Changelog:** Summarized changelog for easy version comparison.
*   **Image Placeholder:** Added a placeholder for the performance chart.
*   **Clear Structure:**  The content is well-organized and easy to scan.
*   **Contact Information:**  Easy access to how to contact the team with links and email
*   **Citation:** A place to cite the project.
*   **License:** Included the license.