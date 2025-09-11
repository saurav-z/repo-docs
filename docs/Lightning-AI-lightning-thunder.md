<!-- Improved README with SEO Optimization -->

# Lightning Thunder: Supercharge Your PyTorch Models ⚡

**Lightning Thunder is a source-to-source compiler that empowers you to optimize your PyTorch models with ease, achieving significant speedups and unlocking the full potential of your hardware.** [Check out the original repo](https://github.com/Lightning-AI/lightning-thunder)

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderLightModewByline.png#gh-light-mode-only" width="400px" style="max-width: 100%;">
<img alt="Thunder" src="docs/source/_static/images/LightningThunderDarkModewByline.png#gh-dark-mode-only" width="400px" style="max-width: 100%;">
</div>

## Key Features

*   **Blazing Fast Performance:** Achieve up to 81% speedups on LLMs and other models.
*   **Out-of-the-Box Optimizations:** Benefit from pre-built plugins for immediate performance gains.
*   **Flexible and Extensible:** Customize your model optimization with composable transformations.
*   **Quantization Support:** Utilize FP4/FP6/FP8 precision for efficient model execution.
*   **Distributed Training:** Scale your models using TP/PP/DP strategies.
*   **CUDA Graph Integration:** Reduce CPU overhead with CUDA Graphs.
*   **Optimized for Latest Hardware:** Ready for NVIDIA Blackwell and other modern architectures.
*   **Easy-to-Use:** Simple `thunder.compile()` API for quick integration.
*   **Comprehensive Examples:** Get started quickly with pre-built examples for LLMs, vision transformers, and more.

<div align="center">
<pre>
✅ Run PyTorch up to 81% faster  ✅ FP4/FP6/FP8 precision    ✅ Kernel fusion
✅ Training recipes            ✅ Ready for NVIDIA Blackwell ✅ CUDA Graphs
✅ Inference recipes           ✅ Custom Triton kernels     ✅ Compose all the above
</pre>
</div>

## Quick Start

**Installation:**

```bash
pip install torch==2.6.0 torchvision==0.21 nvfuser-cu124-torch26
pip install lightning-thunder
```

**Basic Usage:**

1.  **Define your model:**

    ```python
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(2048, 4096), nn.ReLU(), nn.Linear(4096, 64))
    ```

2.  **Compile with Thunder:**

    ```python
    import thunder
    import torch
    thunder_model = thunder.compile(model)
    x = torch.randn(64, 2048)
    y = thunder_model(x)
    torch.testing.assert_close(y, model(x))
    ```

## Examples

*   **Speed up LLM Training:** Train models like Llama 3 faster by using `LitGPT` and Thunder.
*   **Accelerate Hugging Face Inference:** Boost inference speed for BERT, DeepSeek R1, and other Hugging Face models.
*   **Optimize Vision Transformers:** Enhance the performance of models like ViT.

**See the examples section for more in-depth examples with code snippets.**

## Performance

Thunder delivers significant speedups.  For instance, on a pre-training task using LitGPT, Thunder achieves remarkable improvements on H100 and B200 hardware relative to PyTorch eager execution.

<div align="center">
<img alt="Thunder" src="docs/source/_static/images/pretrain_perf.png" width="800px" style="max-width: 100%;">
</div>

## Additional Resources

*   [Installation Guide](https://lightning.ai/docs/thunder/latest/fundamentals/installation.html)
*   [Documentation](https://lightning-thunder.readthedocs.io/en/latest/)
*   [Discord Community](https://discord.com/invite/XncpTy7DSt)

**Ready to experience the power of Lightning Thunder?**