# Transformer Engine: Accelerate Transformer Models on NVIDIA GPUs

**Supercharge your Transformer model performance with NVIDIA Transformer Engine, enabling faster training and inference with FP8 precision.**

[Get Started](https://github.com/NVIDIA/TransformerEngine)

Transformer Engine (TE) is a powerful library designed to accelerate Transformer models on NVIDIA GPUs. It provides highly optimized building blocks and an easy-to-use API for integrating FP8 precision, resulting in significant performance gains and reduced memory usage.  It's compatible with Hopper, Ada, and Blackwell GPUs.

**Key Features:**

*   **FP8 Support:**  Enables FP8 (8-bit floating point) precision for Hopper, Ada, and Blackwell GPUs.
*   **Optimized Modules:** Provides easy-to-use modules for building Transformer layers with optimized kernels, including fused kernels.
*   **Mixed Precision API:**  Offers an automatic mixed-precision-like API for seamless integration with framework-specific code.
*   **Framework Agnostic C++ API:**  Includes a C++ API for integration with other deep learning libraries, enabling FP8 support.
*   **Performance Boost:**  Achieve faster training and inference speeds with reduced memory utilization.
*   **Compatibility:** Supports FP16, BF16, and FP8 across various NVIDIA GPU architectures.

**Latest News**

*   [03/2025] `Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/>`_
*   [03/2025] `Measure and Improve AI Workload Performance with NVIDIA DGX Cloud Benchmarking <https://developer.nvidia.com/blog/measure-and-improve-ai-workload-performance-with-nvidia-dgx-cloud-benchmarking/>`_
    ![Comparison of FP8 versus BF16 training](docs/examples/comparison-fp8-bf16-training-nvidia-dgx-cloud-benchmarking-performance-explorer.jpg)
*   [02/2025] `Understanding the Language of Life's Biomolecules Across Evolution at a New Scale with Evo 2 <https://developer.nvidia.com/blog/understanding-the-language-of-lifes-biomolecules-across-evolution-at-a-new-scale-with-evo-2/>`_
*   [02/2025] `NVIDIA DGX Cloud Introduces Ready-To-Use Templates to Benchmark AI Platform Performance <https://developer.nvidia.com/blog/nvidia-dgx-cloud-introduces-ready-to-use-templates-to-benchmark-ai-platform-performance/>`_
*   [01/2025] `Continued Pretraining of State-of-the-Art LLMs for Sovereign AI and Regulated Industries with iGenius and NVIDIA DGX Cloud <https://developer.nvidia.com/blog/continued-pretraining-of-state-of-the-art-llms-for-sovereign-ai-and-regulated-industries-with-igenius-and-nvidia-dgx-cloud/>`_

**Examples**

*   **PyTorch:**
    ```python
    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    # Set dimensions.
    in_features = 768
    out_features = 3072
    hidden_size = 2048

    # Initialize model and inputs.
    model = te.Linear(in_features, out_features, bias=True)
    inp = torch.randn(hidden_size, in_features, device="cuda")

    # Create an FP8 recipe. Note: All input args are optional.
    fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

    # Enable autocasting for the forward pass
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model(inp)

    loss = out.sum()
    loss.backward()
    ```
*   **JAX/Flax:**
    ```python
    import flax
    import jax
    import jax.numpy as jnp
    import transformer_engine.jax as te
    import transformer_engine.jax.flax as te_flax
    from transformer_engine.common import recipe

    BATCH = 32
    SEQLEN = 128
    HIDDEN = 1024

    # Initialize RNG and inputs.
    rng = jax.random.PRNGKey(0)
    init_rng, data_rng = jax.random.split(rng)
    inp = jax.random.normal(data_rng, [BATCH, SEQLEN, HIDDEN], jnp.float32)

    # Create an FP8 recipe. Note: All input args are optional.
    fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.HYBRID)

    # Enable autocasting for the forward pass
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        model = te_flax.DenseGeneral(features=HIDDEN)

        def loss_fn(params, other_vars, inp):
            out = model.apply({'params':params, **other_vars}, inp)
            return jnp.mean(out)

        # Initialize models.
        variables = model.init(init_rng, inp)
        other_variables, params = flax.core.pop(variables, 'params')

        # Construct the forward and backward function
        fwd_bwd_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

        for _ in range(10):
            loss, (param_grads, other_grads) = fwd_bwd_fn(params, other_variables, inp)
    ```

    For a more comprehensive tutorial, check out our `Quickstart Notebook <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb>`_.

**Installation**

*   **System Requirements:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, and Ampere GPUs with Linux OS, CUDA 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell), cuDNN 9.3+, GCC 9+ or Clang 10+, Python 3.12 recommended, and compatible drivers.
*   **Methods:**
    *   **Docker (Recommended):** Utilize NGC containers for pre-installed dependencies: `nvcr.io/nvidia/pytorch:25.08-py3`, `nvcr.io/nvidia/jax:25.08-py3`.
    *   **pip:**
        ```bash
        # For PyTorch integration
        pip install --no-build-isolation transformer_engine[pytorch]
        # For JAX integration
        pip install --no-build-isolation transformer_engine[jax]
        # For both frameworks
        pip install --no-build-isolation transformer_engine[pytorch,jax]
        ```
    *   **conda:**
        ```bash
        conda install -c conda-forge transformer-engine-torch #PyTorch
        ```
        (JAX installation coming soon)
    *   **Source:** Follow the detailed instructions in the [User Guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source).
*   **Environment Variables:**  Customize the build process using environment variables such as `CUDA_PATH`, `CUDNN_PATH`, `CXX`, `NVTE_FRAMEWORK`, `MAX_JOBS`, and `NVTE_BUILD_THREADS_PER_JOB`.

**Troubleshooting**

*   **ABI Compatibility:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI setting. Rebuild PyTorch from source with matching ABI.
*   **Missing Headers/Libraries:** Install development packages or set environment variables (e.g., `CUDA_PATH`, `CUDNN_PATH`).
*   **Build Resource Issues:** Limit parallel builds using `MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1`.
*   **Verbose Build Logging:** Use `pip install -v -v -v --no-build-isolation .` for detailed logs.

**Breaking Changes**
* **v1.7**: Padding mask definition for PyTorch was changed

**FP8 Convergence**

Extensive testing has shown **no significant difference** in convergence between FP8 and BF16 training.

**Integrations**

Transformer Engine is integrated with the following popular LLM frameworks:

*   DeepSpeed
*   Hugging Face Accelerate
*   Lightning
*   MosaicML Composer
*   NVIDIA JAX Toolbox
*   NVIDIA Megatron-LM
*   NVIDIA NeMo Framework
*   Amazon SageMaker Model Parallel Library
*   Levanter
*   GPT-NeoX
*   Hugging Face Nanotron (Coming soon!)
*   Colossal-AI (Coming soon!)
*   PeriFlow (Coming soon!)

**Contribute**

Contribute to Transformer Engine by following the guidelines in the `<CONTRIBUTING.rst>`_ guide.

**Papers & Videos**
Includes links to relevant research papers and video resources.

**[View the original repo on GitHub](https://github.com/NVIDIA/TransformerEngine)**