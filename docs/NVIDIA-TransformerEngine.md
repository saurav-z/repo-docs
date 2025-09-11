# Transformer Engine: Accelerate Your Transformer Models with FP8 on NVIDIA GPUs

Transformer Engine (TE) is a powerful library designed to supercharge the performance of your Transformer models on NVIDIA GPUs, offering significant speedups and reduced memory usage, especially with the introduction of 8-bit floating point (FP8) precision. Explore the [Transformer Engine repository](https://github.com/NVIDIA/TransformerEngine) for cutting-edge advancements in deep learning acceleration.

**Key Features:**

*   **FP8 Support:** Leverage the efficiency of FP8 precision on Blackwell, Hopper, Ada, and Ampere GPUs for faster training and inference.
*   **Optimized Modules:** Utilize easy-to-use building blocks specifically designed for Transformer architectures.
*   **Mixed Precision API:** Seamlessly integrate with your existing framework code, enabling mixed-precision training with ease.
*   **Fused Kernels:** Benefit from optimized, fused kernels for enhanced performance.
*   **Framework Agnostic C++ API:** Integrate TE with other deep learning libraries for FP8 support.

**Latest News:**

*   **[03/2025]** [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/)
*   **[03/2025]** [Measure and Improve AI Workload Performance with NVIDIA DGX Cloud Benchmarking](https://developer.nvidia.com/blog/measure-and-improve-ai-workload-performance-with-nvidia-dgx-cloud-benchmarking/)
*   **[02/2025]** [Understanding the Language of Life's Biomolecules Across Evolution at a New Scale with Evo 2](https://developer.nvidia.com/blog/understanding-the-language-of-lifes-biomolecules-across-evolution-at-a-new-scale-with-evo-2/)
*   **[02/2025]** [NVIDIA DGX Cloud Introduces Ready-To-Use Templates to Benchmark AI Platform Performance](https://developer.nvidia.com/blog/nvidia-dgx-cloud-introduces-ready-to-use-templates-to-benchmark-ai-platform-performance/)
*   **[01/2025]** [Continued Pretraining of State-of-the-Art LLMs for Sovereign AI and Regulated Industries with iGenius and NVIDIA DGX Cloud](https://developer.nvidia.com/blog/continued-pretraining-of-state-of-the-art-llms-for-sovereign-ai-and-regulated-industries-with-igenius-and-nvidia-dgx-cloud/)

  Compare FP8 versus BF16 training with the [NVIDIA DGX Cloud Benchmarking Performance Explorer](https://developer.nvidia.com/blog/measure-and-improve-ai-workload-performance-with-nvidia-dgx-cloud-benchmarking/).

**[Previous News](#previous-news)**

**What is Transformer Engine?**

Transformer Engine (TE) is a library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper, Ada, and Blackwell GPUs, to provide better performance with lower memory utilization in both training and inference. TE provides a collection of highly optimized building blocks for popular Transformer architectures and an automatic mixed precision-like API that can be used seamlessly with your framework-specific code. TE also includes a framework agnostic C++ API that can be integrated with other deep learning libraries to enable FP8 support for Transformers.

As the number of parameters in Transformer models continues to grow, training and inference for architectures such as BERT, GPT and T5 become very memory and compute-intensive. Most deep learning frameworks train with FP32 by default. This is not essential, however, to achieve full accuracy for many deep learning models. Using mixed-precision training, which combines single-precision (FP32) with lower precision (e.g. FP16) format when training a model, results in significant speedups with minimal differences in accuracy as compared to FP32 training. With Hopper GPU architecture FP8 precision was introduced, which offers improved performance over FP16 with no degradation in accuracy. Although all major deep learning frameworks support FP16, FP8 support is not available natively in frameworks today.

TE addresses the problem of FP8 support by providing APIs that integrate with popular Large Language Model (LLM) libraries. It provides a Python API consisting of modules to easily build a Transformer layer as well as a framework-agnostic library in C++ including structs and kernels needed for FP8 support. Modules provided by TE internally maintain scaling factors and other values needed for FP8 training, greatly simplifying mixed precision training for users.

**Examples**

*   **PyTorch**

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

*   **JAX**

    *   **Flax**

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

    For a more comprehensive tutorial, check out our [Quickstart Notebook](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb).

**Installation**

*   **System Requirements:**
    *   **Hardware:** Blackwell, Hopper, Grace Hopper/Blackwell, Ada, Ampere
    *   **OS:** Linux (official), WSL2 (limited support)
    *   **Software:**
        *   CUDA: 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell) with compatible NVIDIA drivers
        *   cuDNN: 9.3+
        *   Compiler: GCC 9+ or Clang 10+ with C++17 support
        *   Python: 3.12 recommended
    *   **Source Build Requirements:** CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+
    *   **Notes:** FP8 features require Compute Capability 8.9+ (Ada/Hopper/Blackwell)

*   **Installation Methods:**

    *   **Docker (Recommended):**

        Use pre-built Docker images from the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
        *   **PyTorch:**
            ```bash
            docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3
            ```
        *   **JAX:**
            ```bash
            docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.08-py3
            ```
        Where 25.08 (corresponding to August 2025 release) is the container version.

        **Benefits of using NGC containers:**

        *   All dependencies pre-installed with compatible versions and optimized configurations
        *   NGC PyTorch 23.08+ containers include FlashAttention-2
    *   **pip Installation:**

        *   **Prerequisites:** A compatible C++ compiler and CUDA Toolkit with cuDNN and NVCC.
        *   **PyTorch:**
            ```bash
            pip install --no-build-isolation transformer_engine[pytorch]
            ```
        *   **JAX:**
            ```bash
            pip install --no-build-isolation transformer_engine[jax]
            ```
        *   **Both Frameworks:**
            ```bash
            pip install --no-build-isolation transformer_engine[pytorch,jax]
            ```
            or install directly from the GitHub repository:
        ```bash
        pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
        ```
        You can specify frameworks using the environment variable:
        ```bash
        NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable
        ```
    *   **conda Installation:**

        *   **PyTorch:**
            ```bash
            conda install -c conda-forge transformer-engine-torch
            ```
            *   **JAX:** (coming soon)
    *   **Source Installation:**
        [See the installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source)

*   **Environment Variables:**

    Customize the build process with these environment variables:

    *   `CUDA_PATH`: Path to CUDA installation
    *   `CUDNN_PATH`: Path to cuDNN installation
    *   `CXX`: Path to C++ compiler
    *   `NVTE_FRAMEWORK`: Comma-separated list of frameworks (e.g., `pytorch,jax`)
    *   `MAX_JOBS`: Limit parallel build jobs (default varies)
    *   `NVTE_BUILD_THREADS_PER_JOB`: Control threads per build job

*   **Compiling with FlashAttention**

    Transformer Engine supports FlashAttention-2 and FlashAttention-3 in PyTorch. FlashAttention-3 was added in release v1.11 and is prioritized over FlashAttention-2 when both are present in the environment.

    Verify FlashAttention version:

    ```bash
    NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py
    ```

    FlashAttention-2 compilation may require a large amount of RAM. Setting `MAX_JOBS=1` can help.

**Troubleshooting**

1.  **ABI Compatibility Issues:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI. Rebuild PyTorch from source if necessary.
2.  **Missing Headers or Libraries:** Install development packages or set `CUDA_PATH` and `CUDNN_PATH`. Set the `CXX` environment variable if the compiler can't be found.
3.  **Build Resource Issues:** Limit parallel builds with `MAX_JOBS=1` and `NVTE_BUILD_THREADS_PER_JOB=1`.
4.  **Verbose Build Logging:** For detailed build logs:
    ```bash
    cd transformer_engine
    pip install -v -v -v --no-build-isolation .
    ```

**Breaking Changes**

*   **v1.7: Padding mask definition for PyTorch:**

    The padding mask definition in PyTorch has changed.  `True` now indicates masking (exclusion) of the corresponding position.

    ```
    # for a batch of 3 sequences where `a`s, `b`s and `c`s are the useful tokens
    # and `0`s are the padding tokens,
    [a, a, a, 0, 0,
     b, b, 0, 0, 0,
     c, c, c, c, 0]
    # the padding mask for this batch before v1.7 is,
    [ True,  True,  True, False, False,
      True,  True, False, False, False,
      True,  True,  True,  True, False]
    # and for v1.7 onwards it should be,
    [False, False, False,  True,  True,
     False, False,  True,  True,  True,
     False, False, False, False,  True]
    ```

**FP8 Convergence**

FP8 has been extensively tested and demonstrates **no significant difference** in training loss curves compared to BF16. Accuracy has also been validated on downstream LLM tasks.

**Integrations**

Transformer Engine seamlessly integrates with:

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

**Contributing**

We welcome contributions! Please follow the guidelines in the `<CONTRIBUTING.rst>` guide.

**Papers**

*   [Attention original paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*   [Megatron-LM tensor parallel](https://arxiv.org/pdf/1909.08053.pdf)
*   [Megatron-LM sequence parallel](https://arxiv.org/pdf/2205.05198.pdf)
*   [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

**Videos**

*   [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/)
*   [Blackwell Numerics for AI | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72458/)
*   [Building LLMs: Accelerating Pretraining of Foundational Models With FP8 Precision | GTC 2025](https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=zoho#/session/1726152813607001vnYK)
*   [From FP8 LLM Training to Inference: Language AI at Scale | GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72799/)
*   [What's New in Transformer Engine and FP8 Training | GTC 2024](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/)
*   [FP8 Training with Transformer Engine | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393)
*   [FP8 for Deep Learning | GTC 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/)
*   [Inside the Hopper Architecture | GTC 2022](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/)

**[Previous News](#previous-news)**