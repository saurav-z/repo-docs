# Transformer Engine: Accelerate Transformer Models with FP8 Precision

**Supercharge your Transformer models with NVIDIA Transformer Engine, achieving faster training and inference through optimized kernels and FP8 support.**  [Explore the original repository](https://github.com/NVIDIA/TransformerEngine).

**Key Features:**

*   **FP8 Precision Support:** Leverage 8-bit floating point (FP8) precision on NVIDIA Hopper, Ada, and Blackwell GPUs for significant performance gains and reduced memory utilization.
*   **Optimized Kernels:** Benefit from highly optimized building blocks and fused kernels tailored for Transformer architectures.
*   **Framework Agnostic C++ API:** Integrate Transformer Engine with other deep learning libraries to enable FP8 support across various frameworks.
*   **Ease of Use:** Utilize user-friendly modules for building Transformer layers, simplifying mixed-precision training.
*   **Broad Compatibility:** Support for FP8, FP16, and BF16 precisions across NVIDIA GPU architectures (Ampere and later).

**Latest News**

*   [03/2025] `Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/>`
*   [03/2025] `Measure and Improve AI Workload Performance with NVIDIA DGX Cloud Benchmarking <https://developer.nvidia.com/blog/measure-and-improve-ai-workload-performance-with-nvidia-dgx-cloud-benchmarking/>`
*   [02/2025] `Understanding the Language of Life's Biomolecules Across Evolution at a New Scale with Evo 2 <https://developer.nvidia.com/blog/understanding-the-language-of-lifes-biomolecules-across-evolution-at-a-new-scale-with-evo-2/>`
*   [02/2025] `NVIDIA DGX Cloud Introduces Ready-To-Use Templates to Benchmark AI Platform Performance <https://developer.nvidia.com/blog/nvidia-dgx-cloud-introduces-ready-to-use-templates-to-benchmark-ai-platform-performance/>`
*   [01/2025] `Continued Pretraining of State-of-the-Art LLMs for Sovereign AI and Regulated Industries with iGenius and NVIDIA DGX Cloud <https://developer.nvidia.com/blog/continued-pretraining-of-state-of-the-art-llms-for-sovereign-ai-and-regulated-industries-with-igenius-and-nvidia-dgx-cloud/>`

**What is Transformer Engine?**

Transformer Engine (TE) is a powerful library designed to accelerate Transformer models on NVIDIA GPUs. It focuses on optimizing both training and inference, particularly through the use of 8-bit floating point (FP8) precision on compatible GPUs (Hopper, Ada, and Blackwell). TE provides a suite of highly optimized components, including specialized building blocks for common Transformer architectures and a streamlined API that easily integrates with your existing framework code. It also offers a framework-agnostic C++ API, allowing for FP8 support to be added to other deep learning libraries. As Transformer models grow in size, TE provides the tools to overcome memory and compute bottlenecks, achieving speedups with minimal impact on accuracy.

**Highlights**

*   Easy-to-use modules for building Transformer layers with FP8 support
*   Optimizations (e.g. fused kernels) for Transformer models
*   Support for FP8 on NVIDIA Hopper, Ada, and Blackwell GPUs
*   Support for optimizations across all precisions (FP16, BF16) on NVIDIA Ampere GPU architecture generations and later

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

For a more comprehensive tutorial, check out our `Quickstart Notebook <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/quickstart.ipynb>`_.

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

    *   **Docker (Recommended):**  Leverage pre-built Docker images on the [NVIDIA GPU Cloud (NGC) Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
        *   Example (PyTorch): `docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3`
        *   Example (JAX): `docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.08-py3`
    *   **pip Installation:**
        *   Prerequisites: A compatible C++ compiler and the CUDA Toolkit with cuDNN and NVCC.
        *   Install the latest stable version:
            *   For PyTorch: `pip install --no-build-isolation transformer_engine[pytorch]`
            *   For JAX: `pip install --no-build-isolation transformer_engine[jax]`
            *   For both: `pip install --no-build-isolation transformer_engine[pytorch,jax]`
        *   Install from GitHub: `pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable`
        *   Specify frameworks (GitHub): `NVTE_FRAMEWORK=pytorch,jax pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable`
    *   **conda Installation:**
        *   For PyTorch: `conda install -c conda-forge transformer-engine-torch`
        *   JAX (coming soon)
    *   **Source Installation:**  See the [installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source)

*   **Environment Variables:**

    *   CUDA_PATH: Path to CUDA installation
    *   CUDNN_PATH: Path to cuDNN installation
    *   CXX: Path to C++ compiler
    *   NVTE_FRAMEWORK: Comma-separated list of frameworks to build for (e.g., ``pytorch,jax``)
    *   MAX_JOBS: Limit number of parallel build jobs (default varies by system)
    *   NVTE_BUILD_THREADS_PER_JOB: Control threads per build job

*   **Compiling with FlashAttention:**

    *   Transformer Engine supports FlashAttention-2 and FlashAttention-3 in PyTorch for increased performance.
    *   FlashAttention-3 is prioritized if both versions are present.
    *   Verify the version using: `NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python your_script.py`
    *   For FlashAttention-2 compilation issues, consider setting `MAX_JOBS=1`.

**Troubleshooting**

*   **Common Issues and Solutions:**

    1.  **ABI Compatibility Issues:**
        *   **Symptoms:** `ImportError` with undefined symbols.
        *   **Solution:** Rebuild PyTorch from source with the same C++ ABI as your system.
    2.  **Missing Headers or Libraries:**
        *   **Symptoms:** CMake errors about missing headers (e.g., `cudnn.h`).
        *   **Solution:** Install missing development packages, and/or set the following environment variables to point to the correct locations, e.g.:
          *   `export CUDA_PATH=/path/to/cuda`
          *   `export CUDNN_PATH=/path/to/cudnn`
    3.  **Build Resource Issues:**
        *   **Symptoms:** Compilation hangs or out-of-memory errors.
        *   **Solution:** Limit parallel builds:  `MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1 pip install ...`
    4.  **Verbose Build Logging:**
        *   For detailed logs:
            ```bash
            cd transformer_engine
            pip install -v -v -v --no-build-isolation .
            ```

**Breaking Changes**

*   **v1.7: Padding Mask Definition (PyTorch):**
    *   The padding mask now uses `True` to indicate *masking* (excluding) a position, aligning with the standard definition across all frameworks.  The previous behavior used `True` for inclusion.

**FP8 Convergence**

*   Extensive testing demonstrates that FP8 training achieves convergence results comparable to BF16, across a variety of model architectures and configurations.
*   Accuracy validation has been performed on downstream LLM tasks (e.g., LAMBADA and WikiText).
*   Examples of models tested for convergence are listed in the original README, for example:
    *   T5-770M (JAX/T5x)
    *   MPT-1.3B (Mosaic Composer)
    *   GPT-5B (JAX/Paxml, NeMo Framework)
    *   LLama2-7B (Alibaba Pai)
    *   T5-11B (JAX/T5x)
    *   MPT-13B (Mosaic Composer)
    *   GPT-22B (NeMo Framework)
    *   LLama2-70B (Alibaba Pai)
    *   GPT-175B (JAX/Paxml)

**Integrations**

Transformer Engine integrates with popular LLM frameworks including: DeepSpeed, Hugging Face Accelerate, Lightning, MosaicML Composer, NVIDIA JAX Toolbox, NVIDIA Megatron-LM, NVIDIA NeMo Framework, Amazon SageMaker Model Parallel Library, Levanter, GPT-NeoX, Hugging Face Nanotron (coming soon), Colossal-AI (coming soon), PeriFlow (coming soon).

**Contributing**

*   We welcome contributions!  Please follow the guidelines outlined in the `<CONTRIBUTING.rst>` guide.

**Papers**

*   [Attention original paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*   [Megatron-LM tensor parallel](https://arxiv.org/pdf/1909.08053.pdf)
*   [Megatron-LM sequence parallel](https://arxiv.org/pdf/2205.05198.pdf)
*   [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

**Videos**

*   [Stable and Scalable FP8 Deep Learning Training on Blackwell | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/>]
*   [Blackwell Numerics for AI | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc25-s72458/>]
*   [Building LLMs: Accelerating Pretraining of Foundational Models With FP8 Precision | GTC 2025 <https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=zoho#/session/1726152813607001vnYK>]
*   [From FP8 LLM Training to Inference: Language AI at Scale | GTC 2025 <https://www.nvidia.com/en-us/on-demand/session/gtc25-s72799/>]
*   [What's New in Transformer Engine and FP8 Training | GTC 2024 <https://www.nvidia.com/en-us/on-demand/session/gtc24-s62457/>]
*   [FP8 Training with Transformer Engine | GTC 2023 <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393>]
*   [FP8 for Deep Learning | GTC 2023 <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/>]
*   [Inside the Hopper Architecture | GTC 2022 <https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/>]

**Previous News**

*   [11/2024] Developing a 172B LLM with Strong Japanese Capabilities Using NVIDIA Megatron-LM
*   [11/2024] How FP8 boosts LLM training by 18% on Amazon SageMaker P5 instances
*   [11/2024] Efficiently train models with large sequence lengths using Amazon SageMaker model parallel
*   [09/2024] Reducing AI large model training costs by 30% requires just a single line of code from FP8 mixed precision training upgrades
*   [05/2024] Accelerating Transformers with NVIDIA cuDNN 9
*   [03/2024] Turbocharged Training: Optimizing the Databricks Mosaic AI stack with FP8
*   [03/2024] FP8 Training Support in SageMaker Model Parallelism Library
*   [12/2023] New NVIDIA NeMo Framework Features and NVIDIA H200
*   [11/2023] Inflection-2: The Next Step Up
*   [11/2023] Unleashing The Power Of Transformers With NVIDIA Transformer Engine
*   [11/2023] Accelerating PyTorch Training Workloads with FP8
*   [09/2023] Transformer Engine added to AWS DL Container for PyTorch Training
*   [06/2023] Breaking MLPerf Training Records with NVIDIA H100 GPUs
*   [04/2023] Benchmarking Large Language Models on NVIDIA H100 GPUs with CoreWeave (Part 1)