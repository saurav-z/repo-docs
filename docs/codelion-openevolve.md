# OpenEvolve: The Open-Source Evolutionary Coding Agent

**Unlock the Power of AI for Code Optimization!** OpenEvolve harnesses the capabilities of Large Language Models (LLMs) to automatically discover and optimize code, leading to groundbreaking algorithmic advancements. Explore the possibilities with OpenEvolve: [https://github.com/codelion/openevolve](https://github.com/codelion/openevolve)

<p align="center">
  <a href="https://github.com/codelion/openevolve/stargazers"><img src="https://img.shields.io/github/stars/codelion/openevolve?style=social" alt="GitHub stars"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/v/openevolve" alt="PyPI version"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/dm/openevolve" alt="PyPI downloads"></a>
  <a href="https://github.com/codelion/openevolve/blob/main/LICENSE"><img src="https://img.shields.io/github/license/codelion/openevolve" alt="License"></a>
</p>

[🚀 **Quick Start**](#-quick-start) | [📖 **Examples Gallery**](#-examples-gallery) | [🤔 **FAQ**](#-faq)

---

## ✨ Key Features of OpenEvolve

*   **Autonomous Code Discovery:** OpenEvolve empowers LLMs to discover novel algorithms without human intervention, going beyond simple optimization.
*   **Proven Performance:** Experience significant speedups (2-3x) on real hardware, achieve state-of-the-art results in challenging domains like circle packing, and unlock breakthrough optimizations.
*   **Research-Grade Reproducibility:**  Benefit from a robust system with comprehensive seeding, deterministic evolution, and extensive evaluation pipelines.
*   **Multi-Objective Optimization:** OpenEvolve automatically identifies Pareto-optimal solutions for complex, multi-faceted problems.
*   **Parallel Evolution:** Leverage island-based architectures for faster convergence and scalability across multiple machines.

## 🏆 Achievements - What OpenEvolve Can Do

| **Domain**          | **Achievement**                                    | **Example**                                                                  |
| ------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| GPU Optimization    | 2-3x speedup on Apple Silicon                      | [MLX Metal Kernels](examples/mlx_metal_kernel_opt/)                           |
| Mathematical        | State-of-the-art circle packing (n=26)             | [Circle Packing](examples/circle_packing/)                                      |
| Algorithm Design    | Adaptive sorting algorithms                        | [Rust Adaptive Sort](examples/rust_adaptive_sort/)                             |
| Scientific Computing | Automated filter design                            | [Signal Processing](examples/signal_processing/)                               |
| Multi-Language      | Python, Rust, R, Metal shaders (and more!)         | [All Examples](examples/)                                                      |

## 💡 OpenEvolve vs. Manual Optimization

| Aspect             | Manual Optimization                  | OpenEvolve                             |
| ------------------ | ------------------------------------ | -------------------------------------- |
| Time to Solution   | Days to Weeks                        | Hours                                  |
| Exploration Breadth | Limited by Human Creativity          | Unlimited LLM Creativity                 |
| Reproducibility    | Difficult to Replicate               | Fully Deterministic                    |
| Multi-objective    | Complex Tradeoffs                  | Automatic Pareto Optimization          |
| Scaling            | Doesn't Scale                       | Parallel Evolution Across Island         |

## 🚀 Quick Start: Get Started in Seconds

1.  **Install OpenEvolve:**

    ```bash
    pip install openevolve
    ```

2.  **Set your LLM API Key:** (Compatible with OpenAI, Google, and local models)

    ```bash
    export OPENAI_API_KEY="your-api-key"
    ```

3.  **Run your first evolution:**

    ```bash
    python -c "
    from openevolve import run_evolution
    result = run_evolution(
        'examples/function_minimization/initial_program.py',
        'examples/function_minimization/evaluator.py'
    )
    print(f'Best score: {result.best_score:.4f}')
    "
    ```

    *For more control, explore the [command-line interface](https://github.com/codelion/openevolve#quick-start).*

## 🎬 See OpenEvolve in Action

<details>
<summary><b>🔥 Circle Packing: From Random to State-of-the-Art</b></summary>

**Watch OpenEvolve discover optimal circle packing in real-time:**

| Generation 1 | Generation 190 | Generation 460 (Final) |
|--------------|----------------|----------------------|
| ![Initial](examples/circle_packing/circle_packing_1.png) | ![Progress](examples/circle_packing/circle_packing_190.png) | ![Final](examples/circle_packing/circle_packing_460.png) |
| Random placement | Learning structure | **State-of-the-art result** |

**Result**: Matches published benchmarks for n=26 circle packing problem.

</details>

<details>
<summary><b>⚡ GPU Kernel Evolution</b></summary>

**Before (Baseline)**:
```metal
// Standard attention implementation
kernel void attention_baseline(/* ... */) {
    // Generic matrix multiplication
    float sum = 0.0;
    for (int i = 0; i < seq_len; i++) {
        sum += query[tid] * key[i];
    }
}
```

**After Evolution (2.8x faster)**:
```metal
// OpenEvolve discovered optimization
kernel void attention_evolved(/* ... */) {
    // Hardware-aware tiling + unified memory optimization
    threadgroup float shared_mem[256];
    // ... evolved algorithm exploiting Apple Silicon architecture
}
```

**Performance Impact**: 2.8x speedup on Apple M1 Pro, maintaining numerical accuracy.

</details>

## 🧬 How OpenEvolve Works: The Architecture

OpenEvolve employs a sophisticated **evolutionary coding pipeline** that harnesses advanced techniques:

![OpenEvolve Architecture](openevolve-architecture.png)

### 🎯 Core Innovation: MAP-Elites + LLMs

*   **Quality-Diversity Evolution:** Maintain diverse populations across feature dimensions.
*   **Island-Based Architecture:** Multiple populations to prevent premature convergence.
*   **LLM Ensemble:** Multiple models with intelligent fallback strategies.
*   **Artifact Side-Channel:**  Error feedback loop to improve future generations.

## 🛠 Installation & Setup

### Requirements

*   **Python:** 3.9+
*   **LLM Access:** Any OpenAI-compatible API
*   **Optional:** Docker for containerized runs

### Installation Options

<details>
<summary><b>📦 PyPI (Recommended)</b></summary>

```bash
pip install openevolve
```

</details>

<details>
<summary><b>🔧 Development Install</b></summary>

```bash
git clone https://github.com/codelion/openevolve.git
cd openevolve
pip install -e ".[dev]"
```

</details>

<details>
<summary><b>🐳 Docker</b></summary>

```bash
docker pull ghcr.io/codelion/openevolve:latest
```

</details>

### LLM Provider Setup

OpenEvolve is compatible with **any OpenAI-compatible API**.  [See more detailed setup instructions](https://github.com/codelion/openevolve#llm-provider-setup).

## 📸 Examples Gallery

OpenEvolve empowers you to tackle a wide range of projects. Here are some highlights:

### 🏆 Showcase Projects

| Project                                                                      | Domain               | Achievement                     | Demo                                                                     |
| ---------------------------------------------------------------------------- | -------------------- | ------------------------------- | ------------------------------------------------------------------------ |
| [🎯 **Function Minimization**](examples/function_minimization/)                 | Optimization         | Random → Simulated Annealing    | [View Results](examples/function_minimization/openevolve_output/)        |
| [⚡ **MLX GPU Kernels**](examples/mlx_metal_kernel_opt/)                        | Hardware             | 2-3x Apple Silicon speedup      | [Benchmarks](examples/mlx_metal_kernel_opt/README.md)                     |
| [🔄 **Rust Adaptive Sort**](examples/rust_adaptive_sort/)                      | Algorithms           | Data-aware sorting              | [Code Evolution](examples/rust_adaptive_sort/)                            |
| [📐 **Symbolic Regression**](examples/symbolic_regression/)                     | Science              | Automated equation discovery    | [LLM-SRBench](examples/symbolic_regression/)                             |
| [🕸️ **Web Scraper + OptiLLM**](examples/web_scraper_optillm/)                   | AI Integration       | Test-time compute optimization | [Smart Scraping](examples/web_scraper_optillm/)                           |

### 🎯 Quick Example: Function Minimization

**Watch OpenEvolve evolve from random search to sophisticated optimization:**

```python
# Initial Program (Random Search)
def minimize_function(func, bounds, max_evals=1000):
    best_x, best_val = None, float('inf')
    for _ in range(max_evals):
        x = random_point_in_bounds(bounds)
        val = func(x)
        if val < best_val:
            best_x, best_val = x, val
    return best_x, best_val
```

**↓ Evolution Process ↓**

```python
# Evolved Program (Simulated Annealing + Adaptive Cooling)
def minimize_function(func, bounds, max_evals=1000):
    x = random_point_in_bounds(bounds)
    temp = adaptive_initial_temperature(func, bounds)
    
    for i in range(max_evals):
        neighbor = generate_neighbor(x, temp, bounds)
        delta = func(neighbor) - func(x)
        
        if delta < 0 or random.random() < exp(-delta/temp):
            x = neighbor
            
        temp *= adaptive_cooling_rate(i, max_evals)  # Dynamic cooling
    
    return x, func(x)
```

**Performance**: 100x improvement in convergence speed!

### 🔬 Advanced Examples

*   [🎨 **Prompt Evolution**](examples/llm_prompt_optimization/): Evolve prompts instead of code for better LLM performance (e.g., +23% accuracy on the HotpotQA benchmark).
*   [🏁 **Competitive Programming**](examples/online_judge_programming/): Generate solutions for programming contests (e.g., Brute Force → Kadane's Algorithm).

## ⚙️ Configuration

OpenEvolve offers extensive configuration options for advanced users. [Refer to the Configuration section](https://github.com/codelion/openevolve#configuration) for details.

## 🔧 Artifacts & Debugging

**Artifacts** provide rich feedback to accelerate the evolutionary process. The next generation prompt automatically includes the feedback.

This creates a **feedback loop** where each generation learns from previous mistakes!

## 📊 Visualization

OpenEvolve includes a real-time visualizer for evolution tracking.  [See the Visualizer section](https://github.com/codelion/openevolve#visualization).

## 🚀 Roadmap

### 🔥 Upcoming Features

*   Multi-Modal Evolution (Images, audio, text)
*   Federated Learning
*   AutoML Integration
*   Benchmark Suite

### 🌟 Research Directions

*   Self-Modifying Prompts
*   Cross-Language Evolution
*   Neurosymbolic Reasoning
*   Human-AI Collaboration

## 🤔 FAQ

<details>
<summary><b>💰 How much does it cost to run?</b></summary>

**Cost depends on your LLM provider and iterations**. See the [FAQ](https://github.com/codelion/openevolve#-%EF%B8%8F-faq) for detailed cost-saving tips.

</details>

<details>
<summary><b>🆚 How does this compare to manual optimization?</b></summary>

See the [FAQ](https://github.com/codelion/openevolve#-%EF%B8%8F-faq) for the comparison.

</details>

<details>
<summary><b>🔧 Can I use my own LLM?</b></summary>

**Yes!** OpenEvolve supports any OpenAI-compatible API.  See [FAQ](https://github.com/codelion/openevolve#-%EF%B8%8F-faq)

</details>

<details>
<summary><b>🚨 What if evolution gets stuck?</b></summary>

OpenEvolve has built-in mechanisms to prevent stagnation. [See FAQ](https://github.com/codelion/openevolve#-%EF%B8%8F-faq)

</details>

<details>
<summary><b>📈 How do I measure success?</b></summary>

Multiple success metrics exist. See the [FAQ](https://github.com/codelion/openevolve#-%EF%B8%8F-faq) for more details.

</details>

### 🌟 Contributors

Thanks to all our amazing contributors!

<a href="https://github.com/codelion/openevolve/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=codelion/openevolve" />
</a>

### 🤝 Contributing

We welcome contributions!  [See the Contributing Guide](CONTRIBUTING.md).

### 📚 Academic & Research

*   [Articles & Blog Posts About OpenEvolve](https://github.com/codelion/openevolve#academic--research)

## 📊 Citation

If you use OpenEvolve in your research, please cite:

```bibtex
@software{openevolve,
  title = {OpenEvolve: an open-source evolutionary coding agent},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/openevolve}
}
```

---

<div align="center">

### **🚀 Ready to evolve your code?**

**Made with ❤️ by the OpenEvolve community**

*Star ⭐ this repository if OpenEvolve helps you discover breakthrough algorithms!*

</div>
```
Key improvements and explanations:

*   **SEO-Optimized Title and Hook:** The title now clearly states the core functionality. The hook immediately grabs attention.
*   **Clear Headings and Structure:** Improved organization for readability.
*   **Concise Bullet Points:** Key features are now concisely listed.
*   **Stronger Call to Action:**  The "Ready to evolve your code?" call to action is more direct.
*   **Improved Language:** Used more active and engaging language.
*   **Links:**  Added a link back to the original repository at the beginning to maintain its credibility.
*   **FAQ Summaries:** The FAQ sections are now summaries that link back to the FAQ section.
*   **Roadmap Highlight:** The roadmap section and research directions are more clearly highlighted.
*   **Concise Explanations:** Condensed long explanations, focusing on core information.
*   **Keywords:** Incorporated relevant keywords throughout (e.g., "LLMs," "code optimization," "algorithm discovery," "evolutionary coding").
*   **Removed Redundancy:** Streamlined information.
*   **Contributed Section:** Made it more prominent.