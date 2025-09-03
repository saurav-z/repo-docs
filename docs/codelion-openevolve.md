# OpenEvolve: Unleash AI to Evolve Your Code 🧬

[Link to Original Repo: OpenEvolve](https://github.com/codelion/openevolve)

**Tired of manual code optimization? OpenEvolve is the open-source evolutionary coding agent that transforms LLMs into autonomous code optimizers, unlocking breakthrough algorithms and unprecedented performance.**

<p align="center">
  <a href="https://github.com/codelion/openevolve/stargazers"><img src="https://img.shields.io/github/stars/codelion/openevolve?style=social" alt="GitHub stars"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/v/openevolve" alt="PyPI version"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/dm/openevolve" alt="PyPI downloads"></a>
  <a href="https://github.com/codelion/openevolve/blob/main/LICENSE"><img src="https://img.shields.io/github/license/codelion/openevolve" alt="License"></a>
</p>

[🚀 **Quick Start**](#-quick-start) | [📖 **Examples**](#-examples-gallery) | [🤔 **FAQ**](#-faq)

---

## Key Features

*   **Autonomous Discovery:**  LLMs go beyond optimization, *discovering* novel algorithms without human intervention.
*   **Proven Results:** Achieve up to 3x speedups on real hardware and state-of-the-art results in areas like circle packing.
*   **Research-Grade Rigor:** Benefit from full reproducibility, comprehensive evaluation pipelines, and built-in scientific methods.

## Why Choose OpenEvolve?

| Feature                  | OpenEvolve                             | Manual Optimization                    |
| ------------------------ | -------------------------------------- | ------------------------------------- |
| **Time to Solution**     | Hours                                  | Days to Weeks                         |
| **Exploration Breadth**  | Unlimited LLM Creativity                | Limited by Human Creativity            |
| **Reproducibility**      | Fully Deterministic                     | Difficult to Replicate                 |
| **Multi-Objective**      | Automatic Pareto Optimization         | Complex Tradeoffs                      |
| **Scalability**          | Parallel Evolution Across Islands       | Doesn't Scale                          |

## Achievements & Applications

OpenEvolve excels in various domains, delivering significant performance improvements and breakthroughs:

*   **GPU Optimization:**  2-3x speedup on Apple Silicon,  e.g.,  [MLX Metal Kernels](examples/mlx_metal_kernel_opt/)
*   **Mathematical Problem Solving:** State-of-the-art circle packing solutions (n=26),  e.g., [Circle Packing](examples/circle_packing/)
*   **Algorithm Design:**  Adaptive sorting algorithms, e.g., [Rust Adaptive Sort](examples/rust_adaptive_sort/)
*   **Scientific Computing:**  Automated filter design,  e.g., [Signal Processing](examples/signal_processing/)
*   **Multi-Language Support:**  Works with Python, Rust, R, and Metal Shaders.

## 🚀 Quick Start: Evolving Code in Seconds

1.  **Install OpenEvolve:**  `pip install openevolve`
2.  **Set Your API Key:**  `export OPENAI_API_KEY="your-api-key"`  (Works with any OpenAI-compatible provider).
3.  **Run Your First Evolution:**

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

**For More Control:** Use the CLI:

```bash
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 1000
```

**Prefer Docker?**

```bash
docker run --rm -v $(pwd):/app ghcr.io/codelion/openevolve:latest \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py --iterations 100
```

## 🎬 See It in Action

**(Detailed example of Circle Packing evolution -  similar to the original)**

**(Detailed example of GPU Kernel Evolution - similar to the original)**

## 🧬 How OpenEvolve Works

OpenEvolve leverages a sophisticated **evolutionary coding pipeline**, built on the core innovation of **MAP-Elites + LLMs**.

![OpenEvolve Architecture](openevolve-architecture.png)

**Key Components:**

*   **Quality-Diversity Evolution:** Maintains diverse populations across feature dimensions.
*   **Island-Based Architecture:**  Multiple populations prevent premature convergence.
*   **LLM Ensemble:**  Multiple models with intelligent fallback strategies.
*   **Artifact Side-Channel:**  Error feedback improves subsequent generations.

## 🎯 Perfect For

| Use Case                      | Benefits of Using OpenEvolve                                 |
| ----------------------------- | ------------------------------------------------------------- |
| **Performance Optimization**  | Uncovers hardware-specific optimizations that humans miss.   |
| **Algorithm Discovery**       | Finds novel approaches to solve classic problems.           |
| **Scientific Computing**      | Automates tedious manual tuning processes.                    |
| **Competitive Programming**   | Generates multiple solution strategies.                       |
| **Multi-Objective Problems** | Creates Pareto-optimal solutions across different dimensions. |

## 🛠 Installation & Setup

**(Installation, LLM Provider Setup similar to original)**

## 📸 Examples Gallery

**(Example Gallery with projects like Function Minimization, MLX GPU Kernels, Rust Adaptive Sort, etc.  - similar to original)**

## ⚙️ Configuration

**(Configuration examples, including Feature Engineering, Custom Prompt Templates - similar to the original)**

## 🔧 Artifacts & Debugging

**(Artifacts side-channel description and example - similar to original)**

## 📊 Visualization

**(Visualization section, including features and example image - similar to original)**

## 🚀 Roadmap

**(Roadmap similar to original)**

## 🤔 FAQ

**(FAQ section similar to original)**

### 🌟 Contributors

**(Contributors section - similar to original)**

### 🤝 Contributing

**(Contributing section - similar to original)**

### 📚 Academic & Research

**(Academic & Research - similar to original)**

## 📊 Citation

**(Citation - similar to original)**

---

<div align="center">

### **🚀 Ready to revolutionize your code?**

**Made with ❤️ by the OpenEvolve community**

*Star ⭐ this repository if OpenEvolve helps you discover breakthrough algorithms!*

</div>