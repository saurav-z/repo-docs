# OpenEvolve: Unleash AI-Powered Code Evolution 🧬

**Transform your code with OpenEvolve, the open-source evolutionary coding agent that harnesses the power of Large Language Models (LLMs) to automatically discover and optimize algorithms.** [Explore the OpenEvolve Repository](https://github.com/codelion/openevolve)

[![GitHub stars](https://img.shields.io/github/stars/codelion/openevolve?style=social)](https://github.com/codelion/openevolve/stargazers)
[![PyPI version](https://img.shields.io/pypi/v/openevolve)](https://pypi.org/project/openevolve/)
[![PyPI downloads](https://img.shields.io/pypi/dm/openevolve)](https://pypi.org/project/openevolve/)
[![License](https://img.shields.io/github/license/codelion/openevolve)](https://github.com/codelion/openevolve/blob/main/LICENSE)

## Key Features:

*   **Autonomous Discovery:** LLMs don't just optimize – they discover entirely new algorithms without human guidance.
*   **Proven Results:** Achieve **2-3x speedups** on real hardware, state-of-the-art circle packing, and breakthrough optimizations.
*   **Research-Grade Reproducibility:** Fully deterministic evolution, extensive evaluation pipelines, and built-in scientific rigor.
*   **Multi-Language Support:** Python, Rust, R, and Metal shaders.
*   **Versatile Integration:**  Use OpenEvolve as a library, CLI, or within Docker containers.

## Why Choose OpenEvolve?

| Feature                 | OpenEvolve                           | Manual Optimization                 |
| ----------------------- | ------------------------------------ | ----------------------------------- |
| Time to Solution        | Hours                                | Days to weeks                       |
| Exploration Breadth     | Unlimited LLM creativity             | Limited by human expertise          |
| Reproducibility         | Fully deterministic                  | Hard to replicate                  |
| Multi-objective Optimization | Automatic Pareto optimization          | Complex tradeoffs                   |
| Scalability             | Parallel evolution across islands      | Doesn't scale                       |

##  Getting Started: Quick Start

Evolve code in **30 seconds**:

```bash
# Install OpenEvolve
pip install openevolve

# Set your LLM API key (works with any OpenAI-compatible provider)
export OPENAI_API_KEY="your-api-key"

# Run your first evolution!
python -c "
from openevolve import run_evolution
result = run_evolution(
    'examples/function_minimization/initial_program.py',
    'examples/function_minimization/evaluator.py'
)
print(f'Best score: {result.best_score:.4f}')
"
```

##  Core Technologies:

*   **MAP-Elites + LLMs:** Utilizes Quality-Diversity (QD) algorithms with Large Language Models (LLMs)
*   **Island-Based Architecture:** Parallel populations prevent premature convergence.
*   **LLM Ensemble:** Multiple models with intelligent fallback strategies.
*   **Artifact Side-Channel:** Error feedback enhances subsequent generations.

## Examples Gallery

| Project | Domain | Achievement | Demo |
|---------|--------|-------------|------|
| [🎯 **Function Minimization**](examples/function_minimization/) | Optimization | Random → Simulated Annealing | [View Results](examples/function_minimization/openevolve_output/) |
| [⚡ **MLX GPU Kernels**](examples/mlx_metal_kernel_opt/) | Hardware | 2-3x Apple Silicon speedup | [Benchmarks](examples/mlx_metal_kernel_opt/README.md) |
| [🔄 **Rust Adaptive Sort**](examples/rust_adaptive_sort/) | Algorithms | Data-aware sorting | [Code Evolution](examples/rust_adaptive_sort/) |
| [📐 **Symbolic Regression**](examples/symbolic_regression/) | Science | Automated equation discovery | [LLM-SRBench](examples/symbolic_regression/) |
| [🕸️ **Web Scraper + OptiLLM**](examples/web_scraper_optillm/) | AI Integration | Test-time compute optimization | [Smart Scraping](examples/web_scraper_optillm/) |

## Further Information:

*   **Installation and Setup:** Detailed instructions on [installation](#-installation--setup) and LLM provider configuration.
*   **How OpenEvolve Works:** An in-depth look at the [evolutionary coding pipeline](#-how-openevolve-works) and its features.
*   **Configuration:** Instructions on [advanced configurations](#-configuration) for customizing your evolution experiments.
*   **Roadmap:** Explore [upcoming features and research directions](#-roadmap).
*   **FAQ:** Get answers to [frequently asked questions](#-faq) about OpenEvolve.
*   **Contributing:** Learn how to [contribute to OpenEvolve](#-contributing).
*   **Citation:** Reference OpenEvolve in your research with our [citation information](#-citation).

---
<div align="center">
**Ready to Evolve? ⭐ Star this repository to support OpenEvolve and unlock the future of code optimization!**
</div>
```
Key improvements:

*   **SEO-optimized hook:**  The opening sentence is a strong, keyword-rich hook that immediately explains what OpenEvolve does.
*   **Clear Headings:**  Improved headings for better structure and readability (e.g., "Key Features").
*   **Bulleted Lists:**  Key features and benefits are presented in easy-to-scan bulleted lists.
*   **Concise summaries:** Redundant language removed to keep it concise.
*   **Emphasis:** Stronger use of bolding to highlight key information.
*   **Actionable links:** The link back to the original repo is prominent.
*   **Reduced duplication:** Trimmed repeated sections of the original readme
*   **Combined sections:** More concise combined sections such as "Why Choose OpenEvolve?"
*   **Overall Tone:**  The revised README is more professional, engaging, and informative.