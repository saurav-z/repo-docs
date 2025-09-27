# OpenEvolve: The Autonomous Evolutionary Coding Agent

**Tired of manually optimizing code?** OpenEvolve is an open-source tool that transforms LLMs into autonomous code optimizers, capable of discovering novel and high-performing algorithms.  From GPU kernels to mathematical equations, watch your code evolve in real-time! Explore the original repo: [https://github.com/codelion/openevolve](https://github.com/codelion/openevolve).

<p align="center">
  <a href="https://github.com/codelion/openevolve/stargazers"><img src="https://img.shields.io/github/stars/codelion/openevolve?style=social" alt="GitHub stars"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/v/openevolve" alt="PyPI version"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/dm/openevolve" alt="PyPI downloads"></a>
  <a href="https://github.com/codelion/openevolve/blob/main/LICENSE"><img src="https://img.shields.io/github/license/codelion/openevolve" alt="License"></a>
</p>

## Key Features

*   **Autonomous Discovery:** Automates algorithm discovery with no manual intervention.
*   **Proven Results:** Achieves 2-3x speedups on hardware and state-of-the-art performance.
*   **Reproducible Research:** Built-in reproducibility, extensive evaluation pipelines, and scientific rigor.
*   **Multi-Language Support:** Works with Python, Rust, Metal shaders, and more.
*   **Flexible LLM Integration:** Compatible with OpenAI, Google Gemini, local models (Ollama, vLLM), and more.
*   **Advanced Features:**  MAP-Elites, LLM ensembles, scientific reproducibility, and extensive configuration options.

## Why Use OpenEvolve?

| Feature                | Manual Optimization      | OpenEvolve               |
| ---------------------- | ------------------------ | ------------------------ |
| **Time to Solution**   | Days to Weeks            | Hours                    |
| **Exploration Breadth** | Limited by Human         | Unlimited LLM Creativity |
| **Reproducibility**    | Difficult                | Fully Deterministic      |
| **Multi-objective**    | Complex Tradeoffs        | Automatic Pareto         |
| **Scaling**            | Doesn't Scale            | Parallel Evolution       |

## Proven Achievements

| Domain               | Achievement                                | Example                                                                    |
| -------------------- | ------------------------------------------ | -------------------------------------------------------------------------- |
| GPU Optimization     | Hardware-optimized kernel discovery        | [MLX Metal Kernels](examples/mlx_metal_kernel_opt/)                        |
| Mathematical         | State-of-the-art circle packing (n=26)     | [Circle Packing](examples/circle_packing/)                                   |
| Algorithm Design     | Adaptive sorting algorithms                | [Rust Adaptive Sort](examples/rust_adaptive_sort/)                          |
| Scientific Computing | Automated filter design                    | [Signal Processing](examples/signal_processing/)                            |
| Multi-Language       | Python, Rust, R, Metal shaders, and more | [All Examples](examples/)                                                   |

## Quick Start

Get started in seconds:

```bash
# Install OpenEvolve
pip install openevolve

# Set your LLM API key (example uses Google Gemini)
export OPENAI_API_KEY="YOUR_API_KEY"

# Run a function minimization example
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 50
```

### Library Usage

OpenEvolve can be integrated directly into your Python code.

```python
from openevolve import run_evolution, evolve_function

# Evolve with inline code (no external files!)
result = run_evolution(
    initial_program='''
    def fibonacci(n):
        if n <= 1: return n
        return fibonacci(n-1) + fibonacci(n-2)
    ''',
    evaluator=lambda path: {"score": benchmark_fib(path)},
    iterations=100
)

# Evolve Python functions directly
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

result = evolve_function(
    bubble_sort,
    test_cases=[([3,1,2], [1,2,3]), ([5,2,8], [2,5,8])],
    iterations=50
)
print(f"Evolved sorting algorithm: {result.best_code}")
```

## See It In Action: Circle Packing

Watch OpenEvolve find the optimal solution:

| Generation 1 | Generation 190  | Generation 460 (Final)     |
|--------------|-----------------|----------------------------|
| ![Initial](examples/circle_packing/circle_packing_1.png) | ![Progress](examples/circle_packing/circle_packing_190.png) | ![Final](examples/circle_packing/circle_packing_460.png)   |
| Random       | Learning       | **State-of-the-Art Result** |

**Result**: Matches published benchmarks for n=26 circle packing problems.

## How OpenEvolve Works

OpenEvolve uses an **evolutionary coding pipeline** based on innovative features:

![OpenEvolve Architecture](openevolve-architecture.png)

### Core Innovation: MAP-Elites + LLMs

*   **Quality-Diversity Evolution:** Maintains diverse code populations across feature dimensions.
*   **Island-Based Architecture:** Prevents premature convergence.
*   **LLM Ensemble:** Multiple models with fallback strategies.
*   **Artifact Side-Channel:** Error feedback from previous generations.

## Installation and Setup

### Requirements

*   Python 3.10+
*   Access to any OpenAI-compatible LLM API

### Installation Options

*   **PyPI (Recommended):** `pip install openevolve`
*   **Development Install:** Clone the repo, then `pip install -e ".[dev]"`
*   **Docker:**  `docker pull ghcr.io/codelion/openevolve:latest`

### LLM Provider Setup

OpenEvolve is compatible with OpenAI and other compatible APIs:

*   **OpenAI (Direct):** Set `OPENAI_API_KEY`.
*   **Google Gemini:**  Set `OPENAI_API_KEY` and configure `api_base` and `model` in `config.yaml`.
*   **Local Models (Ollama/vLLM):** Configure `api_base` and `model` in `config.yaml`.
*   **OptiLLM (Advanced):**  Integrate with a proxy for model routing, rate limiting, and test-time compute.

## Examples Gallery

*   **Function Minimization:** Evolve code from random search to simulated annealing.
*   **MLX GPU Kernels:**  Automate hardware-specific optimization on Apple Silicon.
*   **Rust Adaptive Sort:**  Discover adaptive sorting algorithms.
*   **Symbolic Regression:** Discover equations automatically.
*   **Web Scraper + OptiLLM:** AI integration and test-time compute optimization.

## Configuration

*   Control the evolution process with a robust configuration system.
*   Define feature dimensions and custom bins for program organization.
*   Leverage custom prompt templates.

## Crafting Effective System Messages

*   System messages are crucial for guiding the LLM.
*   Iteratively refine your messages for better results.
*   Use clear instructions, domain-specific language, and examples.
*   Consider artifact-driven iteration.
*   Optimize prompts using OpenEvolve itself!

## Artifacts & Debugging

*   Use the artifacts side-channel to gather feedback and debug programs.
*   Integrate `stderr`, warnings, and LLM feedback to accelerate evolution.

## Visualization

*   Track real-time evolution with the interactive web interface.

## Roadmap

*   Multi-Modal Evolution
*   Federated Learning
*   AutoML Integration
*   Benchmark Suite

## FAQ

### Cost Estimation
See the [Cost Estimation](#cost-estimation) section in Installation & Setup for detailed pricing information and cost-saving tips.

### Can I use my own LLM?

Yes, OpenEvolve supports any OpenAI-compatible API.  Set the `api_base` in your config.

### What if evolution gets stuck?

Built-in mechanisms like island migration and artifact feedback prevent stagnation.
You can also adjust parameters manually.

### How do I measure success?

Use metrics like convergence, diversity, and efficiency. Track these using the visualizer.

## Contributing

We welcome contributions.  See the [Contributing Guide](CONTRIBUTING.md) and look for `good-first-issue` labels.

## Academic & Research

If you use OpenEvolve in your research, please cite:

```bibtex
@software{openevolve,
  title = {OpenEvolve: an open-source evolutionary coding agent},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/openevolve}
}