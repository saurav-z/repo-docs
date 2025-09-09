# OpenEvolve: Evolve Code, Unleash Innovation

**OpenEvolve** empowers you to transform Large Language Models (LLMs) into autonomous code optimizers, driving the discovery of breakthrough algorithms. [Explore the original repo](https://github.com/codelion/openevolve) to revolutionize your code optimization process.

<div align="center">
    <img src="openevolve-logo.png" alt="OpenEvolve Logo" width="400">
    <p align="center">
        <a href="https://github.com/codelion/openevolve/stargazers"><img src="https://img.shields.io/github/stars/codelion/openevolve?style=social" alt="GitHub stars"></a>
        <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/v/openevolve" alt="PyPI version"></a>
        <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/dm/openevolve" alt="PyPI downloads"></a>
        <a href="https://github.com/codelion/openevolve/blob/main/LICENSE"><img src="https://img.shields.io/github/license/codelion/openevolve" alt="License"></a>
    </p>
</div>

---

## Key Features

*   **Autonomous Discovery:** No human guidance needed.  LLMs evolve code, often discovering novel algorithms.
*   **Proven Results:** Achieve 2-3x speedups on real hardware and state-of-the-art results in areas like circle packing.
*   **Research Grade:** Built for reproducibility, with comprehensive evaluation pipelines and rigorous scientific methods.

## Why Choose OpenEvolve?

OpenEvolve offers a paradigm shift in code optimization, outperforming manual methods in several key areas:

**Manual Optimization vs. OpenEvolve**

| Feature              | Manual Optimization          | OpenEvolve                               |
| -------------------- | ---------------------------- | ---------------------------------------- |
| Time to Solution     | Days to weeks                | Hours                                    |
| Exploration Breadth  | Limited by human creativity  | Unlimited LLM creativity                 |
| Reproducibility      | Difficult to replicate      | Fully deterministic                       |
| Multi-objective     | Complex tradeoffs           | Automatic Pareto optimization            |
| Scaling              | Doesn't scale efficiently   | Parallel evolution across multiple systems |

## Achievements

OpenEvolve has achieved significant breakthroughs across various domains:

| Domain                 | Achievement                                       | Example                                                               |
| ---------------------- | ------------------------------------------------- | --------------------------------------------------------------------- |
| GPU Optimization       | 2-3x speedup on Apple Silicon                     | [MLX Metal Kernels](examples/mlx_metal_kernel_opt/)                   |
| Mathematical           | State-of-the-art circle packing (n=26)            | [Circle Packing](examples/circle_packing/)                             |
| Algorithm Design       | Adaptive sorting algorithms                       | [Rust Adaptive Sort](examples/rust_adaptive_sort/)                     |
| Scientific Computing   | Automated filter design                           | [Signal Processing](examples/signal_processing/)                       |
| Multi-Language         | Python, Rust, R, Metal shaders                    | [All Examples](examples/)                                               |

## Getting Started: Quick Start

Get up and running in 30 seconds:

```bash
# Install OpenEvolve
pip install openevolve

# Set your LLM API key (compatible with any OpenAI-compatible provider)
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

## Core Functionality

OpenEvolve provides rich library and CLI usage:

### Library Usage

```python
from openevolve import run_evolution, evolve_function

# Evolution with inline code
result = run_evolution(
    initial_program='''
    def fibonacci(n):
        if n <= 1: return n
        return fibonacci(n-1) + fibonacci(n-2)
    ''',
    evaluator=lambda path: {"score": benchmark_fib(path)},
    iterations=100
)
```

### CLI Usage

```bash
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 1000
```

### Docker

```bash
docker run --rm -v $(pwd):/app ghcr.io/codelion/openevolve:latest \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py --iterations 100
```

## Examples Gallery

### Circle Packing

**Watch OpenEvolve discover optimal circle packing in real-time:**

| Generation 1 | Generation 190 | Generation 460 (Final) |
|--------------|----------------|----------------------|
| ![Initial](examples/circle_packing/circle_packing_1.png) | ![Progress](examples/circle_packing/circle_packing_190.png) | ![Final](examples/circle_packing/circle_packing_460.png) |
| Random placement | Learning structure | **State-of-the-art result** |

### GPU Kernel Evolution

**Before (Baseline)**:

```metal
kernel void attention_baseline(/* ... */) {
    // Standard attention implementation
    float sum = 0.0;
    for (int i = 0; i < seq_len; i++) {
        sum += query[tid] * key[i];
    }
}
```

**After Evolution (2.8x faster)**:

```metal
kernel void attention_evolved(/* ... */) {
    // Hardware-aware tiling + unified memory optimization
    threadgroup float shared_mem[256];
    // ... evolved algorithm exploiting Apple Silicon architecture
}
```

**Performance Impact**: 2.8x speedup on Apple M1 Pro, maintaining numerical accuracy.

## How OpenEvolve Works

OpenEvolve uses a sophisticated evolutionary coding pipeline that goes beyond simple optimization:

![OpenEvolve Architecture](openevolve-architecture.png)

### Core Innovation: MAP-Elites + LLMs

*   **Quality-Diversity Evolution**: Preserves diverse populations across feature dimensions.
*   **Island-Based Architecture**: Prevents premature convergence through multiple populations.
*   **LLM Ensemble**: Employs multiple models with intelligent fallback.
*   **Artifact Side-Channel**: Leverages error feedback to enhance subsequent generations.

## Configuration

OpenEvolve provides extensive configuration options for customization and advanced use.

### LLM Provider Setup

OpenEvolve supports **any OpenAI-compatible API**:

<details>
<summary>🔥 **OpenAI (Direct)**</summary>

```bash
export OPENAI_API_KEY="sk-..."
# Uses OpenAI endpoints by default
```

</details>

<details>
<summary>🤖 **Google Gemini**</summary>

```yaml
# config.yaml
llm:
  api_base: "https://generativelanguage.googleapis.com/v1beta/openai/"
  model: "gemini-2.5-pro"
```

```bash
export OPENAI_API_KEY="your-gemini-api-key"
```

</details>

<details>
<summary>🏠 **Local Models (Ollama/vLLM)**</summary>

```yaml
# config.yaml
llm:
  api_base: "http://localhost:11434/v1"  # Ollama
  model: "codellama:7b"
```

</details>

<details>
<summary>⚡ **OptiLLM (Advanced)**</summary>

For maximum flexibility with rate limiting, model routing, and test-time compute:

```bash
# Install OptiLLM
pip install optillm

# Start OptiLLM proxy
optillm --port 8000

# Point OpenEvolve to OptiLLM
export OPENAI_API_KEY="your-actual-key"
```

```yaml
llm:
  api_base: "http://localhost:8000/v1"
  model: "moa&readurls-o3"  # Test-time compute + web access
```

</details>

## Installation

### Requirements

*   **Python**: 3.10+
*   **LLM Access**: Any OpenAI-compatible API
*   **Optional**: Docker for containerized runs

### Installation Options

<details>
<summary>📦 **PyPI (Recommended)**</summary>

```bash
pip install openevolve
```

</details>

<details>
<summary>🔧 **Development Install**</summary>

```bash
git clone https://github.com/codelion/openevolve.git
cd openevolve
pip install -e ".[dev]"
```

</details>

<details>
<summary>🐳 **Docker**</summary>

```bash
docker pull ghcr.io/codelion/openevolve:latest
```

</details>

## Artifacts & Debugging

**Artifacts side-channel** provides rich feedback to accelerate evolution:

```python
# Evaluator can return execution context
from openevolve.evaluation_result import EvaluationResult

return EvaluationResult(
    metrics={"performance": 0.85, "correctness": 1.0},
    artifacts={
        "stderr": "Warning: suboptimal memory access pattern",
        "profiling_data": {...},
        "llm_feedback": "Code is correct but could use better variable names",
        "build_warnings": ["unused variable x"]
    }
)
```

**Next generation prompt automatically includes:**
```markdown
## Previous Execution Feedback
⚠️ Warning: suboptimal memory access pattern
💡 LLM Feedback: Code is correct but could use better variable names
🔧 Build Warnings: unused variable x
```

This creates a **feedback loop** where each generation learns from previous mistakes!

## Roadmap

### 🔥 Upcoming Features

*   Multi-Modal Evolution: Images, audio, and text simultaneously
*   Federated Learning: Distributed evolution across multiple machines
*   AutoML Integration: Hyperparameter and architecture evolution
*   Benchmark Suite: Standardized evaluation across domains

### 🌟 Research Directions

*   Self-Modifying Prompts: Evolution modifies its own prompting strategy
*   Cross-Language Evolution: Python → Rust → C++ optimization chains
*   Neurosymbolic Reasoning: Combine neural and symbolic approaches
*   Human-AI Collaboration: Interactive evolution with human feedback

## FAQ

<details>
<summary>💰 **How much does it cost to run?**</summary>

**Cost depends on your LLM provider and iterations:**

*   **o3**: ~$0.15-0.60 per iteration (depending on code size)
*   **o3-mini**: ~$0.03-0.12 per iteration (more cost-effective)
*   **Gemini-2.5-Pro**: ~$0.08-0.30 per iteration
*   **Gemini-2.5-Flash**: ~$0.01-0.05 per iteration (fastest and cheapest)
*   **Local models**: Nearly free after setup
*   **OptiLLM**: Use cheaper models with test-time compute for better results

</details>

<details>
<summary>🆚 **How does this compare to manual optimization?**</summary>

| Aspect              | Manual | OpenEvolve    |
| ------------------- | ------ | ------------- |
| Initial Learning    | Weeks  | Minutes       |
| Solution Quality    | Expertise | Novel approaches |
| Time Investment     | Days-weeks | Hours      |
| Reproducibility     | Hard   | Perfect       |
| Scaling             | Limited  | Parallel  |

</details>

<details>
<summary>🔧 **Can I use my own LLM?**</summary>

**Yes!** OpenEvolve supports any OpenAI-compatible API:

*   Commercial: OpenAI, Google, Cohere
*   Local: Ollama, vLLM, LM Studio, text-generation-webui
*   Advanced: OptiLLM for routing and test-time compute

Just set the `api_base` in your config to point to your endpoint.

</details>

<details>
<summary>🚨 **What if evolution gets stuck?**</summary>

**Built-in mechanisms prevent stagnation:**

*   Island migration: Fresh genes from other populations
*   Temperature control: Exploration vs exploitation balance
*   Diversity maintenance: MAP-Elites prevents convergence
*   Artifact feedback: Error messages guide improvements
*   Template stochasticity: Randomized prompts break patterns

**Manual interventions:**
* Increase `num_diverse_programs` for more exploration
* Add custom feature dimensions to diversify search
* Use template variations to randomize prompts
* Adjust migration intervals for more cross-pollination

</details>

<details>
<summary>📈 **How do I measure success?**</summary>

**Multiple success metrics:**

1.  Primary Metric: Your evaluator's `combined_score` or metric average
2.  Convergence: Best score improvement over time
3.  Diversity: MAP-Elites grid coverage
4.  Efficiency: Iterations to reach target performance
5.  Robustness: Performance across different test cases

</details>

### 🌟 Contributors

Thanks to all our amazing contributors!
<a href="https://github.com/codelion/openevolve/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=codelion/openevolve" />
</a>

### 🤝 Contributing

We welcome contributions!

1.  Fork the repository.
2.  Create your feature branch: `git checkout -b feat-amazing-feature`
3.  Add your changes and tests.
4.  Test everything: `python -m unittest discover tests`
5.  Commit with a clear message.
6.  Push and create a Pull Request.

### 📚 Academic & Research

**Articles & Blog Posts About OpenEvolve**:
- [Towards Open Evolutionary Agents](https://huggingface.co/blog/driaforall/towards-open-evolutionary-agents) - Evolution of coding agents and the open-source movement
- [OpenEvolve: GPU Kernel Discovery](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery) - Automated discovery of optimized GPU kernels with 2-3x speedups
- [OpenEvolve: Evolutionary Coding with LLMs](https://huggingface.co/blog/codelion/openevolve) - Introduction to evolutionary algorithm discovery using large language models

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
### 🚀 Ready to Evolve Your Code?
**Made with ❤️ by the OpenEvolve community**
*Star ⭐ this repository if OpenEvolve helps you discover breakthrough algorithms!*
</div>