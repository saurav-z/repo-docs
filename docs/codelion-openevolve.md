# OpenEvolve: Evolving Code with AI 🧬

**Unlock the power of autonomous code optimization by turning your LLMs into evolutionary coding agents!** This open-source project empowers you to discover breakthrough algorithms, automate performance enhancements, and push the boundaries of software development.  [Explore OpenEvolve on GitHub](https://github.com/codelion/openevolve).

<p align="center">
  <a href="https://github.com/codelion/openevolve/stargazers"><img src="https://img.shields.io/github/stars/codelion/openevolve?style=social" alt="GitHub stars"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/v/openevolve" alt="PyPI version"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/dm/openevolve" alt="PyPI downloads"></a>
  <a href="https://github.com/codelion/openevolve/blob/main/LICENSE"><img src="https://img.shields.io/github/license/codelion/openevolve" alt="License"></a>
</p>

[🚀 **Quick Start**](#quick-start) • [**Key Features**](#key-features) • [**Examples**](#examples-gallery) • [**How It Works**](#how-openevolve-works) • [**Installation**](#installation--setup)

---

## Key Features

*   **Autonomous Discovery:** LLMs discover entirely new algorithms without human intervention.
*   **Proven Results:** Achieve **2-3x speedups** on real hardware and state-of-the-art results in areas like circle packing.
*   **Research Grade:** Built-in reproducibility, evaluation pipelines, and scientific rigor.

## Why OpenEvolve?  OpenEvolve vs. Manual Optimization

| Feature               | Manual Optimization                 | OpenEvolve                                     |
|-----------------------|-------------------------------------|------------------------------------------------|
| Time to Solution      | Days to weeks                        | Hours                                          |
| Exploration Breadth   | Limited by human creativity          | Unlimited LLM creativity                        |
| Reproducibility       | Difficult to replicate              | Fully deterministic                               |
| Multi-objective Opt. | Complex tradeoffs                   | Automatic Pareto optimization                    |
| Scalability           | Doesn't scale                        | Parallel evolution across islands                |

## Proven Achievements

| Domain                  | Achievement                                   | Example                                                     |
|-----------------------|-----------------------------------------------|-------------------------------------------------------------|
| GPU Optimization        | Hardware-optimized kernel discovery          | [MLX Metal Kernels](examples/mlx_metal_kernel_opt/)          |
| Mathematical            | State-of-the-art circle packing (n=26)       | [Circle Packing](examples/circle_packing/)                   |
| Algorithm Design        | Adaptive sorting algorithms                   | [Rust Adaptive Sort](examples/rust_adaptive_sort/)          |
| Scientific Computing  | Automated filter design                       | [Signal Processing](examples/signal_processing/)            |
| Multi-Language          | Python, Rust, R, Metal shaders                 | [All Examples](examples/)                                 |

## 🚀 Quick Start

Evolve your first code in **30 seconds**:

```bash
# Install OpenEvolve
pip install openevolve

# Set your API key (uses Google Gemini by default, see [LLM Provider Setup](#llm-provider-setup))
export OPENAI_API_KEY="your-gemini-api-key"  # Or configure in config.yaml

# Run an example
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 50
```

**Alternatively, use OpenEvolve as a library:**

```python
from openevolve import run_evolution, evolve_function

# Evolve inline code
result = run_evolution(
    initial_program='''...''',  # Your code
    evaluator=lambda path: {"score": benchmark_func(path)},
    iterations=100
)

# Evolve functions directly
def my_function(input):  #Your function
    ...

result = evolve_function(
    my_function,
    test_cases=[(input1, expected_output1), ...],
    iterations=50
)
print(f"Evolved: {result.best_code}")
```

## See It In Action

<details>
<summary><b>Circle Packing: From Random to State-of-the-Art</b></summary>

**Observe OpenEvolve's real-time evolution towards optimal circle packing:**

| Generation 1 | Generation 190 | Generation 460 (Final) |
|--------------|----------------|----------------------|
| ![Initial](examples/circle_packing/circle_packing_1.png) | ![Progress](examples/circle_packing/circle_packing_190.png) | ![Final](examples/circle_packing/circle_packing_460.png) |
| Random placement | Learning structure | **State-of-the-art result** |

**Result:** Matches published benchmarks for the n=26 circle packing problem.

</details>

<details>
<summary><b>GPU Kernel Evolution</b></summary>

**Before (Baseline):**

```metal
// Standard implementation
kernel void attention_baseline(/* ... */) { ... }
```

**After Evolution (2.8x faster):**

```metal
// OpenEvolve optimized
kernel void attention_evolved(/* ... */) { ... }
```

**Performance Impact:** 2.8x speedup on Apple M1 Pro, while maintaining numerical accuracy.

</details>

## How OpenEvolve Works

OpenEvolve employs a sophisticated **evolutionary coding pipeline**, that leverages:

![OpenEvolve Architecture](openevolve-architecture.png)

### Core Innovations:

*   **MAP-Elites + LLMs:** Quality-Diversity evolution for diverse populations.
*   **Island-Based Architecture:** Multiple populations to prevent premature convergence.
*   **LLM Ensemble:** Multiple models with intelligent fallback strategies.
*   **Artifact Side-Channel:** Error feedback to inform and improve the next generations.

## 🛠 Installation & Setup

### Requirements:

*   Python 3.10+
*   Access to an OpenAI-compatible API (e.g., OpenAI, Google Gemini)
*   Optional: Docker for containerized runs

### Installation Options:

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
# Pull the image
docker pull ghcr.io/codelion/openevolve:latest

# Run an example
docker run --rm -v $(pwd):/app ghcr.io/codelion/openevolve:latest \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py --iterations 100
```

</details>

### Cost Estimation:

Cost depends on your LLM provider and iteration count. Here are some examples:

*   **o3**: ~$0.15-0.60 per iteration (depending on code size)
*   **o3-mini**: ~$0.03-0.12 per iteration (more cost-effective)
*   **Gemini-2.5-Pro**: ~$0.08-0.30 per iteration
*   **Gemini-2.5-Flash**: ~$0.01-0.05 per iteration (fastest and cheapest)
*   **Local models**: Nearly free after setup
*   **OptiLLM**: Use cheaper models with test-time compute for better results

**Cost-saving tips:**

*   Start with fewer iterations.
*   Use cheaper models or local options for exploration.
*   Employ cascade evaluation to filter bad programs early.
*   Configure smaller population sizes initially.

### LLM Provider Setup

OpenEvolve is compatible with any OpenAI-compatible API.

<details>
<summary><b>🔥 OpenAI (Direct)</b></summary>

```bash
export OPENAI_API_KEY="sk-..."
```

</details>

<details>
<summary><b>🤖 Google Gemini</b></summary>

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
<summary><b>🏠 Local Models (Ollama/vLLM)</b></summary>

```yaml
# config.yaml
llm:
  api_base: "http://localhost:11434/v1"  # Ollama
  model: "codellama:7b"
```

</details>

<details>
<summary><b>⚡ OptiLLM (Advanced)</b></summary>

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

## Examples Gallery

<div align="center">

### Showcase Projects

| Project                               | Domain             | Achievement                             | Demo                                                                   |
|---------------------------------------|--------------------|-----------------------------------------|------------------------------------------------------------------------|
| [Function Minimization](examples/function_minimization/) | Optimization         | Random → Simulated Annealing            | [View Results](examples/function_minimization/openevolve_output/)      |
| [MLX GPU Kernels](examples/mlx_metal_kernel_opt/)          | Hardware           | Apple Silicon optimization            | [Benchmarks](examples/mlx_metal_kernel_opt/README.md)                |
| [Rust Adaptive Sort](examples/rust_adaptive_sort/)         | Algorithms           | Data-aware sorting                    | [Code Evolution](examples/rust_adaptive_sort/)                       |
| [Symbolic Regression](examples/symbolic_regression/)       | Science            | Automated equation discovery          | [LLM-SRBench](examples/symbolic_regression/)                          |
| [Web Scraper + OptiLLM](examples/web_scraper_optillm/)      | AI Integration     | Test-time compute optimization        | [Smart Scraping](examples/web_scraper_optillm/)                      |

</div>

## How to Configure

OpenEvolve offers extensive configuration options for advanced users, enabling you to control aspects like:

```yaml
# Advanced Configuration Example
max_iterations: 1000
random_seed: 42  # Full reproducibility

llm:
  # Ensemble configuration
  models:
    - name: "gemini-2.5-pro"
      weight: 0.6
    - name: "gemini-2.5-flash"
      weight: 0.4
  temperature: 0.7

database:
  # MAP-Elites quality-diversity
  population_size: 500
  num_islands: 5  # Parallel evolution
  migration_interval: 20
  feature_dimensions: ["complexity", "diversity", "performance"]

evaluator:
  enable_artifacts: true      # Error feedback to LLM
  cascade_evaluation: true    # Multi-stage testing
  use_llm_feedback: true      # AI code quality assessment

prompt:
  # Sophisticated inspiration system
  num_top_programs: 3         # Best performers
  num_diverse_programs: 2     # Creative exploration
  include_artifacts: true     # Execution feedback

  # Custom templates
  template_dir: "custom_prompts/"
  use_template_stochasticity: true  # Randomized prompts
```

### Feature Engineering:

Control the organization of programs in the quality-diversity grid using custom feature dimensions and bins.

```yaml
database:
  feature_dimensions:
    - "complexity"      # Built-in: code length
    - "diversity"       # Built-in: structural diversity
    - "performance"     # Custom: from your evaluator
    - "memory_usage"    # Custom: from your evaluator

  feature_bins:
    complexity: 10      # 10 complexity levels
    performance: 20     # 20 performance buckets
    memory_usage: 15    # 15 memory usage categories
```

### Custom Prompt Templates:

Fine-tune prompt engineering using custom templates, randomized variations, and placeholders:

```yaml
prompt:
  template_dir: "custom_templates/"
  use_template_stochasticity: true
  template_variations:
    greeting:
      - "Let's enhance this code:"
      - "Time to optimize:"
      - "Improving the algorithm:"
    improvement_suggestion:
      - "Here's how we could improve this code:"
      - "I suggest the following improvements:"
      - "We can enhance this code by:"
```

## Crafting Effective System Messages

Well-crafted system messages are crucial for guiding the LLM towards successful evolution. The iterative process involves drafting, refining, and specializing prompts.

### Examples:

*   **Simple: General Optimization:**
    ```yaml
    prompt:
      system_message: |
        You are an expert programmer specializing in optimization algorithms.
        Your task is to improve a function minimization algorithm to find the
        global minimum reliably, escaping local minima.
    ```

*   **Intermediate: Domain-Specific Guidance:**
    ```yaml
    prompt:
      system_message: |
        You are an expert prompt engineer.  Revise prompts for LLMs.
        Improve:
        *   Clarify instructions, eliminate ambiguity.
        *   Strengthen alignment between prompt and task.
        *   Improve robustness against edge cases.
        *   Include formatting instructions/examples.
        *   Avoid unnecessary verbosity.
    ```

*   **Advanced: Hardware-Specific Optimization:**
    ```yaml
    prompt:
      system_message: |
        You are an expert Metal GPU programmer optimizing attention
        kernels for Apple Silicon.
        # TARGET: Optimize Metal Kernel for Grouped Query Attention (GQA)
        # HARDWARE: Apple M-series GPUs with unified memory architecture
        # GOAL: 5-15% performance improvement
        # OPTIMIZATION OPPORTUNITIES:
        **1. Memory Access Pattern Optimization:**
        - Coalesced access patterns for Apple Silicon
        - Vectorized loading using SIMD
        - Pre-compute frequently used indices
        **2. Algorithm Fusion:**
        - Combine max finding with score computation
        - Reduce number of passes through data
        # CONSTRAINTS - CRITICAL SAFETY RULES:
        **MUST NOT CHANGE:**
        ❌ Kernel function signature
        ❌ Template parameter names or types
        ❌ Overall algorithm correctness
        **ALLOWED TO OPTIMIZE:**
        ✅ Memory access patterns and indexing
        ✅ Computation order and efficiency
        ✅ Vectorization and SIMD utilization
        ✅ Apple Silicon specific optimizations
    ```

### Best Practices:

*   Structure: Role Definition → Task/Context → Optimization Opportunities → Constraints → Success Criteria.
*   Use specific examples.
*   Include domain knowledge.
*   Set clear boundaries.

## Artifacts & Debugging

Leverage the **artifacts side-channel** for rich feedback:

```python
# Evaluator
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

This creates a **feedback loop**!

## Visualization

**Real-time evolution tracking** using the interactive web interface:

```bash
pip install -r scripts/requirements.txt
python scripts/visualizer.py
# or visualize a specific checkpoint:
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

**Features:**

*   🌳 Evolution tree
*   📈 Performance tracking
*   🔍 Code diff viewer
*   📊 MAP-Elites grid visualization
*   🎯 Multi-metric analysis

![OpenEvolve Visualizer](openevolve-visualizer.png)

## Roadmap

### Upcoming Features:

*   Multi-Modal Evolution
*   Federated Learning
*   AutoML Integration
*   Benchmark Suite

### Research Directions:

*   Self-Modifying Prompts
*   Cross-Language Evolution
*   Neurosymbolic Reasoning
*   Human-AI Collaboration

## FAQ

<details>
<summary><b>💰 How much does it cost to run?</b></summary>
See the [Cost Estimation](#cost-estimation) section in Installation & Setup for detailed pricing information.
</details>

<details>
<summary><b>🆚 How does this compare to manual optimization?</b></summary>
| Aspect               | Manual                      | OpenEvolve                   |
|----------------------|-----------------------------|------------------------------|
| Initial Learning     | Weeks to understand domain | Minutes to start             |
| Solution Quality     | Depends on expertise        | Explores novel approaches    |
| Time Investment      | Days-weeks per optimization | Hours for complete evolution |
| Reproducibility      | Hard to replicate          | Perfect reproduction         |
| Scaling              | Doesn't scale               | Parallel evolution           |

**OpenEvolve excels** when exploring large solution spaces.
</details>

<details>
<summary><b>🔧 Can I use my own LLM?</b></summary>
**Yes!**  Supports any OpenAI-compatible API. Just set the `api_base` in your config.
</details>

<details>
<summary><b>🚨 What if evolution gets stuck?</b></summary>
Built-in mechanisms and manual interventions help prevent stagnation.
</details>

<details>
<summary><b>📈 How do I measure success?</b></summary>
Use multiple metrics, including a primary score, convergence, diversity, efficiency, and robustness.  The visualizer assists with real-time tracking.
</details>

### Contributors

Thank you to all our amazing contributors!

<a href="https://github.com/codelion/openevolve/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=codelion/openevolve" />
</a>

### Contributing

Contributions are welcome! Follow these steps:

1.  Fork the repository.
2.  Create a feature branch: `git checkout -b feat-amazing-feature`
3.  Add your changes and tests.
4.  Test everything: `python -m unittest discover tests`
5.  Commit with a clear message.
6.  Push and create a Pull Request.

Check out our [Contributing Guide](CONTRIBUTING.md) and look for [`good-first-issue`](https://github.com/codelion/openevolve/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) labels!

### Academic & Research

**Articles & Blog Posts About OpenEvolve**:

*   [Towards Open Evolutionary Agents](https://huggingface.co/blog/driaforall/towards-open-evolutionary-agents)
*   [OpenEvolve: GPU Kernel Discovery](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery)
*   [OpenEvolve: Evolutionary Coding with LLMs](https://huggingface.co/blog/codelion/openevolve)

## Citation

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

### 🚀 Ready to evolve your code?

**Maintained by the OpenEvolve community**

*If OpenEvolve helps you discover breakthrough algorithms, please consider starring this repository.*

</div>