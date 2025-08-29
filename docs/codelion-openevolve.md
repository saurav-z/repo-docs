# OpenEvolve: Unleash AI to Evolve and Optimize Your Code

**Harness the power of AI to discover breakthrough algorithms with OpenEvolve, the most advanced open-source evolutionary coding agent. [Explore the Repo](https://github.com/codelion/openevolve)**

<p align="center">
  <a href="https://github.com/codelion/openevolve/stargazers"><img src="https://img.shields.io/github/stars/codelion/openevolve?style=social" alt="GitHub stars"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/v/openevolve" alt="PyPI version"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/dm/openevolve" alt="PyPI downloads"></a>
  <a href="https://github.com/codelion/openevolve/blob/main/LICENSE"><img src="https://img.shields.io/github/license/codelion/openevolve" alt="License"></a>
</p>

**Key Features:**

*   🎯 **Autonomous Discovery:** Transform LLMs into code optimizers, discovering novel algorithms without human guidance.
*   ⚡ **Proven Results:** Achieve significant speedups and state-of-the-art performance in real-world applications.
*   🔬 **Research Grade:** Leverage built-in reproducibility, rigorous evaluation pipelines, and advanced features.

[🚀 **Quick Start**](#-quick-start) | [📖 **Examples**](#-examples-gallery) | [🤔 **FAQ**](#-faq)

---

## 🚀 Why OpenEvolve?

OpenEvolve is the future of code optimization and algorithm discovery, pushing the boundaries of what's possible with AI-driven evolutionary coding.

| Feature             | Manual Optimization                               | OpenEvolve                                         |
| ------------------- | --------------------------------------------------- | -------------------------------------------------- |
| **Time to Solution** | Days to Weeks                                       | Hours                                               |
| **Exploration**     | Limited by Human Creativity                       | Unlimited LLM Creativity                             |
| **Reproducibility** | Difficult to Replicate                           | Fully Deterministic, Guaranteed Repeatability        |
| **Multi-Objective** | Complex Tradeoffs                                  | Automatic Pareto Optimization                       |
| **Scaling**         | Doesn't Scale                                       | Parallel Evolution Across Islands                    |

## 🏆 Proven Achievements

| Domain                    | Achievement                                        | Example                                                                          |
| ------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------- |
| GPU Optimization          | 2-3x Speedup on Apple Silicon                     | [MLX Metal Kernels](examples/mlx_metal_kernel_opt/)                              |
| Mathematical Optimization | State-of-the-Art Circle Packing (n=26)           | [Circle Packing](examples/circle_packing/)                                       |
| Algorithm Design          | Adaptive Sorting Algorithms                        | [Rust Adaptive Sort](examples/rust_adaptive_sort/)                               |
| Scientific Computing      | Automated Filter Design                            | [Signal Processing](examples/signal_processing/)                                 |
| Multi-Language Support    | Python, Rust, R, Metal Shaders                     | [All Examples](examples/)                                                        |

## 🚀 Quick Start: Get Started in 30 Seconds

1.  **Install OpenEvolve:**
    ```bash
    pip install openevolve
    ```
2.  **Set your LLM API Key:** (Works with any OpenAI-compatible provider)
    ```bash
    export OPENAI_API_KEY="your-api-key"
    ```
3.  **Run your first evolution:**
    ```python
    from openevolve import run_evolution
    result = run_evolution(
        'examples/function_minimization/initial_program.py',
        'examples/function_minimization/evaluator.py'
    )
    print(f'Best score: {result.best_score:.4f}')
    ```

For more control, use the CLI or Docker:

```bash
# CLI
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 1000

# Docker
docker run --rm -v $(pwd):/app ghcr.io/codelion/openevolve:latest \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py --iterations 100
```

## 🎬 See It In Action

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

**Before (Baseline):**
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

**After Evolution (2.8x faster):**
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

## 🧬 How OpenEvolve Works

OpenEvolve employs a sophisticated evolutionary coding pipeline, powered by cutting-edge technology:

![OpenEvolve Architecture](openevolve-architecture.png)

### 🎯 Core Innovation: MAP-Elites + LLMs

*   **Quality-Diversity Evolution:** Maintains a diverse population across various feature dimensions
*   **Island-Based Architecture:** Multiple populations prevent premature convergence
*   **LLM Ensemble:** Utilizes multiple models with intelligent fallback strategies
*   **Artifact Side-Channel:** Error feedback from execution enhances subsequent generations

### 🚀 Advanced Features

<details>
<summary><b>🔬 Scientific Reproducibility</b></summary>

*   **Comprehensive Seeding:** Every component (LLM, database, evaluation) is seeded
*   **Default Seed=42:** Results are reproducible out of the box
*   **Deterministic Evolution:** Runs can be exactly reproduced across all machines
*   **Component Isolation:** Hash-based isolation protects against cross-contamination

</details>

<details>
<summary><b>🤖 Advanced LLM Integration</b></summary>

*   **Test-Time Compute:** Integrate with [OptiLLM](https://github.com/codelion/optillm) for MoA and advanced reasoning
*   **Universal API:** Compatible with OpenAI, Google, and local models
*   **Plugin Ecosystem:** Supports OptiLLM plugins (readurls, executecode, z3_solver)
*   **Intelligent Ensembles:** Weighted combinations with sophisticated fallback

</details>

<details>
<summary><b>🧬 Evolution Algorithm Innovations</b></summary>

*   **Double Selection:** Different programs for performance vs inspiration
*   **Adaptive Feature Dimensions:** Custom quality-diversity metrics
*   **Migration Patterns:** Ring topology with controlled gene flow
*   **Multi-Strategy Sampling:** Elite, diverse, and exploratory selection

</details>

## 🎯 Perfect For

| Use Case               | Why OpenEvolve Excels                               |
| ---------------------- | --------------------------------------------------- |
| 🏃‍♂️ Performance Optimization | Discovers hardware-specific optimizations |
| 🧮 Algorithm Discovery      | Finds novel approaches to classical problems      |
| 🔬 Scientific Computing   | Automates tedious manual tuning processes       |
| 🎮 Competitive Programming | Generates multiple solution strategies           |
| 📊 Multi-Objective Problems | Pareto-optimal solutions across dimensions     |

## 🛠 Installation & Setup

### Requirements

*   **Python:** 3.9+
*   **LLM Access:** Any OpenAI-compatible API (e.g., OpenAI, Google, Local)
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

OpenEvolve is compatible with **any OpenAI-compatible API**:

<details>
<summary><b>🔥 OpenAI (Direct)</b></summary>

```bash
export OPENAI_API_KEY="sk-..."
# Uses OpenAI endpoints by default
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

## 📸 Examples Gallery

<div align="center">

### 🏆 Showcase Projects

| Project                                                 | Domain           | Achievement                      | Demo                                                                                                      |
| ------------------------------------------------------- | ---------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [🎯 **Function Minimization**](examples/function_minimization/) | Optimization     | Random → Simulated Annealing     | [View Results](examples/function_minimization/openevolve_output/)                                     |
| [⚡ **MLX GPU Kernels**](examples/mlx_metal_kernel_opt/)     | Hardware         | 2-3x Apple Silicon Speedup         | [Benchmarks](examples/mlx_metal_kernel_opt/README.md)                                                    |
| [🔄 **Rust Adaptive Sort**](examples/rust_adaptive_sort/)     | Algorithms       | Data-Aware Sorting              | [Code Evolution](examples/rust_adaptive_sort/)                                                        |
| [📐 **Symbolic Regression**](examples/symbolic_regression/)   | Science          | Automated Equation Discovery     | [LLM-SRBench](examples/symbolic_regression/)                                                             |
| [🕸️ **Web Scraper + OptiLLM**](examples/web_scraper_optillm/)   | AI Integration   | Test-Time Compute Optimization   | [Smart Scraping](examples/web_scraper_optillm/)                                                          |

</div>

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

<details>
<summary><b>🎨 Prompt Evolution</b></summary>

**Evolve prompts instead of code** for better LLM performance:

```yaml
# Example: HotpotQA dataset
Initial Prompt: "Answer the question based on the context."

Evolved Prompt: "As an expert analyst, carefully examine the provided context.
Break down complex multi-hop reasoning into clear steps. Cross-reference
information from multiple sources to ensure accuracy. Answer: [question]"

Result: +23% accuracy improvement on HotpotQA benchmark
```

[Full Example](examples/llm_prompt_optimization/)

</details>

<details>
<summary><b>🏁 Competitive Programming</b></summary>

**Automatic solution generation** for programming contests:

```python
# Problem: Find maximum subarray sum
# OpenEvolve discovers multiple approaches:

# Evolution Path 1: Brute Force → Kadane's Algorithm
# Evolution Path 2: Divide & Conquer → Optimized Kadane's
# Evolution Path 3: Dynamic Programming → Space-Optimized DP
```

[Online Judge Integration](examples/online_judge_programming/)

</details>

## ⚙️ Configuration

OpenEvolve provides extensive configuration options for advanced users.

```yaml
# Advanced Configuration Example
max_iterations: 1000
random_seed: 42  # Full reproducibility

llm:
  # Ensemble with test-time compute
  models:
    - name: "gemini-2.5-pro"
      weight: 0.6
    - name: "moa&readurls-o3"  # OptiLLM features
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

<details>
<summary><b>🎯 Feature Engineering</b></summary>

**Control how programs are organized in the quality-diversity grid:**

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

**Important:** Return raw values from the evaluator; OpenEvolve handles binning.

</details>

<details>
<summary><b>🎨 Custom Prompt Templates</b></summary>

**Advanced prompt engineering** with custom templates:

```yaml
prompt:
  template_dir: "custom_templates/"
  use_template_stochasticity: true
  template_variations:
    greeting:
      - "Let's enhance this code:"
      - "Time to optimize:"
      - "Improving the algorithm:"
```

See [prompt examples](examples/llm_prompt_optimization/templates/) for complete template customization.

</details>

## 🔧 Artifacts & Debugging

**Artifacts side-channel** delivers enriched feedback to accelerate evolution:

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

**The next generation prompt automatically includes:**

```markdown
## Previous Execution Feedback
⚠️ Warning: suboptimal memory access pattern
💡 LLM Feedback: Code is correct but could use better variable names
🔧 Build Warnings: unused variable x
```

This produces a **feedback loop**, allowing each generation to learn from prior errors!

## 📊 Visualization

**Real-time evolution tracking** with interactive web interface:

```bash
# Install visualization dependencies
pip install -r scripts/requirements.txt

# Launch interactive visualizer
python scripts/visualizer.py

# Or visualize specific checkpoint
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

**Features:**

*   🌳 **Evolution tree** with parent-child relationships
*   📈 **Performance tracking** across generations
*   🔍 **Code diff viewer** showing mutations
*   📊 **MAP-Elites grid** visualization
*   🎯 **Multi-metric analysis** with custom dimensions

![OpenEvolve Visualizer](openevolve-visualizer.png)

## 🚀 Roadmap

### 🔥 Upcoming Features

*   [ ] Multi-Modal Evolution: Incorporating images, audio, and text
*   [ ] Federated Learning: Distributed evolution across numerous machines
*   [ ] AutoML Integration: Integrating hyperparameter and architecture evolution
*   [ ] Benchmark Suite: Establish standardized evaluation across various domains

### 🌟 Research Directions

*   [ ] Self-Modifying Prompts: Evolution modifies its prompting strategy
*   [ ] Cross-Language Evolution: Python → Rust → C++ optimization chains
*   [ ] Neurosymbolic Reasoning: Combine neural and symbolic approaches
*   [ ] Human-AI Collaboration: Interactive evolution with human feedback

Want to contribute? Explore our [roadmap discussions](https://github.com/codelion/openevolve/discussions/categories/roadmap)!

## 🤔 FAQ

<details>
<summary><b>💰 How much does it cost to run?</b></summary>

**Cost varies depending on your LLM provider and iteration count:**

*   **o3:** ~$0.15-0.60 per iteration (dependent on code size)
*   **o3-mini:** ~$0.03-0.12 per iteration (cost-effective choice)
*   **Gemini-2.5-Pro:** ~$0.08-0.30 per iteration
*   **Gemini-2.5-Flash:** ~$0.01-0.05 per iteration (quickest and most affordable)
*   **Local models:** Basically free after setup
*   **OptiLLM:** Benefit from using cheaper models alongside test-time compute for enhanced results

**Cost-saving strategies:**

*   Start with a smaller number of iterations (100-200)
*   Choose o3-mini, Gemini-2.5-Flash, or local models for exploring
*   Employ cascade evaluation to filter substandard programs early
*   Configure smaller population sizes initially

</details>

<details>
<summary><b>🆚 How does this compare to manual optimization?</b></summary>

| Aspect                   | Manual                                         | OpenEvolve                                      |
| ------------------------ | ---------------------------------------------- | ----------------------------------------------- |
| **Initial Learning**     | Weeks to understand domain                     | Minutes to start                                |
| **Solution Quality**     | Dependent on Expertise                         | Consistently explores novel approaches          |
| **Time Investment**      | Days-Weeks per optimization                    | Hours for comprehensive evolution               |
| **Reproducibility**      | Difficult to replicate precise process        | Perfect reproduction with seeds                  |
| **Scaling**              | Limited by human capacity                      | Parallel evolution across islands               |

**OpenEvolve excels** when you must explore extensive solution spaces or optimize for several objectives concurrently.

</details>

<details>
<summary><b>🔧 Can I use my own LLM?</b></summary>

**Yes!** OpenEvolve supports any OpenAI-compatible API:

*   **Commercial:** OpenAI, Google, Cohere
*   **Local:** Ollama, vLLM, LM Studio, text-generation-webui
*   **Advanced:** OptiLLM for routing and test-time compute

Simply specify the `api_base` in your configuration to target your endpoint.

</details>

<details>
<summary><b>🚨 What if evolution gets stuck?</b></summary>

**Built-in mechanisms to prevent stagnation:**

*   **Island migration:** Introduce novel genes from different populations
*   **Temperature control:** Manage the balance between exploration and exploitation
*   **Diversity maintenance:** MAP-Elites prevents convergence
*   **Artifact feedback:** Error messages guide improvements
*   **Template stochasticity:** Randomized prompts help disrupt patterns

**Manual interventions:**

*   Increase `num_diverse_programs` for expanded exploration
*   Add custom feature dimensions to diversify search
*   Utilize template variations to randomize prompts
*   Adjust migration intervals to promote cross-pollination

</details>

<details>
<summary><b>📈 How do I measure success?</b></summary>

**Numerous success metrics:**

1.  **Primary Metric:** Your evaluator's `combined_score` or the average metric
2.  **Convergence:** Improvement in the best score over time
3.  **Diversity:** MAP-Elites grid coverage
4.  **Efficiency:** Iterations required to achieve the target performance
5.  **Robustness:** Performance across various test cases

**Utilize the visualizer** to monitor all metrics in real-time, and to recognize when evolution has converged.

</details>

### 🌟 Contributors

We're grateful to our amazing contributors who make OpenEvolve a reality!

<a href="https://github.com/codelion/openevolve/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=codelion/openevolve" />
</a>

### 🤝 Contributing

Contributions are very welcome! Here's how to get started:

1.  🍴 **Fork** the repository
2.  🌿 **Create** your feature branch: `git checkout -b feat-amazing-feature`
3.  ✨ **Add** your changes and tests
4.  ✅ **Test** everything: `python -m unittest discover tests`
5.  📝 **Commit** with a clear message
6.  🚀 **Push** and create a Pull Request

**New to open source?** Check out our [Contributing Guide](CONTRIBUTING.md) and look for [`good-first-issue`](https://github.com/codelion/openevolve/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) labels!

### 📚 Academic & Research

**Articles & Blog Posts About OpenEvolve:**

*   [Towards Open Evolutionary Agents](https://huggingface.co/blog/driaforall/towards-open-evolutionary-agents) - Evolution of coding agents and the open-source movement
*   [OpenEvolve: GPU Kernel Discovery](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery) - Automated discovery of optimized GPU kernels with 2-3x speedups
*   [OpenEvolve: Evolutionary Coding with LLMs](https://huggingface.co/blog/codelion/openevolve) - Introduction to evolutionary algorithm discovery using large language models

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

### 🚀 Ready to Evolve?

**Created with ❤️ by the OpenEvolve community**

*Star ⭐ this repository if OpenEvolve helps you discover breakthrough algorithms!*

</div>