# OpenEvolve: Unleash AI-Powered Code Evolution

**Discover breakthrough algorithms and accelerate your code with the power of AI.  OpenEvolve transforms Large Language Models (LLMs) into autonomous coding agents, automating optimization and algorithm discovery.** Explore the original repository: [https://github.com/codelion/openevolve](https://github.com/codelion/openevolve)

[![GitHub stars](https://img.shields.io/github/stars/codelion/openevolve?style=social)](https://github.com/codelion/openevolve/stargazers)
[![PyPI version](https://img.shields.io/pypi/v/openevolve)](https://pypi.org/project/openevolve/)
[![PyPI downloads](https://img.shields.io/pypi/dm/openevolve)](https://pypi.org/project/openevolve/)
[![License](https://img.shields.io/github/license/codelion/openevolve)](https://github.com/codelion/openevolve/blob/main/LICENSE)

[🚀 **Quick Start**](#-quick-start) | [✨ **Key Features**](#-key-features) | [📖 **Examples**](#-examples-gallery) | [🛠️ **Installation**](#-installation--setup)

---

## ✨ Key Features

*   **Autonomous Discovery:**  LLMs evolve and discover entirely new algorithms without human intervention.
*   **Proven Results:** Achieve **2-3x speedups** on real hardware and discover **state-of-the-art** solutions.
*   **Research-Grade Reproducibility:**  Benefit from deterministic runs, comprehensive evaluation, and advanced features for scientific rigor.

    *   **Speed & Efficiency:** Outperform manual optimization, delivering solutions in hours compared to days or weeks.
    *   **Unleash Creativity:**  Tap into the unlimited creative potential of LLMs to explore broader solution spaces.
    *   **Deterministic Outcomes:**  Guarantee consistent and reproducible results for reliable research.
    *   **Multi-Objective Optimization:**  Enable automated Pareto optimization for optimal results across multiple objectives.
    *   **Parallel Evolution:**  Scale optimization efforts with an island-based architecture for efficient, concurrent code evolution.

## 🏆 OpenEvolve's Achievements

| Domain                 | Achievement                                    | Example                                                                 |
| :--------------------- | :--------------------------------------------- | :---------------------------------------------------------------------- |
| **GPU Optimization**   | 2-3x speedup on Apple Silicon                | [MLX Metal Kernels](examples/mlx_metal_kernel_opt/)                        |
| **Mathematical**       | State-of-the-art circle packing (n=26)          | [Circle Packing](examples/circle_packing/)                               |
| **Algorithm Design**   | Adaptive sorting algorithms                     | [Rust Adaptive Sort](examples/rust_adaptive_sort/)                         |
| **Scientific Computing** | Automated filter design                        | [Signal Processing](examples/signal_processing/)                           |
| **Multi-Language**      | Python, Rust, R, Metal shaders optimization | [All Examples](examples/)                                                |

## 🚀 Quick Start

Get started with code evolution in under a minute:

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

**For Advanced Control:** Use the CLI:

```bash
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 1000
```

**Docker Integration:**

```bash
docker run --rm -v $(pwd):/app ghcr.io/codelion/openevolve:latest \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py --iterations 100
```

## 🎬 See It in Action

<details>
<summary><b>🔥 Circle Packing: From Random to State-of-the-Art</b></summary>

**Watch OpenEvolve discover optimal circle packing in real-time:**

| Generation 1                                                                       | Generation 190                                                                     | Generation 460 (Final)                                                                 |
| :--------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- |
| <img src="examples/circle_packing/circle_packing_1.png" alt="Generation 1" width="200"> | <img src="examples/circle_packing/circle_packing_190.png" alt="Generation 190" width="200"> | <img src="examples/circle_packing/circle_packing_460.png" alt="Generation 460" width="200"> |
| Random placement                                                                   | Learning structure                                                                  | **State-of-the-art result**                                                            |

**Result**: Achieves published benchmarks for n=26 circle packing problem.

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

**Performance Impact:** Achieves a 2.8x speedup on Apple M1 Pro while maintaining numerical accuracy.

</details>

## 🧬 How OpenEvolve Works

OpenEvolve uses a sophisticated **evolutionary coding pipeline**:

<img src="openevolve-architecture.png" alt="OpenEvolve Architecture" width="700">

### 🎯 Core Innovations: MAP-Elites + LLMs

*   **Quality-Diversity Evolution:** Maintains diverse populations.
*   **Island-Based Architecture:** Prevents premature convergence through multiple populations.
*   **LLM Ensemble:** Employs multiple models with smart fallback.
*   **Artifact Side-Channel:** Feedback improves generations.

### 🚀 Advanced Features

<details>
<summary><b>🔬 Scientific Reproducibility</b></summary>

*   **Comprehensive Seeding:** Every component (LLM, database, evaluation) is seeded.
*   **Default Seed=42:** Provides immediate reproducible results.
*   **Deterministic Evolution:** Ensures exact run reproduction.
*   **Component Isolation:** Prevents cross-contamination with hash-based isolation.

</details>

<details>
<summary><b>🤖 Advanced LLM Integration</b></summary>

*   **Test-Time Compute:** Integration with [OptiLLM](https://github.com/codelion/optillm) for MoA and enhanced reasoning
*   **Universal API:** Compatibility with OpenAI, Google, and local models.
*   **Plugin Ecosystem:** Supports OptiLLM plugins (readurls, executecode, z3_solver).
*   **Intelligent Ensembles:** Utilizes weighted combinations with sophisticated fallback.

</details>

<details>
<summary><b>🧬 Evolution Algorithm Innovations</b></summary>

*   **Double Selection:** Different programs for performance vs. inspiration.
*   **Adaptive Feature Dimensions:** Custom quality-diversity metrics.
*   **Migration Patterns:** Ring topology with controlled gene flow.
*   **Multi-Strategy Sampling:** Elite, diverse, and exploratory selection.

</details>

## 🎯 Perfect For

| Use Case                    | Why OpenEvolve Excels                                      |
| :-------------------------- | :--------------------------------------------------------- |
| 🏃‍♂️ **Performance Optimization** | Discovers hardware-specific optimizations humans miss. |
| 🧮 **Algorithm Discovery**      | Finds novel approaches to classic problems.                |
| 🔬 **Scientific Computing**   | Automates manual tuning processes.                         |
| 🎮 **Competitive Programming** | Generates multiple solution strategies.                   |
| 📊 **Multi-Objective Problems** | Pareto-optimal solutions across multiple dimensions.      |

## 🛠️ Installation & Setup

### Requirements

*   **Python**: 3.9+
*   **LLM Access**: Compatible with any OpenAI-compatible API.
*   **Optional**: Docker for containerized runs.

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

OpenEvolve works with **any OpenAI-compatible API**:

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

| Project                                                                 | Domain            | Achievement                            | Demo                                                               |
| :---------------------------------------------------------------------- | :---------------- | :------------------------------------- | :----------------------------------------------------------------- |
| [🎯 Function Minimization](examples/function_minimization/)              | Optimization      | Random → Simulated Annealing           | [View Results](examples/function_minimization/openevolve_output/)  |
| [⚡ MLX GPU Kernels](examples/mlx_metal_kernel_opt/)                   | Hardware          | 2-3x Apple Silicon speedup           | [Benchmarks](examples/mlx_metal_kernel_opt/README.md)            |
| [🔄 Rust Adaptive Sort](examples/rust_adaptive_sort/)                   | Algorithms        | Data-aware sorting                      | [Code Evolution](examples/rust_adaptive_sort/)                      |
| [📐 Symbolic Regression](examples/symbolic_regression/)                  | Science           | Automated equation discovery           | [LLM-SRBench](examples/symbolic_regression/)                       |
| [🕸️ Web Scraper + OptiLLM](examples/web_scraper_optillm/)               | AI Integration    | Test-time compute optimization          | [Smart Scraping](examples/web_scraper_optillm/)                     |

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
<summary>🎨 Prompt Evolution</summary>

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
<summary>🏁 Competitive Programming</summary>

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

OpenEvolve provides extensive configuration for advanced users:

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
<summary>🎯 Feature Engineering</summary>

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

**Important**: Return raw values from the evaluator; OpenEvolve handles binning automatically.

</details>

<details>
<summary>🎨 Custom Prompt Templates</summary>

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

See [prompt examples](examples/llm_prompt_optimization/templates/) for full customization.

</details>

## 🔧 Artifacts & Debugging

**Artifacts side-channel** for enhanced evolution feedback:

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

*   🌳 **Evolution tree** with parent-child relationships.
*   📈 **Performance tracking** across generations.
*   🔍 **Code diff viewer** showing mutations.
*   📊 **MAP-Elites grid** visualization.
*   🎯 **Multi-metric analysis** with custom dimensions.

![OpenEvolve Visualizer](openevolve-visualizer.png)

## 🚀 Roadmap

### 🔥 Upcoming Features

*   [ ] **Multi-Modal Evolution**:  Simultaneous image, audio, and text evolution.
*   [ ] **Federated Learning**: Distributed evolution across multiple machines.
*   [ ] **AutoML Integration**: Hyperparameter and architecture evolution.
*   [ ] **Benchmark Suite**: Standardized evaluation across domains.

### 🌟 Research Directions

*   [ ] **Self-Modifying Prompts**: Evolution modifies its prompting strategy.
*   [ ] **Cross-Language Evolution**: Python → Rust → C++ optimization chains.
*   [ ] **Neurosymbolic Reasoning**: Combines neural and symbolic approaches.
*   [ ] **Human-AI Collaboration**: Interactive evolution with human feedback.

Want to contribute? Explore our [roadmap discussions](https://github.com/codelion/openevolve/discussions/categories/roadmap)!

## 🤔 FAQ

<details>
<summary>💰 How much does it cost to run?</summary>

**Cost depends on your LLM provider and iterations:**

*   **o3**: ~$0.15-0.60 per iteration (depending on code size)
*   **o3-mini**: ~$0.03-0.12 per iteration (more cost-effective)
*   **Gemini-2.5-Pro**: ~$0.08-0.30 per iteration
*   **Gemini-2.5-Flash**: ~$0.01-0.05 per iteration (fastest and cheapest)
*   **Local models**: Nearly free after setup
*   **OptiLLM**: Use cheaper models with test-time compute for better results

**Cost-saving tips:**

*   Start with fewer iterations (100-200).
*   Use o3-mini, Gemini-2.5-Flash, or local models for exploration.
*   Use cascade evaluation to filter bad programs early.
*   Configure smaller population sizes initially.

</details>

<details>
<summary>🆚 How does this compare to manual optimization?</summary>

| Aspect              | Manual                         | OpenEvolve                                     |
| :------------------ | :----------------------------- | :--------------------------------------------- |
| **Initial Learning** | Weeks to understand domain     | Minutes to start                               |
| **Solution Quality** | Depends on expertise          | Consistently explores novel approaches        |
| **Time Investment** | Days-weeks per optimization     | Hours for complete evolution                 |
| **Reproducibility** | Hard to replicate exact process | Perfect reproduction with seeds                |
| **Scaling**         | Doesn't scale beyond human capacity | Parallel evolution across islands              |

**OpenEvolve shines** when you need to explore large solution spaces or optimize for multiple objectives simultaneously.

</details>

<details>
<summary>🔧 Can I use my own LLM?</summary>

**Yes!** OpenEvolve supports any OpenAI-compatible API:

*   **Commercial**: OpenAI, Google, Cohere
*   **Local**: Ollama, vLLM, LM Studio, text-generation-webui
*   **Advanced**: OptiLLM for routing and test-time compute

Just set the `api_base` in your config to point to your endpoint.

</details>

<details>
<summary>🚨 What if evolution gets stuck?</summary>

**Built-in mechanisms prevent stagnation:**

*   **Island migration**: Fresh genes from other populations.
*   **Temperature control**: Exploration vs. exploitation balance.
*   **Diversity maintenance**: MAP-Elites prevents convergence.
*   **Artifact feedback**: Error messages guide improvements.
*   **Template stochasticity**: Randomized prompts break patterns.

**Manual interventions:**

*   Increase `num_diverse_programs` for more exploration.
*   Add custom feature dimensions to diversify the search.
*   Use template variations to randomize prompts.
*   Adjust migration intervals for more cross-pollination.

</details>

<details>
<summary>📈 How do I measure success?</summary>

**Multiple success metrics:**

1.  **Primary Metric**: Your evaluator's `combined_score` or metric average.
2.  **Convergence**: Best score improvement over time.
3.  **Diversity**: MAP-Elites grid coverage.
4.  **Efficiency**: Iterations to reach target performance.
5.  **Robustness**: Performance across different test cases.

**Use the visualizer** to track all metrics in real-time and identify when evolution has converged.

</details>

### 🌟 Contributors

Thank you to the amazing contributors who make OpenEvolve possible!

<a href="https://github.com/codelion/openevolve/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=codelion/openevolve" alt="Contributors">
</a>

### 🤝 Contributing

We welcome contributions!  Here's how to get started:

1.  🍴 **Fork** the repository.
2.  🌿 **Create** your feature branch: `git checkout -b feat-amazing-feature`.
3.  ✨ **Add** your changes and tests.
4.  ✅ **Test** everything: `python -m unittest discover tests`.
5.  📝 **Commit** with a clear message.
6.  🚀 **Push** and create a Pull Request.

**New to open source?** Check out our [Contributing Guide](CONTRIBUTING.md) and look for [`good-first-issue`](https://github.com/codelion/openevolve/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) labels!

### 📚 Academic & Research

**Articles & Blog Posts About OpenEvolve**:

*   [Towards Open Evolutionary Agents](https://huggingface.co/blog/driaforall/towards-open-evolutionary-agents) - Evolution of coding agents and the open-source movement.
*   [OpenEvolve: GPU Kernel Discovery](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery) - Automated discovery of optimized GPU kernels with 2-3x speedups.
*   [OpenEvolve: Evolutionary Coding with LLMs](https://huggingface.co/blog/codelion/openevolve) - Introduction to evolutionary algorithm discovery using large language models.

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

### 🚀 Ready to evolve your code?

**Made with ❤️ by the OpenEvolve community**

*Star ⭐ this repository if OpenEvolve helps you discover breakthrough algorithms!*

</div>
```
Key improvements and SEO considerations:

*   **Concise Hook:** The first sentence is a clear, impactful hook, using keywords.
*   **Keyword Rich Headings:**  Headings incorporate relevant keywords (e.g., "AI-Powered Code Evolution," "Key Features").
*   **Key Feature Bullets:**  Uses bullet points to make the key features very scannable.
*   **Quantifiable Achievements:** Highlights specific speedups, matches to benchmarks, and other results (crucial for credibility and SEO).
*   **Benefit-Driven Descriptions:**  Focuses on the *benefits* of using OpenEvolve (e.g., "achieve speedups," "discover novel approaches").
*   **Clear Calls to Action:**  Uses visual cues (emojis) and a call to action ("Ready to evolve your code?") to encourage engagement.
*   **Well-Structured Code Blocks:** Code blocks are clearly formatted and marked for easy readability (important for discoverability and engagement).
*   **Internal Linking:** Links within the README to relevant sections (Quick Start, Examples, Installation) to guide users and improve navigation.
*   **Strong Visuals:**  Includes images, including a detailed architecture diagram.
*   **FAQ and Roadmap:**  Addresses common questions and future plans to provide useful information and establish future-proofing.
*   **Contributors Section:** Acknowledges contributors (good for community and SEO).
*   **Citation:**  Includes a citation to promote proper usage and research (if used in research).
*   **Mobile-Friendly/Readability:** The use of markdown tables, details and summaries are great for display across devices.

This revised README is more engaging, easier to read, and optimized to attract users, improve search ranking, and highlight the value of OpenEvolve.