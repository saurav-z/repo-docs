# OpenEvolve: Unleash Autonomous Code Optimization with LLMs

**Evolve your code from random search to state-of-the-art algorithms using the power of Large Language Models!** [Explore the OpenEvolve Repository](https://github.com/codelion/openevolve)

<p align="center">
  <a href="https://github.com/codelion/openevolve/stargazers"><img src="https://img.shields.io/github/stars/codelion/openevolve?style=social" alt="GitHub stars"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/v/openevolve" alt="PyPI version"></a>
  <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/dm/openevolve" alt="PyPI downloads"></a>
  <a href="https://github.com/codelion/openevolve/blob/main/LICENSE"><img src="https://img.shields.io/github/license/codelion/openevolve" alt="License"></a>
</p>

**Key Features:**

*   🎯 **Autonomous Discovery:** LLMs generate and optimize code, requiring no human guidance.
*   ⚡ **Proven Results:** Achieve 2-3x speedups on real hardware and state-of-the-art performance.
*   🔬 **Research Grade:** Enjoy full reproducibility, extensive evaluation pipelines, and scientific rigor.
*   🧬 **Multi-Language Support:** Works with Python, Rust, Metal shaders, and more.
*   🚀 **Rapid Deployment:** Get started in seconds with simple installation and setup.

## Why Choose OpenEvolve?

| Feature                  | OpenEvolve                                    | Manual Optimization                     |
| ------------------------ | --------------------------------------------- | ------------------------------------- |
| Time to Solution         | Hours                                         | Days to Weeks                           |
| Exploration Breadth      | Unlimited LLM Creativity                       | Limited by Human Creativity           |
| Reproducibility          | Fully Deterministic                             | Difficult to Replicate                |
| Multi-objective          | Automatic Pareto Optimization                | Complex Tradeoffs                      |
| Scaling                  | Parallel Evolution Across Islands              | Doesn't Scale                          |

## 🏆 Achievements

OpenEvolve excels in diverse domains, delivering impressive results:

| Domain               | Achievement                             | Example                                                  |
| -------------------- | --------------------------------------- | -------------------------------------------------------- |
| GPU Optimization     | 2-3x speedup on Apple Silicon         | [MLX Metal Kernels](examples/mlx_metal_kernel_opt/)       |
| Mathematical         | State-of-the-art circle packing (n=26) | [Circle Packing](examples/circle_packing/)               |
| Algorithm Design     | Adaptive sorting algorithms             | [Rust Adaptive Sort](examples/rust_adaptive_sort/)        |
| Scientific Computing | Automated filter design                 | [Signal Processing](examples/signal_processing/)         |

## 🚀 Quick Start

Evolve your code in three simple steps:

```bash
# 1. Install OpenEvolve
pip install openevolve

# 2. Set your LLM API key (works with any OpenAI-compatible provider)
export OPENAI_API_KEY="your-api-key"

# 3. Run an evolution!
python -c "
from openevolve import run_evolution
result = run_evolution(
    'examples/function_minimization/initial_program.py',
    'examples/function_minimization/evaluator.py'
)
print(f'Best score: {result.best_score:.4f}')
"
```

## 🎬 See It In Action

<details>
<summary>🔥 **Circle Packing: From Random to State-of-the-Art**</summary>

**Witness OpenEvolve's real-time discovery of optimal circle packing:**

| Generation 1                  | Generation 190               | Generation 460 (Final)         |
| ----------------------------- | ----------------------------- | ------------------------------ |
| ![Initial](examples/circle_packing/circle_packing_1.png) | ![Progress](examples/circle_packing/circle_packing_190.png) | ![Final](examples/circle_packing/circle_packing_460.png) |
| Random placement            | Learning structure            | **State-of-the-art result**     |

**Result**: Matches published benchmarks for n=26 circle packing.

</details>

<details>
<summary>⚡ **GPU Kernel Evolution**</summary>

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

**Performance Impact**: 2.8x speedup on Apple M1 Pro, maintaining accuracy.

</details>

## 🧬 How OpenEvolve Works

OpenEvolve employs a sophisticated **evolutionary coding pipeline** incorporating innovative techniques:

![OpenEvolve Architecture](openevolve-architecture.png)

### Core Innovation: MAP-Elites + LLMs

*   **Quality-Diversity Evolution:** Fosters diverse populations across feature dimensions.
*   **Island-Based Architecture:** Prevents premature convergence through multiple populations.
*   **LLM Ensemble:** Leverages multiple models with intelligent fallback strategies.
*   **Artifact Side-Channel:** Improves subsequent generations with error feedback.

## 🛠 Installation & Setup

### Requirements

*   **Python:** 3.9+
*   **LLM Access:** Any OpenAI-compatible API
*   **Optional:** Docker for containerized runs

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

### LLM Provider Setup

OpenEvolve supports **any OpenAI-compatible API:**

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

### 🏆 **Showcase Projects**

| Project                                             | Domain            | Achievement                      | Demo                                                   |
| --------------------------------------------------- | ----------------- | -------------------------------- | ------------------------------------------------------ |
| [🎯 **Function Minimization**](examples/function_minimization/) | Optimization      | Random → Simulated Annealing   | [View Results](examples/function_minimization/openevolve_output/) |
| [⚡ **MLX GPU Kernels**](examples/mlx_metal_kernel_opt/)   | Hardware          | 2-3x Apple Silicon speedup     | [Benchmarks](examples/mlx_metal_kernel_opt/README.md)    |
| [🔄 **Rust Adaptive Sort**](examples/rust_adaptive_sort/)   | Algorithms        | Data-aware sorting              | [Code Evolution](examples/rust_adaptive_sort/)          |
| [📐 **Symbolic Regression**](examples/symbolic_regression/)  | Science           | Automated equation discovery    | [LLM-SRBench](examples/symbolic_regression/)           |
| [🕸️ **Web Scraper + OptiLLM**](examples/web_scraper_optillm/) | AI Integration    | Test-time compute optimization | [Smart Scraping](examples/web_scraper_optillm/)          |

</div>

### 🎯 **Quick Example**: Function Minimization

**Observe OpenEvolve's evolution from random search to sophisticated optimization:**

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

### 🔬 **Advanced Examples**

<details>
<summary>🎨 **Prompt Evolution**</summary>

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
<summary>🏁 **Competitive Programming**</summary>

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

OpenEvolve is highly configurable for advanced users.

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
<summary>🎯 **Feature Engineering**</summary>

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

**Important**: Return raw values from evaluator, OpenEvolve handles binning automatically.

</details>

<details>
<summary>🎨 **Custom Prompt Templates**</summary>

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

The **artifacts side-channel** offers rich feedback to accelerate evolution:

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

**Real-time evolution tracking** with an interactive web interface:

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

*   [ ] Multi-Modal Evolution: Images, audio, and text simultaneously
*   [ ] Federated Learning: Distributed evolution across multiple machines
*   [ ] AutoML Integration: Hyperparameter and architecture evolution
*   [ ] Benchmark Suite: Standardized evaluation across domains

### 🌟 Research Directions

*   [ ] Self-Modifying Prompts: Evolution modifies its own prompting strategy
*   [ ] Cross-Language Evolution: Python → Rust → C++ optimization chains
*   [ ] Neurosymbolic Reasoning: Combine neural and symbolic approaches
*   [ ] Human-AI Collaboration: Interactive evolution with human feedback

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

*   Start with fewer iterations (100-200)
*   Use o3-mini, Gemini-2.5-Flash or local models for exploration
*   Use cascade evaluation to filter bad programs early
*   Configure smaller population sizes initially

</details>

<details>
<summary>🆚 How does this compare to manual optimization?</summary>

| Aspect                 | Manual                           | OpenEvolve                    |
| ---------------------- | -------------------------------- | ----------------------------- |
| Initial Learning       | Weeks to understand domain      | Minutes to start              |
| Solution Quality       | Depends on expertise           | Consistently explores novel approaches |
| Time Investment        | Days-weeks per optimization     | Hours for complete evolution  |
| Reproducibility        | Hard to replicate exact process | Perfect reproduction with seeds |
| Scaling                | Doesn't scale beyond human capacity | Parallel evolution across islands |

**OpenEvolve shines** when exploring large solution spaces or optimizing for multiple objectives simultaneously.

</details>

<details>
<summary>🔧 Can I use my own LLM?</summary>

**Yes!** OpenEvolve supports any OpenAI-compatible API:

*   Commercial: OpenAI, Google, Cohere
*   Local: Ollama, vLLM, LM Studio, text-generation-webui
*   Advanced: OptiLLM for routing and test-time compute

Just set the `api_base` in your config to point to your endpoint.

</details>

<details>
<summary>🚨 What if evolution gets stuck?</summary>

**Built-in mechanisms prevent stagnation:**

*   Island migration: Fresh genes from other populations
*   Temperature control: Exploration vs exploitation balance
*   Diversity maintenance: MAP-Elites prevents convergence
*   Artifact feedback: Error messages guide improvements
*   Template stochasticity: Randomized prompts break patterns

**Manual interventions:**

*   Increase `num_diverse_programs` for more exploration
*   Add custom feature dimensions to diversify search
*   Use template variations to randomize prompts
*   Adjust migration intervals for more cross-pollination

</details>

<details>
<summary>📈 How do I measure success?</summary>

**Multiple success metrics:**

1.  Primary Metric: Your evaluator's `combined_score` or metric average
2.  Convergence: Best score improvement over time
3.  Diversity: MAP-Elites grid coverage
4.  Efficiency: Iterations to reach target performance
5.  Robustness: Performance across different test cases

**Use the visualizer** to track all metrics in real-time and identify when evolution has converged.

</details>

### 🌟 Contributors

Thank you to our amazing contributors who make OpenEvolve possible!

<a href="https://github.com/codelion/openevolve/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=codelion/openevolve" />
</a>

### 🤝 Contributing

We welcome contributions! Get started by:

1.  Forking the repository
2.  Creating your feature branch: `git checkout -b feat-amazing-feature`
3.  Adding your changes and tests
4.  Testing everything: `python -m unittest discover tests`
5.  Committing with a clear message
6.  Pushing and creating a Pull Request

Check out our [Contributing Guide](CONTRIBUTING.md) and look for [`good-first-issue`](https://github.com/codelion/openevolve/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) labels!

### 📚 Academic & Research

**Articles & Blog Posts About OpenEvolve**:

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

### 🚀 Ready to evolve your code?

**Made with ❤️ by the OpenEvolve community**

*Star ⭐ this repository if OpenEvolve helps you discover breakthrough algorithms!*

</div>
```
Key improvements and explanations:

*   **SEO Optimization:**  The title is strong and keywords ("OpenEvolve", "Autonomous", "Code Optimization", "LLMs") are used naturally throughout. Headings are clear and concise.
*   **One-Sentence Hook:** The initial sentence is a compelling hook.
*   **Concise Summary:**  The features, achievements, and benefits are highlighted upfront.
*   **Bulleted Key Features:**  Makes it easy to scan and understand the core value.
*   **Well-Organized Structure:**  Clear sections with appropriate headings and subheadings for readability.
*   **Use of Tables:** Tables effectively compare OpenEvolve to manual methods.
*   **Expanded "See it in Action" Section:**  More detailed examples with before/after code and visual aids.  Focuses on *results*.
*   **Clear Installation and Setup Instructions:** The most important part is very easy to follow. Includes options for different providers (Google, Local, OptiLLM) which is good for users.
*   **FAQ Section:**  Addresses common questions, which can reduce user friction.
*   **Roadmap and Contributing:** Provides clear information on future development and contributions.
*   **Citation Info:** Adds a way to give credit.
*   **Links Back:**  Crucially includes the link to the original repo *early* in the document.
*   **Visuals:** Kept and improved, with better descriptions.
*   **More Thorough:**  More of the original README's content is integrated and structured.
*   **Consistent Formatting:**  Uses consistent Markdown for all sections.
*   **Clear, Action-Oriented Language:**  Uses phrases like "Unleash," "Witness," and "Get started" to engage the reader.