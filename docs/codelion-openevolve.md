# OpenEvolve: Evolve Your Code with AI 🧬

**Unleash the power of AI to automate code optimization and discover novel algorithms with OpenEvolve, the leading open-source evolutionary coding agent.  [Explore the OpenEvolve Repository](https://github.com/codelion/openevolve) to get started!**

[![GitHub stars](https://img.shields.io/github/stars/codelion/openevolve?style=social)](https://github.com/codelion/openevolve/stargazers)
[![PyPI version](https://img.shields.io/pypi/v/openevolve)](https://pypi.org/project/openevolve/)
[![PyPI downloads](https://img.shields.io/pypi/dm/openevolve)](https://pypi.org/project/openevolve/)
[![License](https://img.shields.io/github/license/codelion/openevolve)](https://github.com/codelion/openevolve/blob/main/LICENSE)

---

## ✨ Key Features of OpenEvolve

*   **Autonomous Discovery:** Leverage Large Language Models (LLMs) to go beyond optimization and discover entirely new, cutting-edge algorithms with no human intervention.
*   **Proven Results:** Achieve significant performance improvements, including 2-3x speedups on real hardware and state-of-the-art results in areas like circle packing.
*   **Research-Grade Rigor:** Benefit from a fully reproducible and deterministic environment, extensive evaluation pipelines, and built-in scientific methodologies.

## 🚀 Why Choose OpenEvolve?

| Feature            | Manual Optimization                  | OpenEvolve                             |
| ------------------ | ------------------------------------ | -------------------------------------- |
| **Time to Solution** | Days to Weeks                        | Hours                                  |
| **Exploration**    | Limited by Human Creativity       | Unlimited, LLM-Powered                  |
| **Reproducibility** | Difficult                          | Fully Deterministic                  |
| **Multi-objective** | Complex Tradeoffs                    | Automatic Pareto Optimization           |
| **Scalability**    | Doesn't Scale                       | Parallel Evolution Across Islands      |

## 🏆 Proven Achievements & Applications

| Domain                 | Achievement                               | Example                                                                 |
| ---------------------- | ----------------------------------------- | ----------------------------------------------------------------------- |
| **GPU Optimization**   | 2-3x Speedup on Apple Silicon             | [MLX Metal Kernels](examples/mlx_metal_kernel_opt/)                     |
| **Mathematics**        | State-of-the-Art Circle Packing (n=26)    | [Circle Packing](examples/circle_packing/)                             |
| **Algorithm Design**   | Adaptive Sorting Algorithms               | [Rust Adaptive Sort](examples/rust_adaptive_sort/)                       |
| **Scientific Computing** | Automated Filter Design                  | [Signal Processing](examples/signal_processing/)                       |
| **Multi-Language**     | Python, Rust, R, Metal Shaders Support     | [All Examples](examples/)                                             |
| **Competitive Programming** | Automatic Solution Generation | [Online Judge Integration](examples/online_judge_programming/) |

## 🚀 Quick Start: Evolve Your First Program in Seconds

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

## 📚  Library Usage and Advanced Capabilities

OpenEvolve offers versatile usage options, including as a library or using the command-line interface.  It features support for evolving functions directly and provides complete control over the evolutionary process with extensive configuration.

```python
from openevolve import run_evolution, evolve_function

# Evolution with inline code (no files needed!)
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

**Command Line Interface**
```bash
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 1000
```

**Docker**
```bash
docker run --rm -v $(pwd):/app ghcr.io/codelion/openevolve:latest \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py --iterations 100
```

## 🎬 See OpenEvolve in Action

<details>
<summary><b>🔥 Circle Packing Evolution</b></summary>

**Watch the evolution from random placement to state-of-the-art circle packing in real-time:**

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

## 🧬 How OpenEvolve Works: Core Innovation

OpenEvolve employs a cutting-edge **evolutionary coding pipeline** that utilizes a combination of:

*   **MAP-Elites + LLMs:** Quality-Diversity Evolution for diverse populations.
*   **Island-Based Architecture:** Parallel evolution preventing premature convergence.
*   **LLM Ensemble:** Multiple models with intelligent fallback strategies.
*   **Artifact Side-Channel:**  Error feedback to improve subsequent generations.

![OpenEvolve Architecture](openevolve-architecture.png)

## 🔬 Advanced Features for Research and Development

<details>
<summary><b>🔬 Scientific Reproducibility</b></summary>

*   **Comprehensive Seeding:** Consistent results.
*   **Default Seed=42:** Reproducibility out-of-the-box.
*   **Deterministic Evolution:** Exact results across machines.
*   **Component Isolation:** Prevents cross-contamination.

</details>

<details>
<summary><b>🤖 Advanced LLM Integration</b></summary>

*   **Test-Time Compute:** Integration with [OptiLLM](https://github.com/codelion/optillm) for enhanced reasoning.
*   **Universal API:** Support for OpenAI, Google, and local models.
*   **Plugin Ecosystem:** OptiLLM plugin support for various functionalities.
*   **Intelligent Ensembles:** Weighted combinations with sophisticated fallback strategies.

</details>

<details>
<summary><b>🧬 Evolution Algorithm Innovations</b></summary>

*   **Double Selection:** Programs for performance and inspiration.
*   **Adaptive Feature Dimensions:** Custom quality-diversity metrics.
*   **Migration Patterns:** Ring topology with controlled gene flow.
*   **Multi-Strategy Sampling:** Elite, diverse, and exploratory selection.

</details>

## 🎯 Perfect For: Use Cases

| Use Case                  | Why OpenEvolve Excels                                     |
| ------------------------- | --------------------------------------------------------- |
| 🏃‍♂️ Performance Optimization | Discovers hardware-specific optimizations humans miss. |
| 🧮 Algorithm Discovery      | Finds novel approaches to classic problems.           |
| 🔬 Scientific Computing    | Automates tedious manual tuning processes.               |
| 🎮 Competitive Programming | Generates multiple solution strategies.                 |
| 📊 Multi-Objective Problems| Pareto-optimal solutions across dimensions.            |

## 🛠 Installation & Setup Guide

### ⚙️ Requirements

*   **Python**: 3.10+
*   **LLM Access**: Any OpenAI-compatible API
*   **Optional**: Docker for containerized runs

### 📦 Installation Options

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

### 🔑 LLM Provider Setup

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

For rate limiting, model routing, and test-time compute:

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

## 📸 Examples Gallery: Showcasing OpenEvolve's Versatility

<div align="center">

### 🏆 **Showcase Projects**

| Project                                 | Domain             | Achievement                       | Demo                                                       |
| --------------------------------------- | ------------------ | --------------------------------- | ---------------------------------------------------------- |
| [🎯 **Function Minimization**](examples/function_minimization/)     | Optimization       | Random → Simulated Annealing | [View Results](examples/function_minimization/openevolve_output/) |
| [⚡ **MLX GPU Kernels**](examples/mlx_metal_kernel_opt/)        | Hardware           | 2-3x Apple Silicon speedup       | [Benchmarks](examples/mlx_metal_kernel_opt/README.md)             |
| [🔄 **Rust Adaptive Sort**](examples/rust_adaptive_sort/)        | Algorithms         | Data-aware sorting               | [Code Evolution](examples/rust_adaptive_sort/)                        |
| [📐 **Symbolic Regression**](examples/symbolic_regression/)  | Science            | Automated equation discovery    | [LLM-SRBench](examples/symbolic_regression/)                     |
| [🕸️ **Web Scraper + OptiLLM**](examples/web_scraper_optillm/)  | AI Integration     | Test-time compute optimization | [Smart Scraping](examples/web_scraper_optillm/)                        |

</div>

### 🎯  Function Minimization: An Illustrative Example

**See OpenEvolve's evolution from random search to sophisticated optimization in action.**

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

**Evolve prompts for enhanced LLM performance:**

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

**Automatic solution generation for programming contests:**

```python
# Problem: Find maximum subarray sum
# OpenEvolve discovers multiple approaches:

# Evolution Path 1: Brute Force → Kadane's Algorithm
# Evolution Path 2: Divide & Conquer → Optimized Kadane's
# Evolution Path 3: Dynamic Programming → Space-Optimized DP
```

[Online Judge Integration](examples/online_judge_programming/)

</details>

## ⚙️ Configuration Options: Tailor Your Evolution

OpenEvolve is highly configurable to meet your specific needs. Below are examples of the advanced features, allowing you to customize LLM behavior, database organization, and prompt engineering.

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

**Important**: Return raw values from evaluator, OpenEvolve handles binning automatically.

</details>

<details>
<summary><b>🎨 Custom Prompt Templates</b></summary>

**Advanced prompt engineering with custom templates:**

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

The **artifacts side-channel** provides rich feedback to accelerate evolution:

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

## 📊 Visualization: Track Your Progress

**Real-time evolution tracking** with an interactive web interface.

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

## 🚀 Roadmap:  Future Developments

### 🔥 Upcoming Features

*   [ ] **Multi-Modal Evolution**: Integrate images, audio, and text.
*   [ ] **Federated Learning**: Distributed evolution across multiple machines.
*   [ ] **AutoML Integration**: Hyperparameter and architecture evolution.
*   [ ] **Benchmark Suite**: Standardized evaluation across domains.

### 🌟 Research Directions

*   [ ] **Self-Modifying Prompts**: Evolution modifies its own prompting strategy.
*   [ ] **Cross-Language Evolution**: Optimize code across languages (Python → Rust → C++).
*   [ ] **Neurosymbolic Reasoning**: Combine neural and symbolic approaches.
*   [ ] **Human-AI Collaboration**: Interactive evolution with human feedback.

Want to contribute?  Check out our [roadmap discussions](https://github.com/codelion/openevolve/discussions/categories/roadmap)!

## 🤔 FAQ: Frequently Asked Questions

<details>
<summary><b>💰 How much does it cost to run?</b></summary>

**Cost depends on your LLM provider and iterations:**

*   **o3**: ~$0.15-0.60 per iteration
*   **o3-mini**: ~$0.03-0.12 per iteration
*   **Gemini-2.5-Pro**: ~$0.08-0.30 per iteration
*   **Gemini-2.5-Flash**: ~$0.01-0.05 per iteration (fastest and cheapest)
*   **Local models**: Nearly free after setup
*   **OptiLLM**: Use cheaper models with test-time compute

**Cost-saving tips:**
* Start with fewer iterations (100-200)
* Use o3-mini, Gemini-2.5-Flash or local models for exploration
* Use cascade evaluation to filter bad programs early
* Configure smaller population sizes initially

</details>

<details>
<summary><b>🆚 How does this compare to manual optimization?</b></summary>

| Aspect               | Manual                                  | OpenEvolve                              |
| -------------------- | --------------------------------------- | --------------------------------------- |
| **Initial Learning**  | Weeks to understand domain             | Minutes to start                        |
| **Solution Quality** | Depends on expertise                  | Consistently explores novel approaches |
| **Time Investment**   | Days-weeks per optimization            | Hours for complete evolution            |
| **Reproducibility**  | Hard to replicate exact process        | Perfect reproduction with seeds         |
| **Scaling**          | Doesn't scale beyond human capacity    | Parallel evolution across islands       |

**OpenEvolve shines** when you need to explore large solution spaces or optimize for multiple objectives simultaneously.

</details>

<details>
<summary><b>🔧 Can I use my own LLM?</b></summary>

**Yes!** OpenEvolve supports any OpenAI-compatible API:

*   **Commercial**: OpenAI, Google, Cohere
*   **Local**: Ollama, vLLM, LM Studio, text-generation-webui
*   **Advanced**: OptiLLM for routing and test-time compute

Just set the `api_base` in your config to point to your endpoint.

</details>

<details>
<summary><b>🚨 What if evolution gets stuck?</b></summary>

**Built-in mechanisms prevent stagnation:**

*   **Island migration**: Fresh genes from other populations
*   **Temperature control**: Exploration vs exploitation balance
*   **Diversity maintenance**: MAP-Elites prevents convergence
*   **Artifact feedback**: Error messages guide improvements
*   **Template stochasticity**: Randomized prompts break patterns

**Manual interventions:**
* Increase `num_diverse_programs` for more exploration
* Add custom feature dimensions to diversify search
* Use template variations to randomize prompts
* Adjust migration intervals for more cross-pollination

</details>

<details>
<summary><b>📈 How do I measure success?</b></summary>

**Multiple success metrics:**

1.  **Primary Metric**: Your evaluator's `combined_score` or metric average
2.  **Convergence**: Best score improvement over time
3.  **Diversity**: MAP-Elites grid coverage
4.  **Efficiency**: Iterations to reach target performance
5.  **Robustness**: Performance across different test cases

**Use the visualizer** to track all metrics in real-time and identify when evolution has converged.

</details>

### 🌟 Contributors:  A Community of Innovation

We are grateful to our amazing contributors who make OpenEvolve possible!

<a href="https://github.com/codelion/openevolve/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=codelion/openevolve" />
</a>

### 🤝 Contributing: Join the OpenEvolve Project

We welcome contributions!  Here's how to get started:

1.  🍴 **Fork** the repository
2.  🌿 **Create** your feature branch: `git checkout -b feat-amazing-feature`
3.  ✨ **Add** your changes and tests
4.  ✅ **Test** everything: `python -m unittest discover tests`
5.  📝 **Commit** with a clear message
6.  🚀 **Push** and create a Pull Request

**New to open source?** Check out our [Contributing Guide](CONTRIBUTING.md) and look for [`good-first-issue`](https://github.com/codelion/openevolve/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) labels!

### 📚 Academic & Research:  Supporting Publications

**Key Articles and Blog Posts on OpenEvolve**:

*   [Towards Open Evolutionary Agents](https://huggingface.co/blog/driaforall/towards-open-evolutionary-agents) - Discussing the evolution of coding agents and open-source.
*   [OpenEvolve: GPU Kernel Discovery](https://huggingface.co/blog/codelion/openevolve-gpu-kernel-discovery) -  Showcasing automated GPU kernel discovery with 2-3x speedups.
*   [OpenEvolve: Evolutionary Coding with LLMs](https://huggingface.co/blog/codelion/openevolve) - Introducing evolutionary algorithm discovery using large language models.

## 📊 Citation:  Cite OpenEvolve

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

### **🚀 Ready to Revolutionize Your Code?**

**Developed with ❤️ by the OpenEvolve community.**

*Star ⭐ this repository to support the discovery of breakthrough algorithms!*

</div>