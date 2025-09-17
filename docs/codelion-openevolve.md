# OpenEvolve: Unleash the Power of Autonomous Code Evolution

**Tired of manual optimization?** OpenEvolve transforms Large Language Models (LLMs) into powerful, autonomous code optimizers, enabling the discovery of breakthrough algorithms across multiple domains. Visit the [original repo](https://github.com/codelion/openevolve) to explore the future of code generation!

<div align="center">
    <img src="openevolve-logo.png" alt="OpenEvolve Logo" width="400">
    <p>
        <a href="https://github.com/codelion/openevolve/stargazers"><img src="https://img.shields.io/github/stars/codelion/openevolve?style=social" alt="GitHub stars"></a>
        <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/v/openevolve" alt="PyPI version"></a>
        <a href="https://pypi.org/project/openevolve/"><img src="https://img.shields.io/pypi/dm/openevolve" alt="PyPI downloads"></a>
        <a href="https://github.com/codelion/openevolve/blob/main/LICENSE"><img src="https://img.shields.io/github/license/codelion/openevolve" alt="License"></a>
    </p>
</div>

**Key Features:**

*   ✅ **Autonomous Discovery:** LLMs discover new algorithms without human intervention.
*   ✅ **Proven Results:** Achieve 2-3x speedups and state-of-the-art performance.
*   ✅ **Research-Grade:** Fully reproducible results with extensive evaluation pipelines.
*   ✅ **Multi-Language Support:** Python, Rust, R, Metal shaders, and more.
*   ✅ **Flexible LLM Integration:** Works with OpenAI, Google Gemini, Local Models (Ollama/vLLM) and OptiLLM.

**Explore OpenEvolve's Power:**

*   [🚀 **Quick Start**](#quick-start)
*   [**Examples Gallery**](#examples-gallery)
*   [**Crafting Effective System Messages**](#crafting-effective-system-messages)
*   [**Discussions**](https://github.com/codelion/openevolve/discussions)

---

## Why Choose OpenEvolve?

OpenEvolve surpasses manual optimization by automating the discovery and refinement of code, offering significant advantages:

| Feature              | Manual Optimization             | OpenEvolve                               |
| :------------------- | :------------------------------ | :--------------------------------------- |
| **Time to Solution** | Days to Weeks                   | Hours                                    |
| **Exploration**      | Limited by Human Creativity   | Unlimited LLM Creativity                  |
| **Reproducibility**  | Hard to Replicate              | Fully Deterministic                        |
| **Multi-objective** | Complex Tradeoffs               | Automatic Pareto Optimization             |
| **Scalability**      | Doesn't Scale                   | Parallel Evolution Across Islands         |

## Real-World Achievements

OpenEvolve has demonstrated impressive results across diverse domains:

<div align="center">

| Domain                | Achievement                                     | Example                                                             |
| :-------------------- | :---------------------------------------------- | :------------------------------------------------------------------ |
| **GPU Optimization**  | Hardware-optimized kernel discovery             | [MLX Metal Kernels](examples/mlx_metal_kernel_opt/)                    |
| **Mathematics**       | State-of-the-art circle packing (n=26)          | [Circle Packing](examples/circle_packing/)                          |
| **Algorithm Design**  | Adaptive sorting algorithms                     | [Rust Adaptive Sort](examples/rust_adaptive_sort/)                   |
| **Scientific Computing** | Automated filter design                         | [Signal Processing](examples/signal_processing/)                     |
| **Multi-Language**    | Python, Rust, R, Metal Shaders                   | [All Examples](examples/)                                           |

</div>

## 🚀 Quick Start: Get Started in 30 Seconds

Evolve your code with these simple steps:

```bash
# Install OpenEvolve
pip install openevolve

# Set your API key (using Google Gemini example)
export OPENAI_API_KEY="your-gemini-api-key"

# Run your first evolution!
python openevolve-run.py examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 50
```

**Note:** The example config uses Gemini by default. To use a different OpenAI-compatible provider, modify the `config.yaml` file. Refer to the [configs](configs/) for configuration options.

### Library Usage

Leverage OpenEvolve as a library, simplifying your workflow:

```python
from openevolve import run_evolution, evolve_function

# Inline Code Evolution (No Files Needed!)
result = run_evolution(
    initial_program='''
    def fibonacci(n):
        if n <= 1: return n
        return fibonacci(n-1) + fibonacci(n-2)
    ''',
    evaluator=lambda path: {"score": benchmark_fib(path)},
    iterations=100
)

# Direct Function Evolution
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

**Prefer Docker?** Explore [Installation & Setup](#installation--setup) for containerized options.

## Witness Code Evolution in Action

<details>
<summary><b>Circle Packing: From Random to State-of-the-Art</b></summary>

**Watch OpenEvolve discover optimal circle packing in real-time:**

| Generation 1 | Generation 190 | Generation 460 (Final) |
|--------------|----------------|----------------------|
| ![Initial](examples/circle_packing/circle_packing_1.png) | ![Progress](examples/circle_packing/circle_packing_190.png) | ![Final](examples/circle_packing/circle_packing_460.png) |
| Random placement | Learning structure | **State-of-the-art result** |

**Result**: Matches published benchmarks for n=26 circle packing problem.

</details>

<details>
<summary><b>GPU Kernel Evolution</b></summary>

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

## How OpenEvolve Works

OpenEvolve uses an advanced **evolutionary coding pipeline**:

![OpenEvolve Architecture](openevolve-architecture.png)

### Core Innovation: MAP-Elites + LLMs

*   **Quality-Diversity Evolution:** Maintains diverse populations.
*   **Island-Based Architecture:** Prevents premature convergence.
*   **LLM Ensemble:** Leverages multiple models with fallback strategies.
*   **Artifact Side-Channel:** Uses error feedback for improvement.

### Advanced Features:

<details>
<summary><b>Scientific Reproducibility</b></summary>

*   **Comprehensive Seeding**: Every component is seeded.
*   **Default Seed=42**: Reproducible out-of-the-box.
*   **Deterministic Evolution**: Exact reproduction of runs.
*   **Component Isolation**: Hash-based isolation.

</details>

<details>
<summary><b>Advanced LLM Integration</b></summary>

*   **Universal API**: Works with OpenAI, Google, local models, and proxies.
*   **Intelligent Ensembles**: Weighted combinations with fallback.
*   **Test-Time Compute**: Enhanced reasoning through proxy systems (see [OptiLLM setup](#llm-provider-setup)).
*   **Plugin Ecosystem**: Supports advanced reasoning plugins.

</details>

<details>
<summary><b>Evolution Algorithm Innovations</b></summary>

*   **Double Selection**: Different programs for performance vs inspiration.
*   **Adaptive Feature Dimensions**: Custom quality-diversity metrics.
*   **Migration Patterns**: Ring topology with controlled gene flow.
*   **Multi-Strategy Sampling**: Elite, diverse, and exploratory selection.

</details>

## Perfect For: Key Use Cases

| Use Case                 | Why OpenEvolve Excels                                  |
| :----------------------- | :---------------------------------------------------- |
| Performance Optimization | Discovers hardware-specific optimizations               |
| Algorithm Discovery      | Finds novel approaches to classic problems             |
| Scientific Computing     | Automates tedious manual tuning                       |
| Competitive Programming  | Generates multiple solution strategies                 |
| Multi-Objective Problems | Pareto-optimal solutions across multiple dimensions |

## 🛠 Installation & Setup

### Requirements

*   **Python:** 3.10+
*   **LLM Access:** OpenAI-compatible API
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
# Pull the image
docker pull ghcr.io/codelion/openevolve:latest

# Run an example
docker run --rm -v $(pwd):/app ghcr.io/codelion/openevolve:latest \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py --iterations 100
```

</details>

### Cost Estimation

**Cost depends on your LLM provider and iteration count:**

*   **o3**: ~$0.15-0.60 per iteration
*   **o3-mini**: ~$0.03-0.12 per iteration
*   **Gemini-2.5-Pro**: ~$0.08-0.30 per iteration
*   **Gemini-2.5-Flash**: ~$0.01-0.05 per iteration
*   **Local models**: Nearly free after setup
*   **OptiLLM**: Leverage cheaper models with test-time compute

**Cost-saving tips:**

*   Start with fewer iterations.
*   Use cost-effective LLM models.
*   Implement cascade evaluation to filter poor programs early.
*   Configure smaller population sizes.

### LLM Provider Setup

OpenEvolve seamlessly integrates with **any OpenAI-compatible API**:

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

## Examples Gallery

<div align="center">

### **Showcase Projects**

| Project                                               | Domain             | Achievement                           | Demo                                                                  |
| :---------------------------------------------------- | :----------------- | :------------------------------------ | :-------------------------------------------------------------------- |
| [**Function Minimization**](examples/function_minimization/)  | Optimization       | Random → Simulated Annealing          | [View Results](examples/function_minimization/openevolve_output/)       |
| [**MLX GPU Kernels**](examples/mlx_metal_kernel_opt/)     | Hardware           | Apple Silicon optimization           | [Benchmarks](examples/mlx_metal_kernel_opt/README.md)                 |
| [**Rust Adaptive Sort**](examples/rust_adaptive_sort/)      | Algorithms         | Data-aware sorting                    | [Code Evolution](examples/rust_adaptive_sort/)                         |
| [**Symbolic Regression**](examples/symbolic_regression/)    | Science            | Automated equation discovery        | [LLM-SRBench](examples/symbolic_regression/)                         |
| [**Web Scraper + OptiLLM**](examples/web_scraper_optillm/) | AI Integration     | Test-time compute optimization      | [Smart Scraping](examples/web_scraper_optillm/)                      |

</div>

### **Quick Example: Function Minimization**

**Evolve from random search to advanced optimization:**

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

**Evolution Process**

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

### **Advanced Examples**

<details>
<summary><b>Prompt Evolution</b></summary>

**Evolve prompts instead of code** for better LLM performance.  See the [LLM Prompt Optimization example](examples/llm_prompt_optimization/) for a complete case study with HotpotQA achieving +23% accuracy improvement.

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

## Configuration: Advanced Customization

OpenEvolve offers extensive customization:

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

<details>
<summary><b>🎯 Feature Engineering</b></summary>

**Control organization in the quality-diversity grid:**

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
    improvement_suggestion:
      - "Here's how we could improve this code:"
      - "I suggest the following improvements:"
      - "We can enhance this code by:"
```

**How it works:** Use placeholders like `{greeting}` or `{improvement_suggestion}` in your templates, and OpenEvolve will randomize variations for each generation.

See [prompt examples](examples/llm_prompt_optimization/templates/) for complete template customization.

</details>

## Crafting Effective System Messages

Well-crafted system messages are crucial for successful evolution. They guide the LLM's understanding of your domain, constraints, and goals.

### Why System Messages Matter

The system message in your `config.yaml` is key for success:

*   **Domain Expertise:** Provides specific knowledge.
*   **Constraint Awareness:** Defines allowed changes.
*   **Optimization Focus:** Guides meaningful improvements.
*   **Error Prevention:** Avoids common pitfalls.

### Iterative Creation Process

Create system messages through iteration:

<details>
<summary><b>🔄 Step-by-Step Process</b></summary>

**Phase 1: Initial Draft**

1.  Start with a basic system message.
2.  Run 20-50 iterations.
3.  Note "stuck" points.

**Phase 2: Refinement**

4.  Add specific guidance based on observations.
5.  Include domain-specific concepts.
6.  Define clear constraints and targets.
7.  Run more iterations.

**Phase 3: Specialization**

8.  Add examples of good/bad approaches.
9.  Include library/framework guidance.
10. Add error-avoidance patterns.
11. Fine-tune based on artifact feedback.

**Phase 4: Optimization**

12. Consider using OpenEvolve to optimize the prompt itself!
13. Measure improvements.

</details>

### Examples by Complexity

#### **Simple: General Optimization**

```yaml
prompt:
  system_message: |
    You are an expert programmer specializing in optimization algorithms.
    Your task is to improve a function minimization algorithm to find the
    global minimum reliably, escaping local minima that might trap simple algorithms.
```

#### **Intermediate: Domain-Specific Guidance**

```yaml
prompt:
  system_message: |
    You are an expert prompt engineer. Your task is to revise prompts for LLMs.

    Your improvements should:
    * Clarify vague instructions and eliminate ambiguity
    * Strengthen alignment between prompt and desired task outcome
    * Improve robustness against edge cases
    * Include formatting instructions and examples where helpful
    * Avoid unnecessary verbosity

    Return only the improved prompt text without explanations.
```

#### ⚡ **Advanced: Hardware-Specific Optimization**

```yaml
prompt:
  system_message: |
    You are an expert Metal GPU programmer specializing in custom attention
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

### Best Practices

<details>
<summary><b>🎨 Prompt Engineering Patterns</b></summary>

**Structure Your Message:** Role definition → Task/context → Optimization opportunities → Constraints → Success criteria

**Use Specific Examples:**

```yaml
# Good: "Focus on reducing memory allocations. Example: Replace `new Vector()` with pre-allocated arrays."
# Avoid: "Make the code faster"
```

**Include Domain Knowledge:**

```yaml
# Good: "For GPU kernels: 1) Memory coalescing 2) Occupancy 3) Shared memory usage"
# Avoid: "Optimize the algorithm"
```

**Set Clear Boundaries:**

```yaml
system_message: |
  MUST NOT CHANGE: ❌ Function signatures ❌ Algorithm correctness ❌ External API
  ALLOWED: ✅ Internal implementation ✅ Data structures ✅ Performance optimizations
```

</details>

<details>
<summary><b>🔬 Advanced Techniques</b></summary>

**Artifact-Driven Iteration:** Enable artifacts → Include common error patterns → Add guidance based on error feedback.

**Multi-Phase Evolution:** Start broad, then focus.

**Template Stochasticity:** See the [Configuration section](#configuration) for template variation examples.

</details>

### Meta-Evolution: Optimizing Prompts with OpenEvolve

**Use OpenEvolve to evolve your system messages!**

See the [LLM Prompt Optimization example](examples/llm_prompt_optimization/) for a complete implementation with a +23% accuracy improvement in the HotpotQA case study.

### Common Pitfalls to Avoid

*   **Too Vague**: "Make the code better" → Be specific.
*   **Too Restrictive**: Over-constraining.
*   **Missing Context**: Include domain knowledge.
*   **No Examples**: Use concrete examples.
*   **Ignoring Artifacts**: Don't skip error feedback.

## Artifacts & Debugging

**Artifacts side-channel** provides rich feedback:

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

This creates a **feedback loop** for each generation!

## Visualization

**Real-time evolution tracking** with an interactive web interface:

```bash
# Install visualization dependencies
pip install -r scripts/requirements.txt

# Launch interactive visualizer
python scripts/visualizer.py

# Or visualize a specific checkpoint
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

**Features:**

*   🌳 **Evolution tree**
*   📈 **Performance tracking**
*   🔍 **Code diff viewer**
*   📊 **MAP-Elites grid**
*   🎯 **Multi-metric analysis**

![OpenEvolve Visualizer](openevolve-visualizer.png)

## Roadmap

### **🔥 Upcoming Features**

*   [ ] **Multi-Modal Evolution**: Images, audio, and text.
*   [ ] **Federated Learning**: Distributed evolution.
*   [ ] **AutoML Integration**: Hyperparameter and architecture evolution.
*   [ ] **Benchmark Suite**: Standardized evaluation.

### **🌟 Research Directions**

*   [ ] **Self-Modifying Prompts**: Evolution modifies its own prompts.
*   [ ] **Cross-Language Evolution**: Optimization chains (Python → Rust → C++).
*   [ ] **Neurosymbolic Reasoning**: Combine neural and symbolic approaches.
*   [ ] **Human-AI Collaboration**: Interactive evolution.

Want to contribute? Check the [roadmap discussions](https://github.com/codelion/openevolve/discussions/categories/roadmap)!

## FAQ

<details>
<summary><b>💰 How much does it cost to run?</b></summary>

See the [Cost Estimation](#cost-estimation) section for detailed pricing.

</details>

<details>
<summary><b>🆚 How does this compare to manual optimization?</b></summary>

| Aspect              | Manual                      | OpenEvolve                                    |
| :------------------ | :-------------------------- | :-------------------------------------------- |
| **Learning**        | Weeks to understand domain  | Minutes to start                               |
| **Solution Quality** | Depends on expertise       | Explores novel approaches consistently          |
| **Time Investment** | Days-weeks per optimization | Hours for complete evolution                   |
| **Reproducibility** | Hard to replicate           | Perfect reproduction with seeds                  |
| **Scaling**         | Doesn't scale              | Parallel evolution across islands             |

**OpenEvolve shines** with large solution spaces and multi-objective optimization.

</details>

<details>
<summary><b>🔧 Can I use my own LLM?</b></summary>

**Yes!** OpenEvolve supports OpenAI-compatible APIs:

*   **Commercial**: OpenAI, Google, Cohere
*   **Local**: Ollama, vLLM, LM Studio
*   **Advanced**: OptiLLM

Set `api_base` in your config.

</details>

<details>
<summary><b>🚨 What if evolution gets stuck?</b></summary>

**Built-in mechanisms:**

*   Island migration.
*   Temperature control.
*   Diversity maintenance.
*   Artifact feedback.
*   Template stochasticity.

**Manual interventions:**

*   Increase exploration.
*   Add feature dimensions.
*   Use template variations.
*   Adjust migration intervals.

</details>

<details>
<summary><b>📈 How do I measure success?</b></summary>

**Multiple success metrics:**

1.  **Primary Metric**: Evaluator's combined score.
2.  **Convergence**: Best score improvement.
3.  **Diversity**: MAP-Elites grid coverage.
4.  **Efficiency**: Iterations to target performance.
5.  **Robustness**: Performance across test cases.

**Use the visualizer** to track metrics.

</details>

### **Contributors**

Thanks to all contributors!

<a href="https://github.com/codelion/openevolve/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=codelion/openevolve" />
</a>

### **Contributing**

Contributions welcome!

1.  🍴 **Fork**
2.  🌿 **Create** feature branch: `feat-amazing-feature`
3.  ✨ **Add** changes and tests.
4.  ✅ **Test**: `python -m unittest discover tests`
5.  📝 **Commit** with a message.
6.  🚀 **Push** and create a Pull Request.

See our [Contributing Guide](CONTRIBUTING.md) and look for [`good-first-issue`](https://github.com/codelion/openevolve/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)!

### **Academic & Research**

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

### **🚀 Ready to evolve your code?**

**Maintained by the OpenEvolve community**

*Star this repository if OpenEvolve helps you discover breakthrough algorithms!*

</div>