# OpenEvolve: Evolving Code with AI for Algorithmic and Scientific Discovery

> Unleash the power of AI to automatically optimize and discover algorithms; start your journey with [OpenEvolve](https://github.com/codelion/openevolve).

[![OpenEvolve Logo](openevolve-logo.png)](https://github.com/codelion/openevolve)

OpenEvolve is an open-source, cutting-edge evolutionary coding agent that leverages Large Language Models (LLMs) to automate the optimization and discovery of algorithms. Building upon the foundations of AlphaEvolve, OpenEvolve pushes beyond its predecessor, providing a robust platform for scientific and algorithmic breakthroughs.

## Key Features

*   **Evolutionary Coding Agent**: LLM-driven evolution of complete code files, not just functions.
*   **Distributed Controller Loop**: Asynchronous pipeline that efficiently coordinates LLMs, evaluators, and a program database.
*   **Program Database**: Secure storage and sampling of evolved programs accompanied by evaluation metrics.
*   **Prompt Sampling**: Context-rich prompts that incorporate past programs, performance scores, and detailed problem descriptions.
*   **LLM Ensemble**: Utilizing multiple language models for improved code generation.
*   **Multi-objective Optimization**: Optimizing across various evaluation metrics simultaneously.
*   **Checkpoint System**: Automatic saving and resuming of the evolution process.

### Scientific Reproducibility

*   **Comprehensive Seeding**: Ensures deterministic reproduction with hash-based component isolation.
*   **Default Reproducibility**: Enables immediate reproducible results with `seed=42`.
*   **Granular Control**: Per-component seeding for LLMs, databases, and the evaluation pipeline.

### Advanced LLM Integration

*   **Ensemble Sophistication**: Smartly combines models, incorporating intelligent fallback strategies.
*   **Test-Time Compute**: Integrates with [optillm](https://github.com/codelion/optillm) for advanced techniques like Mixture of Agents (MoA) and enhanced reasoning.
*   **Universal API Support**: Compatible with any OpenAI-compatible endpoint (Anthropic, Google, local models).
*   **Plugin Ecosystem**: Supports `optillm` plugins (e.g., readurls, executecode, z3_solver).

### Evolution Algorithm Innovations

*   **MAP-Elites Implementation**: Employs a quality-diversity algorithm for balanced exploration and exploitation.
*   **Island-Based Evolution**: Features multiple populations with periodic migration to maintain diversity.
*   **Inspiration vs. Performance**: Sophisticated prompt engineering that differentiates top performers from diverse inspirations.
*   **Multi-Strategy Selection**: Incorporates elite, diverse, and exploratory program sampling strategies.
*   **Adaptive Feature Dimensions**: Default dimensions include complexity & diversity; offers customizable multi-dimensional search spaces.

### Evaluation & Feedback Systems

*   **Artifacts Side-Channel**: Captures build errors, profiling data, and execution feedback for LLM improvements.
*   **Cascade Evaluation**: Multi-stage testing with progressive complexity for efficient resource use.
*   **LLM-Based Feedback**: Automated code quality assessment and reasoning capture.
*   **Comprehensive Error Handling**: Graceful recovery from evaluation failures with detailed diagnostics.

### Multi-Language & Platform Support

*   **Language Agnostic**: Supports Python, Rust, R, Metal shaders, and more.
*   **Platform Optimization**: Includes optimizations for Apple Silicon GPU kernels, CUDA, and CPU-specific tuning.
*   **Framework Integration**: Integrates with MLX, PyTorch, and various scientific computing libraries.

### Developer Experience & Tooling

*   **Real-Time Visualization**: Interactive, web-based evolution tree viewer for performance analysis.
*   **Advanced CLI**: Rich command-line interface for checkpoint management and configuration overrides.
*   **Comprehensive Examples**: Over 12 diverse examples spanning optimization, machine learning, systems programming, and scientific computing.
*   **Error Recovery**: Robust checkpoint loading with automatic fixes for common serialization issues.

### Performance & Scalability

*   **Process-Based Parallelism**: True parallel execution that bypasses Python's GIL for CPU-bound tasks.
*   **Resource Management**: Provides memory limits, timeouts, and resource monitoring.
*   **Efficient Storage**: Utilizes an optimized database with artifact management and cleanup policies.

## How It Works

OpenEvolve's evolution loop is a sophisticated orchestration of several key components:

![OpenEvolve Architecture](openevolve-architecture.png)

### Core Evolution Loop

1.  **Enhanced Prompt Sampler**: Generates rich prompts:
    *   Top-performing programs (for optimization guidance)
    *   Diverse inspiration programs (for creative exploration)
    *   Execution artifacts and error feedback
    *   Dynamic documentation fetching (via `optillm` plugins)

2.  **Intelligent LLM Ensemble**:
    *   Weights model combinations for quality/speed tradeoffs.
    *   Applies test-time compute techniques (MoA, chain-of-thought, reflection).
    *   Ensures deterministic selection with comprehensive seeding.

3.  **Advanced Evaluator Pool**:
    *   Implements multi-stage cascade evaluation.
    *   Collects artifacts for detailed feedback.
    *   Uses LLMs for code quality assessment.
    *   Executes in parallel with resource limits.

4.  **Sophisticated Program Database**:
    *   Utilizes MAP-Elites for quality-diversity balance.
    *   Employs island-based populations with migration.
    *   Provides feature map clustering and archive management.
    *   Offers comprehensive metadata and lineage tracking.

### Island-Based Evolution with Worker Pinning

OpenEvolve incorporates an island-based evolutionary architecture to prevent premature convergence and preserve genetic diversity.

#### How Islands Work

*   **Multiple Isolated Populations**: Each island manages an independent population of evolving programs.
*   **Periodic Migration**: Top programs migrate between islands (ring topology) to share beneficial mutations.
*   **True Population Isolation**: Worker processes are deterministically pinned to specific islands to prevent cross-contamination during parallel evolution.

#### Worker-to-Island Pinning

Worker-to-island pinning is implemented to ensure isolation during parallel execution:

```python
# Workers are distributed across islands using modulo arithmetic
worker_id = 0, 1, 2, 3, 4, 5, ...
island_id = worker_id % num_islands

# Example with 3 islands and 6 workers:
# Worker 0, 3 → Island 0
# Worker 1, 4 → Island 1
# Worker 2, 5 → Island 2
```

**Benefits of Worker Pinning**:
*   **Genetic Isolation**: Prevents accidental population mixing.
*   **Consistent Evolution**: Maintains distinct evolutionary trajectories for each island.
*   **Balanced Load**: Automatically distributes workers across islands.
*   **Migration Integrity**: Ensures migrations happen at intervals, preventing race conditions.

**Automatic Distribution**: Handles all edge cases:
*   **More workers than islands**: Multiple workers per island with balanced distribution.
*   **Fewer workers than islands**: Some islands may not have dedicated workers, but still participate in migration.
*   **Single island**: All workers sample from the same population (degrades to standard evolution).

This architecture ensures that each island develops unique evolutionary pressures and solutions, while allowing successful innovations to spread across the population without destroying diversity.

## Getting Started

### Installation

To install natively, use:

```bash
git clone https://github.com/codelion/openevolve.git
cd openevolve
pip install -e .
```

### Quick Start

#### Setting up LLM Access

OpenEvolve uses the OpenAI SDK and supports any LLM provider with an OpenAI-compatible API:

1.  **Set the API Key**: Export the `OPENAI_API_KEY` environment variable:

    ```bash
    export OPENAI_API_KEY=your-api-key-here
    ```

2.  **Using Alternative LLM Providers**:

    -   Update the `api_base` in `config.yaml` for providers other than OpenAI (e.g., Anthropic, Cohere, local models):

        ```yaml
        llm:
          api_base: "https://your-provider-endpoint.com/v1"
        ```

3.  **Maximum Flexibility with optillm**:

    -   For routing, rate limiting, or multiple providers, use [optillm](https://github.com/codelion/optillm).

    -   Point `api_base` to your `optillm` instance:

        ```yaml
        llm:
          api_base: "http://localhost:8000/v1"
        ```

    This setup ensures compatibility with a wide range of LLM providers.

```python
import os
from openevolve import OpenEvolve

# Ensure API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Initialize the system
evolve = OpenEvolve(
    initial_program_path="path/to/initial_program.py",
    evaluation_file="path/to/evaluator.py",
    config_path="path/to/config.yaml"
)

# Run the evolution
best_program = await evolve.run(iterations=1000)
print(f"Best program metrics:")
for name, value in best_program.metrics.items():
    print(f"  {name}: {value:.4f}")
```

### Command-Line Usage

```bash
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py --config path/to/config.yaml --iterations 1000
```

### Resuming from Checkpoints

```bash
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint_directory \
  --iterations 50
```

### Comparing Results Across Checkpoints

Each checkpoint directory contains the best program found up to that point, making it easy to compare solutions over time:

```
checkpoints/
  checkpoint_10/
    best_program.py         # Best program at iteration 10
    best_program_info.json  # Metrics and details
    programs/               # All programs evaluated so far
    metadata.json           # Database state
  checkpoint_20/
    best_program.py         # Best program at iteration 20
    ...
```

You can compare the evolution of solutions by examining the best programs at different checkpoints:

```bash
# Compare best programs at different checkpoints
diff -u checkpoints/checkpoint_10/best_program.py checkpoints/checkpoint_20/best_program.py

# Compare metrics
cat checkpoints/checkpoint_*/best_program_info.json | grep -A 10 metrics
```

### Visualizing the evolution tree

The script in `scripts/visualize.py` allows you to visualize the evolution tree and display it in your webbrowser. The script watches live for the newest checkpoint directory in the examples/ folder structure and updates the graph. Alternatively, you can also provide a specific checkpoint folder with the `--path` parameter.

```bash
# Install requirements
pip install -r scripts/requirements.txt

# Start the visualization web server and have it watch the examples/ folder
python scripts/visualizer.py

# Start the visualization web server with a specific checkpoint
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

In the visualization UI, you can
- see the branching of your program evolution in a network visualization, with node radius chosen by the program fitness (= the currently selected metric),
- see the parent-child relationship of nodes and click through them in the sidebar (use the yellow locator icon in the sidebar to center the node in the graph),
- select the metric of interest (with the available metric choices depending on your data set),
- highlight nodes, for example the top score (for the chosen metric) or the MAP-elites members,
- click nodes to see their code and prompts (if available from the checkpoint data) in a sidebar,
- in the "Performance" tab, see their selected metric score vs generation in a graph

![OpenEvolve Visualizer](openevolve-visualizer.png)

### Docker

```bash
docker build -t openevolve .
docker run --rm -v $(pwd):/app --network="host" openevolve examples/function_minimization/initial_program.py examples/function_minimization/evaluator.py --config examples/function_minimization/config.yaml --iterations 1000
```

## Configuration

OpenEvolve provides extensive configuration options.

```yaml
# Example configuration showcasing advanced features
max_iterations: 1000
random_seed: 42  # Full reproducibility by default

llm:
  # Advanced ensemble configuration
  models:
    - name: "gemini-2.0-flash-lite"
      weight: 0.7
    - name: "moa&readurls-gemini-2.0-flash"  # optillm test-time compute
      weight: 0.3
  temperature: 0.7

database:
  # MAP-Elites configuration
  population_size: 500
  num_islands: 5  # Island-based evolution
  migration_interval: 20
  feature_dimensions: ["complexity", "diversity"]  # Default quality-diversity features

evaluator:
  # Advanced evaluation features
  enable_artifacts: true  # Capture execution feedback
  cascade_evaluation: true  # Multi-stage testing
  use_llm_feedback: true  # AI-based code quality assessment

prompt:
  # Sophisticated prompt engineering
  num_top_programs: 3      # Performance examples
  num_diverse_programs: 2  # Creative inspiration
  include_artifacts: true  # Execution feedback

  # Template customization
  template_dir: null               # Directory for custom prompt templates
  use_template_stochasticity: true # Enable random variations in prompts
  template_variations: {}          # Define variation placeholders
```

Sample configurations are available in the `configs/` directory:

*   `default_config.yaml`: Comprehensive configuration with all options.
*   `island_config_example.yaml`: Advanced island-based evolution setup.

### Prompt Engineering Design

OpenEvolve employs a sophisticated prompt engineering approach:

#### Program Selection Strategy

*   **Previous Attempts** (`num_top_programs`): Best performing programs.
    *   Focuses on successful patterns.
    *   Helps LLM understand good performance.
*   **Top Programs** (`num_top_programs + num_diverse_programs`): Top performers plus diverse approaches.
    *   Includes diverse programs to prevent local optima.
    *   Balances exploitation and exploration.
*   **Inspirations** (`num_top_programs`): Cross-island program samples for inspiration.
    *   Derived from other evolution islands.
    *   Prevents convergence by exposing different evolutionary trajectories.

#### Design Rationale

This separation serves multiple purposes:

*   **Focused Learning**: Previous attempts show best practices.
*   **Diversity Maintenance**: Top programs encourage exploration.
*   **Cross-Pollination**: Inspirations introduce novel approaches.
*   **Configurable Balance**: `num_top_programs` and `num_diverse_programs` control exploration vs. exploitation.

### Template Customization

OpenEvolve supports advanced prompt template customization:

#### Custom Templates with `template_dir`

```yaml
prompt:
  template_dir: "path/to/your/templates"
```

Create `.txt` files in your template directory:

*   `diff_user.txt` - Template for diff-based evolution.
*   `full_rewrite_user.txt` - Template for full code rewrites.
*   `evolution_history.txt` - Format for presenting evolution history.
*   `top_program.txt` - Format for top-performing programs.
*   `previous_attempt.txt` - Format for previous attempts.

#### Template Variations with Stochasticity

```yaml
prompt:
  use_template_stochasticity: true
  template_variations:
    greeting:
      - "Let's improve this code."
      - "Time to enhance this program."
      - "Here's how we can optimize:"
    analysis_intro:
      - "Current metrics show"
      - "Performance analysis indicates"
      - "The evaluation reveals"
```

Use variation placeholders in custom templates:

```
# custom_template.txt
{greeting}
{analysis_intro} the following results:
{metrics}
```

### Feature Dimensions in MAP-Elites

Feature dimensions control program organization in the MAP-Elites grid:

**Default Features**: If `feature_dimensions` is NOT specified in your config, OpenEvolve uses `["complexity", "diversity"]` as defaults.

**Built-in Features** (always computed internally by OpenEvolve):
- **complexity**: Code length (recommended default)
- **diversity**: Code structure diversity compared to other programs (recommended default)

Only `complexity` and `diversity` are used as defaults because they work well across all program types.

**Custom Features**: You can mix built-in features with metrics from your evaluator:

```yaml
database:
  feature_dimensions: ["complexity", "performance", "correctness"]  # Mix of built-in and custom
  # Per-dimension bin configuration (optional)
  feature_bins:
    complexity: 10        # 10 bins for complexity
    performance: 20       # 20 bins for performance (from YOUR evaluator)
    correctness: 15       # 15 bins for correctness (from YOUR evaluator)
```

**CRITICAL: Return Raw Values, Not Bin Indices**: For custom feature dimensions, your evaluator must return **raw continuous values**, not pre-computed bin indices. OpenEvolve handles all scaling and binning internally.

```python
# ✅ CORRECT: Return raw values
return {
    "combined_score": 0.85,
    "prompt_length": 1247,     # Actual character count
    "execution_time": 0.234    # Raw time in seconds
}

# ❌ WRONG: Don't return bin indices
return {
    "combined_score": 0.85,
    "prompt_length": 7,        # Pre-computed bin index
    "execution_time": 3        # Pre-computed bin index
}
```

OpenEvolve automatically handles:
- Min-max scaling to [0,1] range
- Binning into the specified number of bins
- Adaptive scaling as the value range expands during evolution

**Important**: OpenEvolve will raise an error if a specified feature is not found in the evaluator's metrics. This ensures your configuration is correct. The error message will show available metrics to help you fix the configuration.

See the [Configuration Guide](configs/default_config.yaml) for a full list of options.

### Default Metric for Program Selection

When comparing and selecting programs, OpenEvolve uses the following priority:
1. **combined_score**: If your evaluator returns a `combined_score` metric, it will be used as the primary fitness measure
2. **Average of all metrics**: If no `combined_score` is provided, OpenEvolve calculates the average of all numeric metrics returned by your evaluator

This ensures programs can always be compared even without explicit fitness definitions. For best results, consider having your evaluator return a `combined_score` that represents overall program fitness.

## Artifacts Channel

The artifacts side-channel enables evaluators to capture build errors, profiling results, etc., providing better feedback to the LLM.

```python
from openevolve.evaluation_result import EvaluationResult

return EvaluationResult(
    metrics={"compile_ok": 0.0, "score": 0.0},
    artifacts={
        "stderr": "SyntaxError: invalid syntax (line 15)",
        "traceback": "...",
        "failure_stage": "compilation"
    }
)
```

### Configuration

```yaml
# config.yaml
evaluator:
  enable_artifacts: true

prompt:
  include_artifacts: true
  max_artifact_bytes: 4096  # 4KB limit in prompts
  artifact_security_filter: true
```

```bash
# Environment variable to disable artifacts
export ENABLE_ARTIFACTS=false
```

### Benefits

*   **Faster convergence** - LLMs can see what went wrong and fix it directly
*   **Better error handling** - Compilation and runtime failures become learning opportunities
*   **Rich debugging context** - Full stack traces and error messages guide improvements
*   **Zero overhead** - When disabled, no performance impact on evaluation

## Examples

See the `examples/` directory for examples:

### Mathematical Optimization

#### [Function Minimization](examples/function_minimization/)

Demonstrates evolution from random search to simulated annealing.

#### [Circle Packing](examples/circle_packing/)

Achieves state-of-the-art results.

### Advanced AI & LLM Integration

#### [Web Scraper with optillm](examples/web_scraper_optillm/)

Integration with [optillm](https://github.com/codelion/optillm) for test-time compute optimization.

#### [LLM Prompt Optimization](examples/llm_prompt_optimization/)

Evolving prompts on HuggingFace datasets.

### Systems & Performance Optimization

#### [MLX Metal Kernel Optimization](examples/mlx_metal_kernel_opt/)

Automated discovery of custom GPU kernels for Apple Silicon.

#### [Rust Adaptive Sort](examples/rust_adaptive_sort/)

Evolution of sorting algorithms.

### Scientific Computing & Discovery

#### [Symbolic Regression](examples/symbolic_regression/)

Automated discovery of mathematical expressions.

#### [R Robust Regression](examples/r_robust_regression/)

Developing robust regression methods.

#### [Signal Processing](examples/signal_processing/)

Automated design of digital filters.

### Web and Integration Examples

#### [Online Judge Programming](examples/online_judge_programming/)

Automated competitive programming.

#### [LM-Eval Integration](examples/lm_eval/)

Working with standard ML evaluation harnesses.

## Preparing Your Own Problems

1.  **Mark code sections** with `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`.
2.  **Create an evaluation function**.
3.  **Configure OpenEvolve**.
4.  **Run the evolution**.

## Citation

```
@software{openevolve,
  title = {OpenEvolve: an open-source evolutionary coding agent},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/openevolve}
}
```