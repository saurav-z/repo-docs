# OpenEvolve: Unleash AI-Driven Code Evolution for Algorithmic Discovery 🚀

**OpenEvolve is an open-source evolutionary coding agent that leverages the power of LLMs to automatically optimize and discover algorithms, empowering scientific and algorithmic breakthroughs.** ([Original Repo](https://github.com/codelion/openevolve))

![OpenEvolve Logo](openevolve-logo.png)

## Key Features

*   **Evolutionary Coding Agent:** LLM-guided evolution of complete code files, not just functions.
*   **Distributed Controller Loop:** Asynchronous pipeline for coordinating LLMs, evaluators, and databases.
*   **Program Database:** Storage and retrieval of evolved programs with comprehensive evaluation metrics.
*   **Context-Rich Prompting:** Dynamic prompts incorporating past programs, scores, and problem descriptions.
*   **LLM Ensemble:** Leverages multiple language models for superior code generation.
*   **Multi-Objective Optimization:** Simultaneously optimizes multiple evaluation metrics.
*   **Reproducibility & Control:** Comprehensive seeding for deterministic results, ensuring scientific rigor.
*   **Advanced LLM Integration:** Supports various LLM providers (OpenAI, Anthropic, local models) with [optillm](https://github.com/codelion/optillm) integration for advanced features like Mixture of Agents (MoA).
*   **Evolution Algorithm Innovations:** MAP-Elites, Island-Based Evolution, and advanced prompt engineering.
*   **Robust Evaluation & Feedback Systems:** Artifacts side-channel to capture build errors, profiling, and AI-driven code quality assessment.
*   **Multi-Language & Platform Support:** Python, Rust, R, Metal Shaders, and platform-specific optimization (Apple Silicon GPU kernels, CUDA, CPU tuning).
*   **Developer-Friendly Tools:** Interactive web-based visualization, CLI, and comprehensive examples.
*   **Performance & Scalability:** Process-based parallelism, resource management, and efficient storage.

## Overview

OpenEvolve is a powerful evolutionary coding agent designed for automating algorithm optimization and scientific discovery. Built upon the foundation of AlphaEvolve, OpenEvolve has evolved far beyond, offering advanced features for reproducibility, multi-language support, sophisticated evaluation pipelines, and seamless integration with cutting-edge LLM optimization techniques. It serves as a leading research platform for evolutionary AI and a practical tool for automated code optimization.

## How It Works

OpenEvolve employs a sophisticated evolutionary pipeline:

1.  **Enhanced Prompt Sampler:** Creates dynamic prompts including top-performing programs, diverse inspirations, execution artifacts, error feedback, and dynamic documentation (via optillm plugins).
2.  **Intelligent LLM Ensemble:** Utilizes weighted model combinations for quality/speed tradeoffs, along with test-time compute techniques (MoA, chain-of-thought, reflection) and deterministic selection.
3.  **Advanced Evaluator Pool:** Implements multi-stage cascade evaluation, artifact collection for detailed feedback, LLM-based code quality assessment, and parallel execution with resource limits.
4.  **Sophisticated Program Database:** Utilizes MAP-Elites for quality-diversity balance, island-based populations with migration, feature map clustering, and archive management with comprehensive metadata and lineage tracking.

### Core Evolution Loop

1.  **Enhanced Prompt Sampler**: Creates rich prompts containing:
    *   Top-performing programs (for optimization guidance)
    *   Diverse inspiration programs (for creative exploration)
    *   Execution artifacts and error feedback
    *   Dynamic documentation fetching (via optillm plugins)

2.  **Intelligent LLM Ensemble**:
    *   Weighted model combinations for quality/speed tradeoffs
    *   Test-time compute techniques (MoA, chain-of-thought, reflection)
    *   Deterministic selection with comprehensive seeding

3.  **Advanced Evaluator Pool**:
    *   Multi-stage cascade evaluation
    *   Artifact collection for detailed feedback
    *   LLM-based code quality assessment
    *   Parallel execution with resource limits

4.  **Sophisticated Program Database**:
    *   MAP-Elites algorithm for quality-diversity balance
    *   Island-based populations with migration
    *   Feature map clustering and archive management
    *   Comprehensive metadata and lineage tracking

## Island-Based Evolution with Worker Pinning

OpenEvolve employs an island-based evolutionary architecture to promote diversity and prevent premature convergence.

### How Islands Work
*   **Multiple Isolated Populations**: Each island maintains its own population of programs that evolve independently
*   **Periodic Migration**: Top-performing programs periodically migrate between adjacent islands (ring topology) to share beneficial mutations
*   **True Population Isolation**: Worker processes are deterministically pinned to specific islands to ensure no cross-contamination during parallel evolution

#### Worker-to-Island Pinning
```python
# Workers are distributed across islands using modulo arithmetic
worker_id = 0, 1, 2, 3, 4, 5, ...
island_id = worker_id % num_islands

# Example with 3 islands and 6 workers:
# Worker 0, 3 → Island 0
# Worker 1, 4 → Island 1
# Worker 2, 5 → Island 2
```

#### Benefits of Worker Pinning:
*   **Genetic Isolation**: Prevents accidental population mixing between islands during parallel sampling
*   **Consistent Evolution**: Each island maintains its distinct evolutionary trajectory
*   **Balanced Load**: Workers are evenly distributed across islands automatically
*   **Migration Integrity**: Controlled migration happens only at designated intervals, not due to race conditions

#### Automatic Distribution
The system handles all edge cases automatically:
*   **More workers than islands**: Multiple workers per island with balanced distribution
*   **Fewer workers than islands**: Some islands may not have dedicated workers but still participate in migration
*   **Single island**: All workers sample from the same population (degrades to standard evolution)

This architecture ensures that each island develops unique evolutionary pressures and solutions, while periodic migration allows successful innovations to spread across the population without destroying diversity.

## Getting Started

### Installation

```bash
git clone https://github.com/codelion/openevolve.git
cd openevolve
pip install -e .
```

### Quick Start

#### Setting up LLM Access

OpenEvolve uses the OpenAI SDK, compatible with any OpenAI-compatible API.

1.  **Set the API Key:**

    ```bash
    export OPENAI_API_KEY=your-api-key-here
    ```
2.  **Alternative LLM Providers:** Update the `api_base` in `config.yaml`:

    ```yaml
    llm:
      api_base: "https://your-provider-endpoint.com/v1"
    ```
3.  **Maximum Flexibility with optillm:** Recommended for advanced routing, rate limiting, or using multiple providers. Point `api_base` to your optillm instance:

    ```yaml
    llm:
      api_base: "http://localhost:8000/v1"
    ```

This setup ensures OpenEvolve integrates seamlessly with various LLM providers.

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

OpenEvolve saves checkpoints automatically (default interval: 10 iterations).  Resume runs with:

```bash
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint_directory \
  --iterations 50
```

### Comparing Results Across Checkpoints

Each checkpoint directory contains the best program up to that point.  Compare solutions over time with these examples:
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

OpenEvolve is highly configurable, see detailed examples and configuration guides in:
- `configs/default_config.yaml`: Comprehensive configuration with all available options
- `configs/island_config_example.yaml`: Advanced island-based evolution setup

### Prompt Engineering Design

OpenEvolve employs a sophisticated prompt engineering approach to optimize LLM learning.

#### Program Selection Strategy

The system distinguishes between three types of program examples shown to the LLM:
1. **Previous Attempts** (`num_top_programs`): Shows only the best performing programs to demonstrate high-quality approaches
   - Used for the "Previous Attempts" section in prompts
   - Focused on proven successful patterns
   - Helps LLM understand what constitutes good performance

2. **Top Programs** (`num_top_programs + num_diverse_programs`): Broader selection including both top performers and diverse approaches
   - Used for the "Top Performing Programs" section
   - Includes diverse programs to prevent local optima
   - Balances exploitation of known good solutions with exploration of novel approaches

3. **Inspirations** (`num_top_programs`): Cross-island program samples for creative inspiration
   - Derived from other evolution islands to maintain diversity
   - Count automatically configures based on `num_top_programs` setting
   - Prevents convergence by exposing LLM to different evolutionary trajectories

#### Design Rationale
This separation is intentional and serves multiple purposes:
- **Focused Learning**: Previous attempts show only the best patterns, helping LLM understand quality standards
- **Diversity Maintenance**: Top programs include diverse solutions to encourage exploration beyond local optima
- **Cross-Pollination**: Inspirations from other islands introduce novel approaches and prevent stagnation
- **Configurable Balance**: Adjust `num_top_programs` and `num_diverse_programs` to control exploration vs exploitation

### Template Customization

Enhance code evolution with OpenEvolve's advanced prompt template customization.

#### Custom Templates with `template_dir`
```yaml
prompt:
  template_dir: "path/to/your/templates"
```
Create `.txt` files in your template directory:
- `diff_user.txt`
- `full_rewrite_user.txt`
- `evolution_history.txt`
- `top_program.txt`
- `previous_attempt.txt`

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

### Feature Dimensions in MAP-Elites

Feature dimensions control how programs are organized in the MAP-Elites quality-diversity grid:

**Default Features**: If `feature_dimensions` is NOT specified in your config, OpenEvolve uses `["complexity", "diversity"]` as defaults.

**Built-in Features**:
- **complexity**: Code length (recommended default)
- **diversity**: Code structure diversity (recommended default)

**Custom Features**: Return raw values from your evaluator, not bin indices:
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

## Artifacts Channel

Enhance your evolution process with an artifacts side-channel:

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

## Examples

Explore the `examples/` directory for OpenEvolve implementations on:

### Mathematical Optimization
*   Function Minimization
*   Circle Packing

### Advanced AI & LLM Integration
*   Web Scraper with optillm
*   LLM Prompt Optimization

### Systems & Performance Optimization
*   MLX Metal Kernel Optimization
*   Rust Adaptive Sort

### Scientific Computing & Discovery
*   Symbolic Regression
*   R Robust Regression
*   Signal Processing

### Web and Integration Examples
*   Online Judge Programming
*   LM-Eval Integration

## Preparing Your Own Problems

1.  Mark code sections to evolve with `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`.
2.  Create an evaluation function returning a dictionary of metrics.
3.  Configure OpenEvolve with appropriate parameters.
4.  Run the evolution process.

## Citation

```
@software{openevolve,
  title = {OpenEvolve: an open-source evolutionary coding agent},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/openevolve}
}