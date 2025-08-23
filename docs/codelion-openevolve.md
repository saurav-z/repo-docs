# OpenEvolve: AI-Powered Algorithmic and Scientific Discovery

**Unleash the power of AI to automatically optimize and discover algorithms with OpenEvolve, an open-source evolutionary coding agent.** ([Original Repository](https://github.com/codelion/openevolve))

[![OpenEvolve Logo](openevolve-logo.png)](https://github.com/codelion/openevolve)

## Key Features

*   **Evolutionary Coding Agent:** LLM-driven evolution of entire code files.
*   **Distributed Controller Loop:** Asynchronous pipeline for efficient LLM, evaluator, and database coordination.
*   **Program Database:** Stores and samples evolved programs with evaluation metrics.
*   **Advanced Prompt Engineering:** Context-rich prompts leveraging past program performance, scores, and problem descriptions.
*   **LLM Ensemble:** Multiple language models collaborate for superior code generation.
*   **Multi-Objective Optimization:** Simultaneously optimize multiple evaluation metrics.
*   **Reproducibility:** Full deterministic reproduction with per-component seeding.
*   **Advanced LLM Integration:** Supports OpenAI-compatible endpoints, optillm integration, and plugin ecosystem.
*   **MAP-Elites Implementation:** Quality-diversity algorithm for balanced exploration/exploitation.
*   **Island-Based Evolution:** Multiple populations with migration for enhanced diversity.
*   **Multi-Language & Platform Support:** Python, Rust, R, Metal shaders, and more with platform-specific optimization.
*   **Real-Time Visualization:** Interactive web-based evolution tree viewer.
*   **Comprehensive Examples:** 12+ diverse examples spanning optimization, ML, systems programming, and scientific computing.

## Overview

OpenEvolve is an open-source platform built for automated scientific and algorithmic discovery through evolutionary computation and Large Language Models (LLMs). Originally inspired by AlphaEvolve, OpenEvolve extends the capabilities of its predecessors to deliver advanced features for reproducibility, multi-language support, sophisticated evaluation pipelines, and seamless integration with cutting-edge LLM optimization techniques. OpenEvolve empowers researchers and developers to explore the frontiers of evolutionary AI and automate code optimization for a wide range of applications.

### How OpenEvolve Works

OpenEvolve's sophisticated evolutionary pipeline includes:

1.  **Enhanced Prompt Sampler**: Creates prompts from top-performing and diverse programs, execution artifacts, and dynamic documentation.
2.  **Intelligent LLM Ensemble**: Uses weighted model combinations and test-time compute techniques.
3.  **Advanced Evaluator Pool**: Employs multi-stage cascade evaluation, artifact collection, LLM-based code quality assessment, and parallel execution.
4.  **Sophisticated Program Database**: Implements MAP-Elites for diversity, island-based populations, feature map clustering, and comprehensive metadata tracking.

### Island-Based Evolution with Worker Pinning

OpenEvolve features an island-based evolutionary architecture with multiple, isolated populations to maintain diversity and prevent premature convergence.

**Key Concepts:**

*   **Multiple Isolated Populations:** Independent evolution within each island.
*   **Periodic Migration:** Top-performing programs share beneficial mutations between islands.
*   **Worker-to-Island Pinning:** Ensures genetic isolation during parallel execution:
    ```python
    # Example: 6 workers, 3 islands
    # Worker 0, 3 → Island 0
    # Worker 1, 4 → Island 1
    # Worker 2, 5 → Island 2
    ```

**Benefits of Worker Pinning:**

*   Genetic Isolation
*   Consistent Evolution
*   Balanced Load
*   Migration Integrity
*   Automatic Distribution

## Getting Started

### Installation

```bash
git clone https://github.com/codelion/openevolve.git
cd openevolve
pip install -e .
```

### Quick Start

1.  **Set OpenAI API Key:**
    ```bash
    export OPENAI_API_KEY=your-api-key-here
    ```
2.  **Configure LLM Providers:**
    -   For non-OpenAI providers, adjust `api_base` in `config.yaml`.
    -   For advanced routing, utilize [optillm](https://github.com/codelion/optillm).

```python
import os
from openevolve import OpenEvolve

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable")

evolve = OpenEvolve(
    initial_program_path="path/to/initial_program.py",
    evaluation_file="path/to/evaluator.py",
    config_path="path/to/config.yaml"
)
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

Resume and continue evolution from saved states:

```bash
python openevolve-run.py path/to/initial_program.py path/to/evaluator.py \
  --config path/to/config.yaml \
  --checkpoint path/to/checkpoint_directory \
  --iterations 50
```

### Comparing Results Across Checkpoints

Compare and analyze solutions at different stages by examining the best programs and metrics in the checkpoint directories.

```bash
diff -u checkpoints/checkpoint_10/best_program.py checkpoints/checkpoint_20/best_program.py
cat checkpoints/checkpoint_*/best_program_info.json | grep -A 10 metrics
```

### Visualizing the Evolution Tree

Use the provided script to visualize evolution in your browser.

```bash
# Install requirements
pip install -r scripts/requirements.txt

# Start the visualization web server and have it watch the examples/ folder
python scripts/visualizer.py

# Start the visualization web server with a specific checkpoint
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

### Docker

Build and run OpenEvolve with Docker:

```bash
docker build -t openevolve .
docker run --rm -v $(pwd):/app --network="host" openevolve examples/function_minimization/initial_program.py examples/function_minimization/evaluator.py --config examples/function_minimization/config.yaml --iterations 1000
```

## Configuration

OpenEvolve is highly configurable:

*   `max_iterations`, `random_seed`
*   `llm`: `models`, `temperature`
*   `database`: `population_size`, `num_islands`, `migration_interval`, `feature_dimensions`
*   `evaluator`: `enable_artifacts`, `cascade_evaluation`, `use_llm_feedback`
*   `prompt`: `num_top_programs`, `num_diverse_programs`, `include_artifacts`

Sample configurations available in `configs/`.

### Prompt Engineering Design

OpenEvolve utilizes sophisticated prompt engineering by categorizing program examples:

1.  **Previous Attempts:** High-performing programs for guidance.
2.  **Top Programs:** Diverse approaches for exploration.
3.  **Inspirations:** Cross-island program samples to prevent convergence.

### Template Customization

Customize prompts using `.txt` files in a `template_dir`:

*   `diff_user.txt`, `full_rewrite_user.txt`, `evolution_history.txt`, `top_program.txt`, `previous_attempt.txt`

Enable template stochasticity to add random variations using placeholders.

```yaml
prompt:
  use_template_stochasticity: true
  template_variations:
    greeting:
      - "Let's improve this code."
      - "Time to enhance this program."
    analysis_intro:
      - "Current metrics show"
      - "Performance analysis indicates"
```

### Feature Dimensions in MAP-Elites

Control how programs are organized in the MAP-Elites grid:

*   **Default:** `["complexity", "diversity"]`
*   **Built-in:** `complexity`, `diversity`
*   **Custom:** Use evaluator metrics (raw values, not bin indices).

### Default Metric for Program Selection

*   Uses `combined_score` if provided.
*   Averages all numeric metrics if `combined_score` is absent.

## Artifacts Channel

Captures build errors and profiling results for LLM feedback.

**Example:**

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

Explore the `examples/` directory for a variety of applications:

*   Mathematical Optimization (Function Minimization, Circle Packing)
*   Advanced AI & LLM Integration (Web Scraper, LLM Prompt Optimization)
*   Systems & Performance Optimization (MLX Metal Kernel, Rust Adaptive Sort)
*   Scientific Computing & Discovery (Symbolic Regression, R Robust Regression, Signal Processing)
*   Web and Integration (Online Judge Programming, LM-Eval Integration)

## Preparing Your Own Problems

1.  Mark code sections to evolve with `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`.
2.  Create an evaluation function.
3.  Configure OpenEvolve.
4.  Run the evolution.

## Citation

If you use OpenEvolve in your research, please cite:

```
@software{openevolve,
  title = {OpenEvolve: an open-source evolutionary coding agent},
  author = {Asankhaya Sharma},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/codelion/openevolve}
}