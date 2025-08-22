# OpenEvolve: Evolving Code for Automated Discovery and Optimization

> Unleash the power of AI to automatically optimize and discover algorithms with OpenEvolve, an open-source evolutionary coding agent. ([Original Repo](https://github.com/codelion/openevolve))

![OpenEvolve Logo](openevolve-logo.png)

## Key Features

*   🚀 **LLM-Driven Evolution:** Automatically optimizes code using Large Language Models.
*   🔬 **Reproducible Science:** Ensure consistent results with comprehensive seeding and granular control.
*   🤖 **Advanced LLM Integration:** Leverages LLM Ensembles, optillm integration, and broad API support.
*   🧬 **Innovative Evolution Algorithms:** Implements MAP-Elites, Island-Based evolution, and multi-strategy selection for robust exploration.
*   📊 **Sophisticated Evaluation:** Utilizes multi-stage cascade evaluation, LLM-based feedback, and an artifact side-channel.
*   🌐 **Multi-Language & Platform Support:** Supports Python, Rust, R, Metal Shaders, and platform-specific optimizations.
*   🔧 **Developer-Friendly:** Includes a real-time web-based visualization, advanced CLI, and comprehensive examples.
*   🚀 **Scalable Performance:** Optimized for CPU-bound tasks with process-based parallelism and resource management.

## Overview

OpenEvolve is an open-source evolutionary coding agent designed for automated scientific and algorithmic discovery. Built upon the principles of AlphaEvolve, OpenEvolve extends beyond, enabling the automatic optimization and discovery of algorithms by leveraging Large Language Models (LLMs). This innovative tool provides a comprehensive platform for research in evolutionary AI and offers a practical solution for automated code optimization.

## How It Works

OpenEvolve orchestrates a sophisticated evolutionary pipeline to refine and improve code:

### Core Evolution Loop

1.  **Enhanced Prompt Sampler**: Generates rich prompts incorporating:
    *   Top-performing programs for optimization guidance
    *   Diverse inspiration programs for creative exploration
    *   Execution artifacts and error feedback
    *   Dynamic documentation fetching (via optillm plugins)

2.  **Intelligent LLM Ensemble**:
    *   Utilizes weighted model combinations for quality and speed tradeoffs
    *   Employs test-time compute techniques (MoA, chain-of-thought, reflection)
    *   Ensures deterministic selection with comprehensive seeding

3.  **Advanced Evaluator Pool**:
    *   Executes multi-stage cascade evaluation
    *   Collects artifacts for detailed feedback
    *   Implements LLM-based code quality assessment
    *   Executes in parallel with resource limits

4.  **Sophisticated Program Database**:
    *   Applies MAP-Elites algorithm for quality-diversity balance
    *   Employs island-based populations with migration
    *   Manages feature map clustering and archives
    *   Tracks comprehensive metadata and lineage

### Island-Based Evolution with Worker Pinning

OpenEvolve uses an island-based evolutionary architecture with worker pinning to maintain diversity and prevent premature convergence. This design facilitates parallel execution while preserving genetic diversity and ensuring each island develops unique evolutionary trajectories.

#### How Islands Work

*   Multiple Isolated Populations: Each island maintains its own population of programs that evolve independently.
*   Periodic Migration: Top-performing programs periodically migrate between adjacent islands to share beneficial mutations.
*   True Population Isolation: Worker processes are deterministically pinned to specific islands to ensure no cross-contamination during parallel evolution.

#### Worker-to-Island Pinning

OpenEvolve implements automatic worker-to-island pinning:

```python
# Workers are distributed across islands using modulo arithmetic
worker_id = 0, 1, 2, 3, 4, 5, ...
island_id = worker_id % num_islands

# Example with 3 islands and 6 workers:
# Worker 0, 3 → Island 0  
# Worker 1, 4 → Island 1
# Worker 2, 5 → Island 2
```

**Benefits of Worker Pinning:**
*   Genetic Isolation: Prevents accidental population mixing between islands during parallel sampling.
*   Consistent Evolution: Each island maintains its distinct evolutionary trajectory.
*   Balanced Load: Workers are evenly distributed across islands automatically.
*   Migration Integrity: Controlled migration happens only at designated intervals, not due to race conditions.

The system handles edge cases automatically:
*   More workers than islands: Multiple workers per island with balanced distribution
*   Fewer workers than islands: Some islands may not have dedicated workers but still participate in migration
*   Single island: All workers sample from the same population (degrades to standard evolution)

## Getting Started

### Installation

```bash
git clone https://github.com/codelion/openevolve.git
cd openevolve
pip install -e .
```

### Quick Start

#### Setting up LLM Access

OpenEvolve uses the OpenAI SDK, so it works with any LLM provider with an OpenAI-compatible API.

1.  **Set the API Key**:
    ```bash
    export OPENAI_API_KEY=your-api-key-here
    ```

2.  **Using Alternative LLM Providers**: Update `api_base` in `config.yaml`:

    ```yaml
    llm:
      api_base: "https://your-provider-endpoint.com/v1"
    ```

3.  **Maximum Flexibility with optillm**:
    ```yaml
    llm:
      api_base: "http://localhost:8000/v1"
    ```

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

```bash
# Compare best programs at different checkpoints
diff -u checkpoints/checkpoint_10/best_program.py checkpoints/checkpoint_20/best_program.py

# Compare metrics
cat checkpoints/checkpoint_*/best_program_info.json | grep -A 10 metrics
```

### Visualizing the Evolution Tree

```bash
# Install requirements
pip install -r scripts/requirements.txt

# Start the visualization web server and have it watch the examples/ folder
python scripts/visualizer.py

# Start the visualization web server with a specific checkpoint
python scripts/visualizer.py --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_100/
```

### Docker

```bash
docker build -t openevolve .
docker run --rm -v $(pwd):/app --network="host" openevolve examples/function_minimization/initial_program.py examples/function_minimization/evaluator.py --config examples/function_minimization/config.yaml --iterations 1000
```

## Configuration

OpenEvolve offers extensive configuration options for fine-tuning your evolution process. Example:

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

See `configs/default_config.yaml` and `configs/island_config_example.yaml` for detailed configuration options.

### Prompt Engineering Design

OpenEvolve employs a sophisticated prompt engineering approach:

#### Program Selection Strategy

1.  **Previous Attempts**: Focus on high-performing programs for quality standards.
2.  **Top Programs**: Include diverse solutions to encourage exploration.
3.  **Inspirations**: Introduce novel approaches from other islands, preventing stagnation.

#### Template Customization

Customize prompt templates and add stochasticity for diverse code evolution.

### Feature Dimensions in MAP-Elites

Control program organization within the quality-diversity grid:

**Default Features**: Uses ["complexity", "diversity"].

**Built-in Features**: Complexity and Diversity.

**Custom Features**: Combine built-in features with evaluator metrics.

Ensure your evaluator returns **raw continuous values**, not bin indices.

### Default Metric for Program Selection

OpenEvolve prioritizes program selection:

1.  **combined\_score**: Primary fitness measure if available.
2.  **Average of all metrics**: If no `combined_score` is provided.

## Artifacts Channel

Capture and utilize build errors and profiling results for better LLM feedback.

### Example: Compilation Failure Feedback

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

*   Faster convergence
*   Better error handling
*   Rich debugging context
*   Zero overhead when disabled

## Examples

Explore the `examples/` directory for various problem applications. Examples include:

*   Mathematical Optimization
*   Advanced AI & LLM Integration
*   Systems & Performance Optimization
*   Scientific Computing & Discovery
*   Web and Integration Examples

## Preparing Your Own Problems

1.  Mark code sections with `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`.
2.  Create an evaluation function returning metrics.
3.  Configure OpenEvolve.
4.  Run the evolution.

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