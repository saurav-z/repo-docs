<div align="center">

<img src="https://banes-lab.com/assets/images/banes_lab/700px_Main_Animated.gif" width="70" />

**Disciplined AI Software Development: A Structured Approach for AI-Driven Projects**

[Original Repository](https://github.com/Varietyz/Disciplined-AI-Software-Development) | © 2025 by Jay Baleine | Licensed under CC BY-SA 4.0 <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" width="16" height="16"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" width="16" height="16"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="" width="16" height="16">

</div>

---

## Disciplined AI Software Development: Conquer AI Code Complexity

This methodology provides a structured, data-driven approach to AI software development, combating common issues like code bloat, architectural drift, and behavioral inconsistency, ultimately leading to more reliable and maintainable AI-powered projects.

**Key Features:**

*   **Structured Stages:** Four distinct stages (Configuration, Planning, Implementation, Iteration) to guide the development process.
*   **Behavioral Consistency:** Enforces consistent collaboration patterns through persona frameworks and systematic constraints.
*   **Architectural Discipline:** Uses file size limits and dependency gates to ensure modularity and maintainability.
*   **Data-Driven Optimization:**  Leverages benchmarking suites for performance data, replacing guesswork with measurable outcomes.
*   **Reduced Debugging:**  Emphasizes thorough planning and validation to minimize time spent on fixing architectural issues.

### The Problem: AI-Driven Code Complexity

AI systems often struggle with complex software development, leading to:

*   Unstructured, poorly organized code.
*   Code duplication across components.
*   Inconsistent architecture over time.
*   Output drift and context dilution.
*   Decreased behavioral patterns with prolonged use.
*   More debugging than planning time.

### The Solution: Disciplined AI Development

This methodology combats these issues by employing:

*   **Systematic Constraints:** Architectural checkpoints, file size limits, and dependency gates ensure consistent behavior.
*   **Behavioral Consistency Enforcement:** Persona systems maintain consistent collaboration patterns, preventing AI drift.
*   **Empirical Validation:** Performance data from benchmarks replaces subjective assessments, driving informed optimization.

### Four Stages for AI Software Success

1.  **Stage 1: AI Behavioral Configuration**
    *   **Configure AI Custom Instructions:** Use `AI-PREFERENCES.XML` to set behavioral constraints and identify uncertainty with ⚠️ indicators.
    *   **Recommended: Load Persona Framework:** Select a domain-appropriate persona from `CORE-PERSONA-FRAMEWORK.json` and persona plugins (e.g., GUIDE-PERSONA, TECDOC-PERSONA, R&D-PERSONA). Create custom personas with CREATE-PERSONA-PLUGIN.json.
    *   **Recommended: Activate Persona:** Issue the command "Simulate Persona."

2.  **Stage 2: Collaborative Planning**
    *   Share `METHODOLOGY.XML` with the AI for structured planning.
    *   Define scope, identify components, structure phases, and generate tasks.
    *   Output: A development plan with dependencies and modular boundaries.

3.  **Stage 3: Systematic Implementation**
    *   Work phase-by-phase, component-by-component.
    *   Request specific components using focused objectives.
    *   **File size constraint: ≤150 lines** to maintain focus and clarity.
    *   **Implementation Flow:** Request component -> AI processes -> Validate -> Benchmark -> Continue.

4.  **Stage 4: Data-Driven Iteration**
    *   Utilize a benchmarking suite (built first) to gather performance data.
    *   Provide data to the AI to inform optimization decisions.

### Why This Approach Works

*   **Focused Problem Solving:** AI excels at answering "Can you do A?" more reliably than multi-faceted requests.
*   **Context Management:** Small files and bounded problems prevent context overload.
*   **Behavioral Constraint Enforcement:** Persona systems prevent AI drift, ensuring consistent collaboration.
*   **Empirical Validation:** Performance data replaces subjective assessments for data-driven optimization.
*   **Systematic Constraints:** Architectural checkpoints, file size limits, and dependency gates enforce consistent behavior.

### Example Projects

The methodology has been successfully applied to the following projects:

*   [Discord Bot Template](https://github.com/Varietyz/discord-js-bot-template)
*   [PhiCode Runtime](https://github.com/Varietyz/phicode-runtime)
*   [PhiPipe](https://github.com/Varietyz/PhiPipe)

You can compare the methodology principles to the codebase structure to see how the approach translates to working code.

### Implementation Steps

*   **Note:** Adapt file formats (.xml, .json, .yaml, .md) to your use case.
*   **XML/JSON:** Ideal for code-like structure, strong code generation.
*   **MD:** Effective for documentation, fosters natural continuation.

1.  **Setup:** Configure AI with custom instructions via `AI-PREFERENCES.XML`.
2.  **Recommended:** Share  `CORE-PERSONA-FRAMEWORK.json` + a selected persona (from `PERSONA.json`).
3.  **Recommended:** Issue the "Simulate Persona" command.
4.  **Planning:** Share `METHODOLOGY.XML` for planning sessions.
5.  **Collaboration:** Collaborate on project structure and phases.
6.  **Execution:** Generate a systematic development plan.

### Quality Assurance

*   Performance regression detection
*   Architectural principle validation
*   Code duplication auditing
*   File size compliance checking
*   Dependency boundary verification

### Project State Extraction Tool

Utilize the included [project extraction tool](scripts/project_extract.py) for structured codebase snapshots:

```bash
python scripts/project_extract.py
```

**Configuration Options:**

*   `SEPARATE_FILES = False`: Single output file (small codebases).
*   `SEPARATE_FILES = True`: Multiple output files per directory (large codebases).
*   `INCLUDE_PATHS`: Specify directories and files.
*   `EXCLUDE_PATTERNS`: Exclude specific patterns.

**Output:**

*   Complete file content with syntax highlighting.
*   File line counts with architectural warnings (⚠️ for 140-150 lines, ‼️ for >150).
*   Tree structure visualization.
*   Ready-to-share format.

Use the tool for:
*   Sharing project state with AI.
*   Tracking architectural compliance.
*   Creating focused development contexts.

### What to Expect

*   **AI Behavior:** Reduced architectural drift, consistent collaboration patterns.
*   **Development Flow:** Systematic planning reduces debugging, focused implementation minimizes feature bloat.
*   **Code Quality:** Architectural consistency, measurable performance, maintainable structure as projects scale.

---

### Learn More

*   [Q&A Documentation](questions_answers/) - Insights for different LLM models.
*   [Methodology Comprehension Analysis: Model Evaluation](questions_answers/Q-A_COMPREHENSION_ANALYSIS.md) - Full evaluation results and comparative analysis.

---

### Learning the Ropes

*   **Getting Started:** Follow the setup steps described above.
*   **Available Personas:** Explore the built-in persona framework and consider creating your own.
*   **Core Documents:** Refer to `AI-PREFERENCES.XML`, `METHODOLOGY.XML`, and `README.XML`.
*   **Ask Targeted Questions:** Guide your AI with focused inquiries.

### Experimental Modifications

*   **Create Project-Specific Personas:** Customize behavior using `CREATE-PERSONA-PLUGIN.json`.
*   **Test Constraint Variations:** Experiment with file size limits and other constraints.
*   **Analyze Outcomes:** Document changes and measure their impact.
*   **Collaborative Refinement:** Partner with your AI to identify improvements.
*   **Progress Indicators:** Track consistent file size compliance, behavioral adherence, and persona consistency.

---

### Frequently Asked Questions

**(See original README for content to insert here - summarizing the questions and answers under separate headings)**

---

### Workflow Visualization

```mermaid
graph TD
    A[Configure AI & Persona] --> B{Collaborative Planning}
    B --> C{Systematic Implementation}
    C --> D{Data-Driven Iteration}
    D --> B
    C --> E[Quality Assurance]
    E --> C