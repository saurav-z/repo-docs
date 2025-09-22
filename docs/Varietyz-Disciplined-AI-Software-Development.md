<div align="center">

<img src="https://banes-lab.com/assets/images/banes_lab/700px_Main_Animated.gif" width="70" alt="Animated GIF" />

<a href="https://github.com/Varietyz/Disciplined-AI-Software-Development">Disciplined AI Software Development Methodology</a> © 2025 by <a href="https://www.linkedin.com/in/jay-baleine/">Jay Baleine</a> is licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a> <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="CC" width="16" height="16"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="BY" width="16" height="16"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="SA" width="16" height="16">

</div>

---

# Disciplined AI Software Development: Build Robust AI-Driven Software with a Structured Approach

This methodology provides a structured approach for developing AI-powered software, combating common issues like code bloat and architectural drift.  [Explore the original repository for full details.](https://github.com/Varietyz/Disciplined-AI-Software-Development)

**Key Features:**

*   **Structured Four-Stage Process:**  Follow a clear, repeatable methodology.
*   **AI Behavioral Configuration:**  Establish consistent AI behavior through custom instructions and persona frameworks.
*   **Collaborative Planning:**  Work with AI to define scope, identify components, and structure your project.
*   **Systematic Implementation:**  Enforce constraints (e.g., file size limits) for focused development.
*   **Data-Driven Iteration:**  Optimize performance based on benchmarking results.
*   **Performance Regression Detection:**  Detect regression with architectural validation and file size compliance checking.
*   **Architectural Principle Validation:** Validate the consistency and boundaries of your code.
*   **Example Projects:**  Learn from and use the example projects.

## How It Works

This methodology utilizes four distinct stages with systematic constraints, behavioral enforcement, and rigorous validation:

### Stage 1: AI Behavioral Configuration

*   **Configure AI Custom Instructions:** Set up behavioral constraints using `AI-PREFERENCES.XML`.
*   **Recommended: Load Persona Framework:** Load a persona to prevent "vibe coding".
*   **Recommended: Activate Persona:** Issue the command to "Simulate Persona".

### Stage 2: Collaborative Planning

*   Share `METHODOLOGY.XML` to structure your project plan with the AI.
*   Define scope, identify components, and structure phases.
*   Generate a systematic development plan.

### Stage 3: Systematic Implementation

*   Implement components, one per interaction, adhering to a file size of ≤150 lines.
*   Implement phase by phase and section by section.
*   **Implementation Flow:**
    ```
    Request specific component → AI processes → Validate → Benchmark → Continue
    ```

### Stage 4: Data-Driven Iteration

*   Use benchmarking data to guide optimization decisions.
*   Continuously refine performance based on measurable results.

## Why This Approach Works

This methodology optimizes the development cycle, which reduces architectural drift and context degradation.  It ensures:

*   **Focus and Consistency:** Addresses multiple concerns, small file sizes, and bounded problems.
*   **Behavioral Constraint Enforcement:** Persona system maintains consistent collaboration patterns.
*   **Empirical Validation:** Performance data replaces subjective assessment.
*   **Systematic Constraints:** Enforces consistent behavior.

## Implementation Steps

1.  **Configure AI:**  Use `AI-PREFERENCES.XML` as custom instructions.
2.  **Load Persona:** Share `CORE-PERSONA-FRAMEWORK.json` + selected `PERSONA.json`.
3.  **Activate Persona:**  "Simulate Persona."
4.  **Plan:** Share `METHODOLOGY.XML`.
5.  **Collaborate:**  Work on the project structure and phases.
6.  **Generate:** Create a systematic development plan.
7.  **QA:** Continuous performance regression detection, architectural principle validation, code duplication auditing, file size compliance checking, and dependency boundary verification.

## Project State Extraction

Use the included `project_extract.py` script to generate structured snapshots of your codebase:

```bash
python scripts/project_extract.py
```

**Configuration Options:**

*   `SEPARATE_FILES = False`: Single output file.
*   `SEPARATE_FILES = True`: Multiple output files per directory.
*   `INCLUDE_PATHS`: Directories and files to analyze.
*   `EXCLUDE_PATTERNS`: Skip cache directories, build artifacts, and generated files.

**Output:**

*   Complete file contents with syntax highlighting.
*   File line counts with architectural warnings.
*   Tree structure visualization.

*[output examples can be found here](scripts/output_example)*

## What to Expect

*   **Reduced Drift:** Minimize architectural drift and context degradation.
*   **Consistent Behavior:** Persona system maintains consistent collaboration.
*   **Reduced Debugging:** Systematic planning reduces debugging cycles.
*   **Maintainable Code:** Achieve architectural consistency and scalable structure.

---

## LLM Model Evaluations - [Q&A Documentation](questions_answers/)

Explore detailed Q&A for each AI model:
*[Grok 3](questions_answers/Q-A_GROK_3.md) , [Claude Sonnet 4](questions_answers/Q-A_CLAUDE_SONNET_4.md) , [DeepSeek-V3](questions_answers/Q-A_DEEPSEEK-V3.md) , [Gemini 2.5 Flash](questions_answers/Q-A_GEMINI_2.5_FLASH.md)*

All models were asked the **exact same questions** using the methodology documents as file uploads. This evaluation focuses on **methodology understanding and operational behavior**, no code was generated.

*   **Note:** No code generation was performed.

#### Coverage includes:

*   Methodology understanding and workflow patterns
*   Context retention and collaborative interaction
*   Communication adherence and AI preference compliance
*   Project initialization and Phase 0 requirements
*   Tool usage and technology stack compatibility
*   Quality enforcement and violation handling
*   User experience across different skill levels

---

## Getting Started

**Configuration Process:**

1.  Configure AI with `AI-PREFERENCES.XML` as custom instructions.
2.  Share `CORE-PERSONA-FRAMEWORK.json` + `GUIDE-PERSONA.json`.
3.  Issue command: "Simulate Persona."
4.  Share `METHODOLOGY.XML`.
5.  Collaborate on project structure and phases.
6.  Generate a systematic development plan.

**Available Personas:**

*   **[GUIDE-PERSONA.json](persona/JSON/persona_plugins/GUIDE-PERSONA.json)** - Methodology enforcement
*   **[TECDOC-PERSONA.json](persona/JSON/persona_plugins/TECDOC-PERSONA.json)** - Technical documentation specialist
*   **[R&D-PERSONA.json](persona/JSON/persona_plugins/R&D-PERSONA.json)** - Research scientist with code quality enforcement
*   **[MURMATE-PERSONA.json](persona/JSON/persona_plugins/MURMATE-PERSONA.json)** - Visual systems and diagram specialist

*[Read more about the persona framework.](persona/README.PERSONAS.md)*

**Core Documents Reference:**

*   **`AI-PREFERENCES.XML`:** Behavioral constraints.
*   **`METHODOLOGY.XML`:** Technical framework.
*   **`README.XML`:** Implementation guidance.

*For machine parsing, use the [XML](prompt_formats/software_development/XML/README.XML) format.*

**Ask targeted questions to improve AI model understanding:**

*   "How would Phase 0 apply to [project type]?"
*   "What does the 150-line constraint mean for [specific component]?"
*   "How should I structure phases for [project description]?"
*   "Can you help decompose this project using the methodology?"

### Experimental Modification

*   **Create Project-Specific Personas:**  Use `CREATE-PERSONA-PLUGIN.json`.
    *[Read more about creating personas.](persona/README.CREATE-PERSONA.md)*
*   **Test Constraint Variations:** File size limits, communication constraints.
*   **Analyze Outcomes:** Document behavior changes and results.
*   **Collaborative Refinement:** Improve the methodology with your AI.

---

# Frequently Asked Questions

*(See the original README for full details on the following topics)*

*   **Origin & Development**
*   **Personal Practice**
*   **AI Development Journey**
*   **Methodology Specifics**
*   **Practical Implementation**

---

## Workflow Visualization

![](mermaid_svg/methodology-workflow.svg)