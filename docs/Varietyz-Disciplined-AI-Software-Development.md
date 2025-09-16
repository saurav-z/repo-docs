<div align="center">

<img src="https://banes-lab.com/assets/images/banes_lab/700px_Main_Animated.gif" width="70" />

<a href="https://github.com/Varietyz/Disciplined-AI-Software-Development">Disciplined AI Software Development Methodology</a> © 2025 by <a href="https://www.linkedin.com/in/jay-baleine/">Jay Baleine</a> is licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a> <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" width="16" height="16"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" width="16" height="16"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="" width="16" height="16">

</div>

---

# Disciplined AI Software Development: Achieve Robust and Maintainable AI-Powered Projects

**Tired of AI-generated code bloat and architectural drift?** This methodology provides a structured, iterative, and empirically-driven approach to AI software development, ensuring consistent results and minimizing debugging time. [Learn more and contribute on GitHub!](https://github.com/Varietyz/Disciplined-AI-Software-Development)

## Key Features:

*   **Structured Planning:** Collaborative planning with AI using `METHODOLOGY.XML` to define scope, components, and dependencies, leading to reduced debugging.
*   **Behavioral Constraint Enforcement:** Utilize `AI-PREFERENCES.XML` and persona frameworks to ensure consistent AI output and prevent drift, promoting collaboration.
*   **Modular Implementation:** Enforce a maximum file size of 150 lines to promote focused coding, easier debugging, and improved AI context management.
*   **Data-Driven Iteration:** Leverage a benchmarking suite to provide performance data, enabling optimization decisions based on measurable outcomes.
*   **Project State Extraction:** Use the included `project_extract.py` tool to generate structured snapshots of your codebase, aiding in architectural compliance.

## Methodology Breakdown

This approach utilizes a four-stage process to guide the development of AI-assisted software:

### Stage 1: AI Behavioral Configuration

*   **Configure Custom Instructions:** Set up `AI-PREFERENCES.XML` to establish behavioral constraints and uncertainty indicators.
*   **Load Persona Framework (Recommended):** Upload `CORE-PERSONA-FRAMEWORK.json` and select a domain-appropriate persona from the `persona/JSON/persona_plugins/` directory.  Example personas include `GUIDE-PERSONA.json` (methodology enforcement), `TECDOC-PERSONA.json` (technical documentation), and `R&D-PERSONA.json` (code quality).
*   **Activate Persona (Recommended):** Issue the command "Simulate Persona" to initialize the chosen persona.

### Stage 2: Collaborative Planning

*   Share `METHODOLOGY.XML` with the AI for project planning, ensuring consistent architectural results.
*   Define scope, completion criteria, and dependencies with the AI.
*   Structure project phases logically, creating measurable checkpoints.

### Stage 3: Systematic Implementation

*   Work phase by phase, implementing one component per interaction, ensuring maximum focus and reducing errors.
*   Adhere to the 150-line file size limit for optimized context and focus.
*   Follow the implementation flow: `Request specific component → AI processes → Validate → Benchmark → Continue`.

### Stage 4: Data-Driven Iteration

*   Use the benchmarking suite built in Phase 0 to evaluate performance continuously.
*   Provide performance data to the AI to inform optimization decisions based on measurements.
*   Continuously assess performance and iterate to create high quality software.

## Why This Approach Works

*   **Focus and Precision:** The methodology optimizes decision-making for AI by addressing one task at a time, promoting clarity.
*   **Context Management:** Small, self-contained files prevent AI from trying to juggle multiple elements at once.
*   **Behavioral Consistency:** The persona system uses character validation, allowing consistent results across the entire project.
*   **Empirical Validation:** Performance data replaces guesswork, supporting decisions with real-world measures.
*   **Consistent Architecture:** Architectural checkpoints, file limits, and dependency gates create strong, repeatable code.

## Example Projects

Explore how this methodology has been applied in real-world projects:

*   **[Discord Bot Template](https://github.com/Varietyz/discord-js-bot-template)** - A production-ready bot foundation.
*   **[PhiCode Runtime](https://github.com/Varietyz/phicode-runtime)** - A programming language runtime engine.
*   **[PhiPipe](https://github.com/Varietyz/PhiPipe)** - A CI/CD regression detection system.

## Implementation Steps

1.  **Configure AI:** Set up custom instructions with `AI-PREFERENCES.XML`.
2.  **Load Persona (Recommended):**  Include `CORE-PERSONA-FRAMEWORK.json` + a persona from `persona/JSON/persona_plugins/`.
3.  **Activate Persona (Recommended):**  Issue the command "Simulate Persona."
4.  **Plan:** Share `METHODOLOGY.XML` for collaborative project planning.
5.  **Develop:** Work through phases, implementing components one at a time, and creating high-quality software.

## Quality Assurance

*   Performance regression detection
*   Architectural principle validation
*   Code duplication auditing
*   File size compliance checking
*   Dependency boundary verification

## Project State Extraction

Utilize the `scripts/project_extract.py` tool to generate structured project snapshots:

*   **Configuration Options:** Customize output with `SEPARATE_FILES`, `INCLUDE_PATHS`, and `EXCLUDE_PATTERNS`.
*   **Output:** Obtain complete file contents with syntax highlighting, line counts, architectural warnings, a tree structure, and a ready-to-share output.
*   **Use Cases:** This tool helps with sharing project state, tracking architectural compliance, and focused development.

## Learning the Ropes

*   **Explore Personas:** Experiment with different personas from the `persona/` directory to tailor AI behavior to your needs.
*   **Ask Targeted Questions:**  Ask your AI questions to improve project understanding and improve the results of the process.
*   **Experimental Modification:** Test constraint variations, document results, and collaborate with the AI to refine your methodology.

## Frequently Asked Questions (FAQ)

*(This section is summarized from the original for brevity.  Refer to the original README for full details.)*

*   **Origin & Development:**  Addresses the problem the methodology solves and the iterative development process.
*   **Personal Practice:** Discusses the author's commitment to the methodology and which principles are most challenging.
*   **AI Development Journey:** Explains the author's AI development experience and past mistakes.
*   **Methodology Specifics:**  Explains key aspects of the approach, such as the 150-line limit and Phase 0 requirements.
*   **Practical Implementation:**  Covers adapting the methodology to different projects and addresses the learning curve.

## Workflow Visualization

```mermaid
graph TD
    A[Start] --> B{Configure AI & Load Persona}
    B --> C{Collaborative Planning (METHODOLOGY.XML)}
    C --> D{Systematic Implementation}
    D --> E{Data-Driven Iteration}
    E --> F[End]
    D --> G{Benchmark}
    G --> E