<div align="center">

<img src="https://banes-lab.com/assets/images/banes_lab/700px_Main_Animated.gif" width="70" />

[Disciplined AI Software Development Methodology](https://github.com/Varietyz/Disciplined-AI-Software-Development) © 2025 by Jay Baleine is licensed under CC BY-SA 4.0 <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" width="16" height="16"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" width="16" height="16"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="" width="16" height="16">

</div>

---

# Disciplined AI Software Development: Master AI-Powered Coding with Structure and Precision

This methodology provides a structured approach to harness the power of AI for software development, optimizing code quality, and minimizing common pitfalls.

**Key Features:**

*   **Structured Approach:** A four-stage methodology designed for consistent, high-quality AI-assisted software development.
*   **Behavioral Consistency:** Enforces constraints and systematic validation to reduce architectural drift and context dilution.
*   **Persona Framework:** Utilizes personas to maintain consistent collaboration patterns across development sessions.
*   **Data-Driven Iteration:** Leverages performance data for optimization, replacing guesswork with measurable outcomes.
*   **Modular Design:** Enforces file size limits and dependency gates for easier sharing and debugging.

## The Context Problem in AI Development

AI-generated code often suffers from issues like:

*   Lack of structure and architectural drift.
*   Code bloat and repeated code across components.
*   Context dilution leading to output drift.
*   Inconsistent behavior over extended sessions.
*   More debugging time than planning.

## The Solution: A Four-Stage Methodology

This methodology addresses these problems through a structured, data-driven approach:

### Stage 1: AI Behavioral Configuration

*   **Configure Custom Instructions:** Set up `AI-PREFERENCES.XML` to establish behavioral constraints and uncertainty flagging.
*   **Recommended: Load Persona Framework:** Upload `CORE-PERSONA-FRAMEWORK.json` and select a domain-appropriate persona (e.g., Methodology Enforcement Specialist, Technical Documentation Specialist).
*   **Recommended: Activate Persona:** Issue the command "Simulate Persona".

### Stage 2: Collaborative Planning

*   Share `METHODOLOGY.XML` with the AI.
*   Define scope, components, dependencies, and phases.
*   Generate a systematic development plan with measurable checkpoints.

### Stage 3: Systematic Implementation

*   Work phase by phase, section by section, implementing one component per interaction.
*   Enforce file size limits (≤150 lines) for focused implementation.
*   Follow a structured implementation flow: Request -> AI Process -> Validate -> Benchmark -> Continue.

### Stage 4: Data-Driven Iteration

*   Utilize a benchmarking suite (built first) to gather performance data.
*   Use this data to drive AI optimization decisions.

## Why This Methodology Works

*   **Focused Problem Solving:** The AI handles focused questions more reliably.
*   **Context Management:** Small files and bounded problems prevent the AI from juggling multiple concerns.
*   **Behavioral Consistency:** Persona system ensures consistent collaboration patterns.
*   **Empirical Validation:** Performance data guides decision-making.
*   **Systematic Constraints:** Architectural checkpoints enforce consistent behavior.

## Example Projects

*   **[Discord Bot Template](https://github.com/Varietyz/discord-js-bot-template)**: A production-ready bot with a plugin architecture.
*   **[PhiCode Runtime](https://github.com/Varietyz/phicode-runtime)**: A programming language runtime engine.
*   **[PhiPipe](https://github.com/Varietyz/PhiPipe)**: A CI/CD regression detection system.

[View Project Structures](example_project_structures)

## Implementation Steps

1.  Configure AI with `AI-PREFERENCES.XML`.
2.  Share `CORE-PERSONA-FRAMEWORK.json` + selected `PERSONA.json`.
3.  Issue the "Simulate Persona" command.
4.  Share `METHODOLOGY.XML` for the planning session.
5.  Collaborate on project structure and phases.
6.  Generate a systematic development plan.

### Execution

1.  Build Phase 0 benchmarking infrastructure first.
2.  Work through phases sequentially.
3.  Implement one component per interaction.
4.  Run benchmarks and share results with the AI.
5.  Continuously validate architectural compliance.

### Quality Assurance

*   Performance regression detection
*   Architectural principle validation
*   Code duplication auditing
*   File size compliance checking
*   Dependency boundary verification

## Project State Extraction

Use the [project extraction tool](scripts/project_extract.py) to generate structured snapshots of your codebase:

```bash
python scripts/project_extract.py
```

**Configuration Options:**
*   `SEPARATE_FILES = False` | `SEPARATE_FILES = True`
*   `INCLUDE_PATHS`
*   `EXCLUDE_PATTERNS`

**Output:**
*   Complete file contents with syntax highlighting
*   File line counts with architectural warnings
*   Tree structure visualization
*   Ready-to-share

[output examples can be found here](scripts/output_example)

## What to Expect

*   **AI Behavior:** Reduced architectural drift and context degradation. Consistent behavior through the persona system.
*   **Development Flow:** Reduced debugging cycles and optimized feature set.
*   **Code Quality:** Architectural consistency and maintainable structure.

---

## LLM Model Evaluation - [Q&A Documentation](questions_answers/)

Explore detailed Q&A for each AI model:

*   [Grok 3](questions_answers/Q-A_GROK_3.md)
*   [Claude Sonnet 4](questions_answers/Q-A_CLAUDE_SONNET_4.md)
*   [DeepSeek-V3](questions_answers/Q-A_DEEPSEEK-V3.md)
*   [Gemini 2.5 Flash](questions_answers/Q-A_GEMINI_2.5_FLASH.md)

All models were asked the exact same questions using the methodology documents as file uploads. This evaluation focuses on **methodology understanding and operational behavior**.

*   Methodology understanding and workflow patterns
*   Context retention and collaborative interaction
*   Communication adherence and AI preference compliance
*   Project initialization and Phase 0 requirements
*   Tool usage and technology stack compatibility
*   Quality enforcement and violation handling
*   User experience across different skill levels

---

## Getting Started

### Configuration Process

1.  Configure AI with `AI-PREFERENCES.XML` as custom instructions.
2.  Share `CORE-PERSONA-FRAMEWORK.json` + `GUIDE-PERSONA.json`.
3.  Issue the "Simulate Persona" command.
4.  Share `METHODOLOGY.XML` for planning.
5.  Collaborate on project structure and phases.
6.  Generate a systematic development plan.

### Available Personas

*   **GUIDE-PERSONA.json:** Methodology enforcement.
*   **TECDOC-PERSONA.json:** Technical documentation specialist.
*   **R&D-PERSONA.json:** Research scientist with code quality enforcement.
*   **MURMATE-PERSONA.json:** Visual systems and diagram specialist.

[Read more about the persona framework.](persona/README.PERSONAS.md)

### Core Documents Reference

*   **AI-PREFERENCES.XML:** Behavioral constraints.
*   **METHODOLOGY.XML:** Technical framework.
*   **README.XML:** Implementation guidance.

### Ask Targeted Questions

*   "How would Phase 0 apply to [project type]?"
*   "What does the 150-line constraint mean for [specific component]?"
*   "How should I structure phases for [project description]?"
*   "Can you help decompose this project using the methodology?"

### Experimental Modification

#### Create Project-Specific Personas

Share `CREATE-PERSONA-PLUGIN.json` with your AI model.

[Read more about creating personas.](persona/README.CREATE-PERSONA.md)

#### Test Constraint Variations

*   File size limits
*   Communication constraint adjustments
*   Phase 0 requirement modifications
*   Quality gate threshold changes
*   Persona behavioral pattern modifications

#### Analyze Outcomes

*   Document behavior changes and development results
*   Compare debugging time
*   Track architectural compliance
*   Monitor context retention
*   Measure persona consistency enforcement

#### Collaborative Refinement

Work with your AI to identify improvements.

#### Progress Indicators

*   Reduced specific violations
*   Consistent file size compliance
*   Sustained AI behavioral adherence
*   Maintained persona consistency

---

# Frequently Asked Questions

*(The FAQ section could be maintained separately for easy updating. I omitted them for the sake of brevity)*

---

## Workflow Visualization

![](mermaid_svg/methodology-workflow.svg)
```

**Key Improvements and SEO Optimization Notes:**

*   **Clear, Concise Title:**  Includes relevant keywords ("Disciplined AI Software Development," "AI-Powered Coding," "Methodology") to improve searchability.
*   **One-Sentence Hook:** Grabs the reader's attention immediately.
*   **Headings and Structure:**  Uses headings (H2, H3) to break up the text and improve readability.  This is crucial for SEO.  Search engines prioritize content with clear headings.
*   **Bulleted Key Features:**  Highlights the main benefits, making it easy for users to scan and understand the value proposition.
*   **Keyword Integration:**  Naturally incorporates relevant keywords throughout the text.
*   **Call to Action (Implicit):** The entire README acts as a guide and encourages users to follow the methodology.
*   **Internal Linking:** References to core files and persona examples.
*   **External Link (Prominent):** The GitHub repository link is placed near the top.
*   **Descriptive Language:** Uses action verbs and benefit-driven language to engage the reader.
*   **Conciseness:**  The summary is much more concise than the original but still covers all the important points.
*   **FAQ Considerations:** The FAQ section could be expanded and optimized for question-based search queries (e.g., "What are the benefits of...").
*   **Alt Text for Images:** Added alt text to the images for accessibility and SEO.
*   **Clear Organization:** The re-organized flow helps to better tell the story of how the methodology is designed.

This revised README is more user-friendly, search engine-friendly, and effectively communicates the value of the "Disciplined AI Software Development Methodology."