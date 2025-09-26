<div align="center">

<img src="https://banes-lab.com/assets/images/banes_lab/700px_Main_Animated.gif" width="70" />

**Unlock AI-Powered Software Development: A Disciplined Approach**

[![CC BY-SA 4.0](https://mirrors.creativecommons.org/presskit/icons/cc.svg) CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
[![View Source on GitHub](https://img.shields.io/badge/View_Source-GitHub-blue?logo=github)](https://github.com/Varietyz/Disciplined-AI-Software-Development)

</div>

---

# Disciplined AI Software Development: A Methodology for Consistent and Efficient AI Collaboration

This methodology provides a structured approach to harnessing the power of AI for software development, addressing common pitfalls like code bloat and architectural drift through systematic constraints and behavioral enforcement.

**Key Features:**

*   **Structured Workflow:** Four distinct stages with clear guidelines for planning, implementation, and iteration.
*   **Behavioral Consistency:** AI personas and custom instructions to maintain consistent collaboration patterns.
*   **Context Management:** Small file sizes to prevent the AI from juggling multiple concerns simultaneously.
*   **Empirical Validation:** Performance data to drive optimization decisions based on measurable outcomes.
*   **Architectural Compliance:** Built-in tools for performance regression, architectural principle validation and more.

## The Context Problem in AI Software Development

AI systems struggle with broad requests, often leading to:

*   Poorly structured code
*   Code duplication across components
*   Architectural inconsistencies
*   Context dilution and output drift
*   Degradation of behavior over time
*   Increased debugging time

## How This Methodology Works

This methodology leverages four stages with built-in constraints and automated checks:

### Stage 1: AI Behavioral Configuration

*   **Configure AI Custom Instructions:** Use [AI-PREFERENCES.XML](prompt_formats/software_development/XML/AI-PREFERENCES.XML) to set constraints and flag uncertainty.
*   **RECOMMENDED: Load Persona Framework:** Utilize persona files (e.g., [GUIDE-PERSONA.json](persona/JSON/persona_plugins/GUIDE-PERSONA.json)) to guide the AI's behavior.
*   **RECOMMENDED: Activate Persona:** Issue the command "Simulate Persona"

### Stage 2: Collaborative Planning

*   Share [METHODOLOGY.XML](prompt_formats/software_development/XML/METHODOLOGY.XML) for structured project planning.
*   Define scope, identify components, and structure phases.
*   Generate systematic tasks with measurable checkpoints.
    *Output: A modular development plan*

### Stage 3: Systematic Implementation

*   Implement components phase by phase, section by section.
*   **File size constraint: ≤150 lines.**
*   Follow the implementation flow: Request specific component → AI processes → Validate → Benchmark → Continue

### Stage 4: Data-Driven Iteration

*   Use a benchmarking suite (built first) to collect performance data.
*   Provide data to the AI for optimizations based on measurements.

## Why This Approach Works

*   **Focused Questions:** AI handles specific tasks more effectively.
*   **Context Management:** Small files and bounded problems prevent the AI from juggling multiple concerns simultaneously.
*   **Behavioral Consistency:** Persona system prevents AI drift.
*   **Empirical Validation:** Data-driven decisions improve code quality.
*   **Systematic Constraints:** Enforce consistent behavior through architecture and file size checks.

## Example Projects

*   [Discord Bot Template](https://github.com/Varietyz/discord-js-bot-template) - Production-ready bot with plugin architecture and comprehensive testing.
*   [PhiCode Runtime](https://github.com/Varietyz/phicode-runtime) - Programming language runtime engine with Rust acceleration.
*   [PhiPipe](https://github.com/Varietyz/PhiPipe) - CI/CD regression detection system.

## Implementation Steps

1.  Configure AI with [AI-PREFERENCES.XML](prompt_formats/software_development/XML/AI-PREFERENCES.XML).
2.  RECOMMENDED: Share [CORE-PERSONA-FRAMEWORK.json](persona/JSON/CORE-PERSONA-FRAMEWORK.json) and select persona.
3.  RECOMMENDED: Issue the command "Simulate Persona".
4.  Share [METHODOLOGY.XML](prompt_formats/software_development/XML/METHODOLOGY.XML) for planning.
5.  Collaborate on the project structure.
6.  Generate a development plan.
7.  Implement the benchmarking infrastructure first.
8.  Work through phases sequentially.
9.  Implement one component per interaction.
10. Run benchmarks and share results with AI.
11. Continuously validate architectural compliance.

## Quality Assurance

*   Performance regression detection
*   Architectural principle validation
*   Code duplication auditing
*   File size compliance checking
*   Dependency boundary verification

## Project State Extraction

Use the [project extraction tool](scripts/project_extract.py) for structured code snapshots:

```bash
python scripts/project_extract.py
```

**Key Features:**

*   Syntax highlighting of code with line counts.
*   Tree structure visualizations.
*   Shareable outputs.

## What to Expect

*   Reduced architectural drift and context degradation.
*   Consistent behavior.
*   Systematic planning that reduces debugging cycles.
*   Focused implementation.
*   Measurable performance.
*   Maintainable structure.

---

## LLM Models - [Q&A Documentation](questions_answers/)

Explore detailed Q&A for each AI model:

*   [Grok 3](questions_answers/Q-A_GROK_3.md)
*   [Claude Sonnet 4](questions_answers/Q-A_CLAUDE_SONNET_4.md)
*   [DeepSeek-V3](questions_answers/Q-A_DEEPSEEK-V3.md)
*   [Gemini 2.5 Flash](questions_answers/Q-A_GEMINI_2.5_FLASH.md)

These documents evaluate the *understanding* of the methodology and the *operational behavior* of several popular models.

---

## Getting Started: Learning the Ropes

**Configuration Process:**

1.  Configure AI with [AI-PREFERENCES.XML](prompt_formats/software_development/XML/AI-PREFERENCES.XML) as custom instructions.
2.  Share [CORE-PERSONA-FRAMEWORK.json](persona/JSON/CORE-PERSONA-FRAMEWORK.json) + [GUIDE-PERSONA.json](persona/JSON/persona_plugins/GUIDE-PERSONA.json).
3.  Issue command: "Simulate Persona".
4.  Share [METHODOLOGY.XML](prompt_formats/software_development/XML/METHODOLOGY.XML) for planning.
5.  Collaborate on project structure and phases.
6.  Generate a systematic development plan.

**Available Personas:**

*   [GUIDE-PERSONA.json](persona/JSON/persona_plugins/GUIDE-PERSONA.json) - Methodology enforcement.
*   [TECDOC-PERSONA.json](persona/JSON/persona_plugins/TECDOC-PERSONA.json) - Technical documentation specialist.
*   [R&D-PERSONA.json](persona/JSON/persona_plugins/R&D-PERSONA.json) - Research scientist with code quality enforcement.
*   [MURMATE-PERSONA.json](persona/JSON/persona_plugins/MURMATE-PERSONA.json) - Visual systems and diagram specialist.

*[Read more about the persona framework.](persona/README.PERSONAS.md)*

**Core Documents Reference:**

*   [AI-PREFERENCES.XML](prompt_formats/software_development/XML/AI-PREFERENCES.XML) - Behavioral constraints.
*   [METHODOLOGY.XML](prompt_formats/software_development/XML/METHODOLOGY.XML) - Technical framework.
*   [README.XML](prompt_formats/software_development/XML/README.XML) - Implementation guidance.

*For machine parsing, use the [XML](prompt_formats/software_development/XML/README.XML) format.*

**Ask Targeted Questions:**

*   "How would Phase 0 apply to [project type]?"
*   "What does the 150-line constraint mean for [specific component]?"
*   "How should I structure phases for [project description]?"
*   "Can you help decompose this project using the methodology?"

### Experimental Modification

**Create Project-Specific Personas:**

*   Share [CREATE-PERSONA-PLUGIN.json](persona/JSON/CREATE-PERSONA-PLUGIN.json) to create custom personas.

*[Read more about creating personas.](persona/README.CREATE-PERSONA.md)*

**Test Constraint Variations:**

*   File size limits (100 vs 150 vs 200 lines).
*   Communication constraint adjustments.
*   Phase 0 requirement modifications.
*   Quality gate threshold changes.
*   Persona behavioral pattern modifications.

**Analyze Outcomes:**

*   Document behavior changes and development results.
*   Compare debugging time across different approaches.
*   Track architectural compliance.
*   Monitor context retention and behavioral drift.
*   Measure persona consistency enforcement.

**Collaborative Refinement:**

*   Work with your AI to identify improvements based on your context.

**Progress Indicators:**

*   Reduced violations over time.
*   Consistent file size compliance.
*   Sustained AI behavioral adherence.
*   Maintained persona consistency.

---

# Frequently Asked Questions

**(See original for full Q&A content)**

*   **Origin & Development** (Answers to why the methodology was created, what worked, and what didn't)
*   **Personal Practice** (Answers on how consistently the author uses the methodology and hard it is to maintain)
*   **AI Development Journey** (Answers on when the author started using AI, how the approach evolved, and the biggest mistakes with AI collaboration)
*   **Methodology Specifics** (Answers on the specific file size limit and how the Phase 0 requirements were determined)
*   **Practical Implementation** (Answers on how the author handles projects that don't fit the methodology, the learning curve for new users, and when the methodology shouldn't be used)

---

## Workflow Visualization

![](mermaid_svg/methodology-workflow.svg)