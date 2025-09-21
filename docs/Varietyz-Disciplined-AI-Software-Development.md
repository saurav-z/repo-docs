<div align="center">

<img src="https://banes-lab.com/assets/images/banes_lab/700px_Main_Animated.gif" width="70" alt="Disciplined AI Development Logo" />

<a href="https://github.com/Varietyz/Disciplined-AI-Software-Development">Disciplined AI Software Development Methodology</a> © 2025 by <a href="https://www.linkedin.com/in/jay-baleine/">Jay Baleine</a> is licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a> <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" width="16" height="16"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" width="16" height="16"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="" width="16" height="16">

</div>

# Disciplined AI Software Development: Build Robust AI-Powered Software with Structure

This methodology offers a structured approach to AI software development, reducing code bloat, architectural drift, and behavioral inconsistencies.  Learn how to effectively leverage AI for software creation.  [View the original repository](https://github.com/Varietyz/Disciplined-AI-Software-Development).

## Key Features

*   **Structured Approach:**  Provides a systematic, four-stage methodology for AI-assisted development.
*   **Behavioral Consistency:**  Enforces consistent AI behavior using custom instructions and persona frameworks.
*   **Context Management:**  Utilizes file size constraints and modular design to optimize AI context windows.
*   **Data-Driven Iteration:**  Employs benchmarking and performance data for informed optimization decisions.
*   **Architectural Validation:**  Includes tools for architectural principle validation and code compliance.
*   **Reduced Debugging:**  Focuses on proactive planning to minimize debugging time.
*   **Improved Code Quality:**  Aims for architectural consistency, maintainability, and measurable performance.

## The Challenge: Context and Drift in AI-Driven Development

AI models often struggle with complex, multi-faceted software requests, leading to:

*   Unstructured and poorly organized code.
*   Code duplication across components.
*   Architectural inconsistencies.
*   Output drift and context dilution.
*   Degradation of behavioral patterns.
*   Increased debugging efforts.

## The Solution: A Four-Stage Methodology

This methodology addresses these challenges through a four-stage process incorporating systematic constraints, behavioral consistency, and validation checkpoints.

### Stage 1: AI Behavioral Configuration

Establish behavioral consistency and constraint enforcement:

1.  **Configure AI Custom Instructions:** Use [AI-PREFERENCES.XML](prompt_formats/software_development/XML/AI-PREFERENCES.XML) to set constraints and signal uncertainty (⚠️).
2.  **Recommended: Load Persona Framework:**  Load [CORE-PERSONA-FRAMEWORK.json](persona/JSON/CORE-PERSONA-FRAMEWORK.json) and select a domain-appropriate persona:
    *   [GUIDE-PERSONA.json](persona/JSON/persona_plugins/GUIDE-PERSONA.json) - Methodology enforcement.
    *   [TECDOC-PERSONA.json](persona/JSON/persona_plugins/TECDOC-PERSONA.json) - Technical documentation.
    *   [R&D-PERSONA.json](persona/JSON/persona_plugins/R&D-PERSONA.json) - Code quality standards.
    *   [MURMATE-PERSONA.json](persona/JSON/persona_plugins/MURMATE-PERSONA.json) - Visual systems.
    *   Create project-specific personas using [CREATE-PERSONA-PLUGIN.json](persona/JSON/CREATE-PERSONA-PLUGIN.json).
3.  **Recommended: Activate Persona:** Issue the command "Simulate Persona".

### Stage 2: Collaborative Planning

Share [METHODOLOGY.XML](prompt_formats/software_development/XML/METHODOLOGY.XML) to structure your project:

1.  Define scope and completion criteria.
2.  Identify components and dependencies.
3.  Structure phases logically.
4.  Generate tasks with checkpoints.

Output: A dependency-driven development plan.

### Stage 3: Systematic Implementation

Work phase by phase, section by section with each request: "Can you implement \[specific component]?"

*   **File Size Limit:** Maintain code files ≤150 lines.
    *   Smaller context windows.
    *   Focused implementation.
    *   Easier debugging and sharing.

**Implementation Flow:**  `Request specific component → AI processes → Validate → Benchmark → Continue`

### Stage 4: Data-Driven Iteration

Use the benchmarking suite to optimize based on data.

*   Feed performance data back to the AI.
*   Make decisions based on measurable outcomes.

## Why This Approach Works

*   **Focused Decision Processing:**  AI excels at focused tasks.
*   **Effective Context Management:**  Small files and bounded problems.
*   **Behavioral Constraint Enforcement:** Persona system prevents AI drift.
*   **Empirical Validation:**  Performance data drives decisions.
*   **Systematic Constraints:** Enforce consistent behavior.

## Example Projects

*   [Discord Bot Template](https://github.com/Varietyz/discord-js-bot-template)
*   [PhiCode Runtime](https://github.com/Varietyz/phicode-runtime)
*   [PhiPipe](https://github.com/Varietyz/PhiPipe)

You can compare the methodology principles to the codebase structure to see how the approach translates to working code.

## Implementation Steps

*Note: XML formats are examples; experiment with formats like JSON or Markdown.*

### Setup
1.  Configure AI with [AI-PREFERENCES.XML](prompt_formats/software_development/XML/AI-PREFERENCES.XML).
2.  RECOMMENDED: Share [CORE-PERSONA-FRAMEWORK.json](persona/JSON/CORE-PERSONA-FRAMEWORK.json) + selected [PERSONA.json](persona/JSON/persona_plugins).
3.  RECOMMENDED: Issue command: "Simulate Persona".
4.  Share [METHODOLOGY.XML](prompt_formats/software_development/XML/METHODOLOGY.XML) for planning.
5.  Collaborate on the project structure.
6.  Generate a systematic development plan.

### Execution
1.  Build Phase 0 benchmarking infrastructure first.
2.  Work through phases sequentially.
3.  Implement one component per interaction.
4.  Run benchmarks and share results.
5.  Validate architectural compliance.

### Quality Assurance
*   Performance regression detection
*   Architectural principle validation
*   Code duplication auditing
*   File size compliance checking
*   Dependency boundary verification

## Project State Extraction

Use the [project extraction tool](scripts/project_extract.py) to create project snapshots.

```bash
python scripts/project_extract.py
```

**Configuration Options:**
* `SEPARATE_FILES = False`: Single output file (for small projects).
* `SEPARATE_FILES = True`: Multiple output files (for larger projects).
* `INCLUDE_PATHS`:  Directories and files to analyze.
* `EXCLUDE_PATTERNS`:  Files to ignore.

**Output:**
*   Complete file contents with syntax highlighting.
*   Line counts with architectural warnings (⚠️ or ‼️).
*   Tree structure visualization.

*[output examples can be found here](scripts/output_example)*

## What to Expect

*   **AI Behavior:**  Reduced architectural drift and context degradation.
*   **Development Flow:** Reduced debugging cycles.
*   **Code Quality:** Consistent architecture, measurable performance, maintainable structure.

---

## LLM Model Evaluations - [Q&A Documentation](questions_answers/)

Explore model evaluations:
*[Grok 3](questions_answers/Q-A_GROK_3.md) , [Claude Sonnet 4](questions_answers/Q-A_CLAUDE_SONNET_4.md) , [DeepSeek-V3](questions_answers/Q-A_DEEPSEEK-V3.md) , [Gemini 2.5 Flash](questions_answers/Q-A_GEMINI_2.5_FLASH.md)*

All models were asked the same questions using the methodology documents as file uploads. This focused on **methodology understanding and operational behavior**.  Full evaluation results and comparative analysis are available in [Methodology Comprehension Analysis: Model Evaluation](questions_answers/Q-A_COMPREHENSION_ANALYSIS.md).

#### Coverage includes:
*   Methodology understanding and workflow patterns
*   Context retention and collaborative interaction
*   Communication adherence and AI preference compliance
*   Project initialization and Phase 0 requirements
*   Tool usage and technology stack compatibility
*   Quality enforcement and violation handling
*   User experience across different skill levels

---

## Learning the Ropes

### Getting Started

**Configuration Process:**
1.  Configure AI with [AI-PREFERENCES.XML](prompt_formats/software_development/XML/AI-PREFERENCES.XML).
2.  Share [CORE-PERSONA-FRAMEWORK.json](persona/JSON/CORE-PERSONA-FRAMEWORK.json) + [GUIDE-PERSONA.json](persona/JSON/persona_plugins/GUIDE-PERSONA.json).
3.  Issue command: "Simulate Persona".
4.  Share [METHODOLOGY.XML](prompt_formats/software_development/XML/METHODOLOGY.XML).
5.  Collaborate on project structure and phases.
6.  Generate development plan.

**Available Personas:**
*   **[GUIDE-PERSONA.json](persona/JSON/persona_plugins/GUIDE-PERSONA.json)** - Methodology enforcement.
*   **[TECDOC-PERSONA.json](persona/JSON/persona_plugins/TECDOC-PERSONA.json)** - Technical documentation.
*   **[R&D-PERSONA.json](persona/JSON/persona_plugins/R&D-PERSONA.json)** - Code quality enforcement.
*   **[MURMATE-PERSONA.json](persona/JSON/persona_plugins/MURMATE-PERSONA.json)** - Visual systems.

*[Read more about the persona framework.](persona/README.PERSONAS.md)*

**Core Documents Reference:**
*   **[AI-PREFERENCES.XML](prompt_formats/software_development/XML/AI-PREFERENCES.XML)** - Behavioral constraints.
*   **[METHODOLOGY.XML](prompt_formats/software_development/XML/METHODOLOGY.XML)** - Technical framework.
*   **[README.XML](prompt_formats/software_development/XML/README.XML)** - Implementation guidance.

*Use XML format for machine parsing.*

**Ask targeted questions:**
*   "How would Phase 0 apply to \[project type]?"
*   "What does the 150-line constraint mean for \[specific component]?"
*   "How should I structure phases for \[project description]?"
*   "Can you help decompose this project using the methodology?"

### Experimental Modification

**Create Project-Specific Personas:**

Share [CREATE-PERSONA-PLUGIN.json](persona/JSON/CREATE-PERSONA-PLUGIN.json) to generate domain-specific personas from:
*   Project documentation patterns
*   Codebase architectural philosophies
*   Domain expert behavioral frameworks

*[Read more about creating personas.](persona/README.CREATE-PERSONA.md)*

**Test constraint variations:**
*   File size limits (100, 150, 200 lines).
*   Communication constraints.
*   Phase 0 modifications.
*   Quality gate adjustments.
*   Persona behavioral pattern modifications.

**Analyze outcomes:**
*   Document behavior changes and development results.
*   Compare debugging time.
*   Track architectural compliance.
*   Monitor context retention.
*   Measure persona consistency.

**Collaborative refinement:**
Work with your AI to improve based on your context.

**Progress indicators:**
*   Reduced violations.
*   Consistent file size compliance.
*   Sustained AI behavioral adherence.
*   Maintained persona consistency.

---

# Frequently Asked Questions

## Origin & Development

<details>
<summary>What problem led you to create this methodology?</summary>

---

The consistent need to restate preferences and architectural requirements to AI systems across projects and languages.  The AI would often produce bloated or underdeveloped implementations.

This led me to focus on underlying software principles rather than syntax.  The breakthrough was understanding that everything transpires to binary - a series of "can you do this? → yes/no" decisions.

---

</details>

<details>
<summary>How did you discover these specific constraints work?</summary>

---

Through trial and error. AI systems drift, but they're more accurate with structured boundaries.  You occasionally need to remind the AI of its role to prevent deviation - like managing a well-intentioned toddler that knows the rules but sometimes pushes boundaries trying to satisfy you.

These tools are effective instruments for software development when properly constrained.

---

</details>

<details>
<summary>What failures or frustrations shaped this approach?</summary>

---

Maintenance hell was the primary driver. I grew tired of verbose responses.

Instead of venting frustration on social media about AI being "just a dumb tool," I decided to find methods that actually work. My approach may not help everyone, but I hope it benefits those who share similar AI development frustrations.

---

</details>

## Personal Practice

<details>
<summary>How consistently do you follow your own methodology?</summary>

---

Since creating the documentation, I haven't deviated. Whenever I see the model producing more lines than my methodology restricts, I immediately interrupt generation with a flag: "‼️ ARCHITECTURAL VIOLATION, ADHERE TO PRINCIPLES ‼️" I then provide the method instructions again, depending on how context is stored and which model I'm using.

---

</details>

<details>
<summary>What happens when you deviate from it?</summary>

---

I become genuinely uncomfortable. Deviation simply isn't an option anymore.

---

</details>

<details>
<summary>Which principles do you find hardest to maintain?</summary>

---

Not cursing at the AI when it drifts during complex algorithms! But seriously, it's a machine - it's not perfect, and neither are we.

---

</details>

## AI Development Journey

<details>
<summary>When did you start using AI for programming?</summary>

---

In August 2024. This experience sparked intense interest in underlying software principles rather than just syntax.

---

</details>

<details>
<summary>How has your approach evolved over time?</summary>

---

I view development like a book: syntax is the cover, logic is the content itself. I focused on core meta-principles - how software interacts, how logic flows, different algorithm types. I quickly realized everything reduces to the same foundation: question and answer sequences.

Large code structures are essentially chaotic meetings. If this applies to human communication, it must apply to software principles.

---

</details>

<details>
<summary>What were your biggest mistakes with AI collaboration?</summary>

---

Expecting it to intuitively understand my requirements, provide perfect fixes, be completely honest, and act like a true expert. This was all elaborate roleplay that produced poor code.

I learned not to feed requirements and hope for the best. Instead, I needed to collaborate actively - create plans, ask for feedback on content clarity, and identify uncertainties. This gradual process taught me the AI's actual capabilities and most effective collaboration methods.

---

</details>

## Methodology Specifics

<details>
<summary>Why 150 lines exactly?</summary>

---

Multiple benefits: easy readability, clear understanding, modularity enforcement, architectural clarity, simple maintenance, component testing, optimal AI context retention, reusability, and KISS principle adherence.

---

</details>

<details>
<summary>How did you determine Phase 0 requirements?</summary>

---

From meta-principles of software: if it displays, it must run; if it runs, it can be measured; if it can be measured, it can be optimized; if it can be optimized, it can be reliable; if it can be reliable, it can be trusted.

Regardless of project type, anything requiring architecture needs these foundations.

By testing during development, you catch inefficiencies early. Integration from the start means simply hooking up new components and running tests via command line - minimal time investment with actual value returned. I prefer validation and consistency throughout development rather than programming blind.

---

</details>

## Practical Implementation

<details>
<summary>How do you handle projects that don't fit the methodology?</summary>

---

I adapt them to fit, or if truly impossible, I adjust the method itself. This is one methodology - I can generate countless variations as needed. Having spent 6700+ hours in AI interactions across multiple domains (not just software), I've developed strong system comprehension that enables creating adjusted methodologies on demand.

---

</details>

<details>
<summary>What's the learning curve for new users?</summary>

---

I cannot accurately answer this question. I've learned that I'm neurologically different - what I perceive as easy or obvious isn't always the case for others. This question is better addressed by someone who has actually used this methodology to determine its learning curve.

---

</details>

<details>
<summary>When shouldn't someone use this approach?</summary>

---

If you're not serious about projects, despise AI, dislike planning, don't care about modularization, or are just writing simple scripts. However, for anything requiring reliability, I believe this is currently the most effective method.

You still need programming fundamentals to use this methodology effectively - it's significantly more structured than ad-hoc approaches.

---

</details>

---

## Workflow Visualization

![](mermaid_svg/methodology-workflow.svg)