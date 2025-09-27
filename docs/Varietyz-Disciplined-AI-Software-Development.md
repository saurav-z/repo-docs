<div align="center">

<img src="https://banes-lab.com/assets/images/banes_lab/700px_Main_Animated.gif" width="70" />

<a href="https://github.com/Varietyz/Disciplined-AI-Software-Development">Disciplined AI Software Development Methodology</a> © 2025 by <a href="https://www.linkedin.com/in/jay-baleine/">Jay Baleine</a> is licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a> <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" width="16" height="16"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" width="16" height="16"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="" width="16" height="16">

</div>

# Disciplined AI Software Development: Conquer AI-Powered Coding Challenges with Structure and Consistency

This methodology provides a structured approach for leveraging AI in software development, combating issues like code bloat and architectural drift to build more reliable and maintainable projects. **[Explore the original repository](https://github.com/Varietyz/Disciplined-AI-Software-Development) to supercharge your AI coding workflow.**

## Key Features

*   **Structured Approach:** Four-stage methodology with defined processes for planning, implementation, and iteration.
*   **Behavioral Consistency:** Enforce constraints and utilize personas to guide AI behavior and maintain project consistency.
*   **Context Management:**  Employs small file sizes (≤150 lines) to optimize AI context windows and improve focus.
*   **Data-Driven Iteration:** Emphasizes performance benchmarking and feedback loops for continuous improvement.
*   **Project Extraction Tool:**  A tool is included to generate structured snapshots of your codebase, aiding analysis and AI collaboration.

## The Challenge: AI-Driven Development Pain Points

AI, when tasked with broad software development goals, can often lead to:

*   Poorly structured code
*   Inconsistent architectural approaches
*   Context dilution issues
*   Unexpected behavioral patterns
*   Increased debugging time

## How This Methodology Solves the Problem

This methodology addresses these issues by implementing:

*   **Systematic Constraints:** Implementing architectural checkpoints, file size limits, and dependency gates forces consistent behavior.
*   **Behavioral Consistency Enforcement:** Using the persona system to prevent AI drift through character validation.
*   **Empirical Validation:** Employing performance data to drive optimization, eliminating guesswork.

## The Four-Stage Process

1.  **AI Behavioral Configuration:**
    *   Set AI custom instructions using [AI-PREFERENCES.XML](prompt_formats/software_development/XML/AI-PREFERENCES.XML) to establish constraints and uncertainty indicators.
    *   (Recommended) Load persona framework: [CORE-PERSONA-FRAMEWORK.json](persona/JSON/CORE-PERSONA-FRAMEWORK.json) and select/create a domain-appropriate persona, e.g., [GUIDE-PERSONA.json](persona/JSON/persona_plugins/GUIDE-PERSONA.json), [TECDOC-PERSONA.json](persona/JSON/persona_plugins/TECDOC-PERSONA.json), [R&D-PERSONA.json](persona/JSON/persona_plugins/R&D-PERSONA.json), [MURMATE-PERSONA.json](persona/JSON/persona_plugins/MURMATE-PERSONA.json), or create a project-specific one via [CREATE-PERSONA-PLUGIN.json](persona/JSON/CREATE-PERSONA-PLUGIN.json)
    *   (Recommended) Issue command: "Simulate Persona" to activate.
2.  **Collaborative Planning:** Share [METHODOLOGY.XML](prompt_formats/software_development/XML/METHODOLOGY.XML) to define scope, dependencies, and generate systematic tasks.
3.  **Systematic Implementation:** Implement components phase-by-phase, ensuring each request follows: "Can you implement [specific component]?" and adhere to the file size constraint (≤150 lines).
4.  **Data-Driven Iteration:** Use a benchmarking suite (built first) to gather performance data and feed it back to the AI for iterative optimization.

## Why This Methodology Works

*   **Focused Interactions:** The methodology enables AI to more reliably handle specific tasks.
*   **Improved Context Management:** Small files and bounded problems prevent context-switching fatigue.
*   **Consistent Collaboration:** The persona system ensures reliable collaboration.
*   **Performance-Driven Decisions:** Decisions are based on measurable outcomes.
*   **Architectural Integrity:** Systematic constraints force consistent behavior.

## Example Projects

*   **[Discord Bot Template](https://github.com/Varietyz/discord-js-bot-template)**
*   **[PhiCode Runtime](https://github.com/Varietyz/phicode-runtime)**
*   **[PhiPipe](https://github.com/Varietyz/PhiPipe)**

## Implementation Steps

1.  **Setup:**
    *   Configure AI custom instructions with [AI-PREFERENCES.XML](prompt_formats/software_development/XML/AI-PREFERENCES.XML).
    *   (Recommended) Share [CORE-PERSONA-FRAMEWORK.json](persona/JSON/CORE-PERSONA-FRAMEWORK.json) and a persona.
    *   (Recommended) Issue command: "Simulate Persona".
    *   Share [METHODOLOGY.XML](prompt_formats/software_development/XML/METHODOLOGY.XML).
    *   Collaborate on the project structure and phases.
    *   Generate a systematic development plan.
2.  **Execution:**
    *   Build Phase 0's benchmarking infrastructure first.
    *   Work through phases sequentially.
    *   Implement one component per interaction.
    *   Run benchmarks and share results.
    *   Continuously validate architectural compliance.
3.  **Quality Assurance:** Employ performance regression detection, architectural principle validation, code duplication auditing, file size compliance checks, and dependency boundary verification.

## Project State Extraction

Use `python scripts/project_extract.py` to generate structured snapshots of your codebase.

**Configuration Options:**
*   `SEPARATE_FILES = False`: Single output file.
*   `SEPARATE_FILES = True`: Multiple files per directory.
*   `INCLUDE_PATHS`: Specify files to analyze.
*   `EXCLUDE_PATTERNS`: Specify files to exclude.

**Output:**
*   Complete file contents with syntax highlighting.
*   File line counts with architectural warnings (⚠️ for 140-150 lines, ‼️ for >150 lines).
*   Tree structure visualization.
*   Ready-to-share project state.

*[output examples can be found here](scripts/output_example)*

## Expected Outcomes

*   Reduced architectural drift and context degradation.
*   Consistent AI behavior through the use of a persona system.
*   Systematic planning, leading to a reduction in debugging cycles.
*   Improved code quality and maintainability as projects scale.

---

## LLM Model Evaluations - [Q&A Documentation](questions_answers/)

Explore detailed Q&A for various AI models, including:
*[Grok 3](questions_answers/Q-A_GROK_3.md) , [Claude Sonnet 4](questions_answers/Q-A_CLAUDE_SONNET_4.md) , [DeepSeek-V3](questions_answers/Q-A_DEEPSEEK-V3.md) , [Gemini 2.5 Flash](questions_answers/Q-A_GEMINI_2.5_FLASH.md)*

These evaluations focus on methodology understanding and operational behavior without code generation.

---

## Learning the Ropes

### Getting Started

**Configuration Process:**
1.  Configure AI with [AI-PREFERENCES.XML](prompt_formats/software_development/XML/AI-PREFERENCES.XML) as custom instructions.
2.  Share [CORE-PERSONA-FRAMEWORK.json](persona/JSON/CORE-PERSONA-FRAMEWORK.json) + [GUIDE-PERSONA.json](persona/JSON/persona_plugins/GUIDE-PERSONA.json)
3.  Issue command: "Simulate Persona"
4.  Share [METHODOLOGY.XML](prompt_formats/software_development/XML/METHODOLOGY.XML) for planning session.
5.  Collaborate on project structure and phases.
6.  Generate systematic development plan.

**Available Personas:**
*   **[GUIDE-PERSONA.json](persona/JSON/persona_plugins/GUIDE-PERSONA.json)** - Methodology enforcement (prevents vibe coding violations).
*   **[TECDOC-PERSONA.json](persona/JSON/persona_plugins/TECDOC-PERSONA.json)** - Technical documentation specialist.
*   **[R&D-PERSONA.json](persona/JSON/persona_plugins/R&D-PERSONA.json)** - Research scientist with code quality enforcement.
*   **[MURMATE-PERSONA.json](persona/JSON/persona_plugins/MURMATE-PERSONA.json)** - Visual systems and diagram specialist.

*[Read more about the persona framework.](persona/README.PERSONAS.md)*

**Core Documents Reference:**
*   **[AI-PREFERENCES.XML](prompt_formats/software_development/XML/AI-PREFERENCES.XML)** - Behavioral constraints.
*   **[METHODOLOGY.XML](prompt_formats/software_development/XML/METHODOLOGY.XML)** - Technical framework.
*   **[README.XML](prompt_formats/software_development/XML/README.XML)** - Implementation guidance.

**Ask targeted questions:**
*   "How would Phase 0 apply to [project type]?"
*   "What does the 150-line constraint mean for [specific component]?"
*   "How should I structure phases for [project description]?"
*   "Can you help decompose this project using the methodology?"

### Experimental Modifications

**Create Project-Specific Personas:**

Share [CREATE-PERSONA-PLUGIN.json](persona/JSON/CREATE-PERSONA-PLUGIN.json) to generate domain-specific personas from:
*   Project documentation patterns.
*   Codebase architectural philosophies.
*   Domain expert behavioral frameworks.

*[Read more about creating personas.](persona/README.CREATE-PERSONA.md)*
  
**Test constraint variations:**
*   File size limits (100 vs 150 vs 200 lines).
*   Communication constraint adjustments.
*   Phase 0 requirement modifications.
*   Quality gate threshold changes.
*   Persona behavioral pattern modifications.

**Analyze outcomes:**
*   Document behavior changes and development results.
*   Compare debugging time across different approaches.
*   Track architectural compliance over extended sessions.
*   Monitor context retention and behavioral drift.
*   Measure persona consistency enforcement.

**Collaborative refinement:**
Work with your AI to identify improvements based on your context. Treat constraint changes as experiments and measure their impact on collaboration effectiveness, code quality, and development velocity.

**Progress indicators:**
*   Reduced specific violations over time.
*   Consistent file size compliance without reminders.
*   Sustained AI behavioral adherence through extended sessions.
*   Maintained persona consistency across development phases.

---

# Frequently Asked Questions

## Origin & Development

<details>
<summary>What problem led you to create this methodology?</summary>

---

I kept having to restate my preferences and architectural requirements to AI systems. It didn't matter which language or project I was working on - the AI would consistently produce either bloated monolithic code or underdeveloped implementations with issues throughout.

This led me to examine the meta-principles driving code quality and software architecture. I questioned whether pattern matching in AI models might be more effective when focused on underlying software principles rather than surface-level syntax. Since pattern matching is logic-driven and machines fundamentally operate on simple question-answer pairs, I realized that functions with multiple simultaneous questions were overwhelming the system.

The breakthrough came from understanding that everything ultimately transpiles to binary - a series of "can you do this? → yes/no" decisions. This insight shaped my approach: instead of issuing commands, ask focused questions in proper context. Rather than mentally managing complex setups alone, collaborate with AI to devise systematic plans.

---

</details>

<details>
<summary>How did you discover these specific constraints work?</summary>

---

Through extensive trial and error. AI systems will always tend to drift even under constraints, but they're significantly more accurate with structured boundaries than without them. You occasionally need to remind the AI of its role to prevent deviation - like managing a well-intentioned toddler that knows the rules but sometimes pushes boundaries trying to satisfy you.

These tools are far from perfect, but they're effective instruments for software development when properly constrained.

---

</details>

<details>
<summary>What failures or frustrations shaped this approach?</summary>

---

Maintenance hell was the primary driver. I grew tired of responses filled with excessive praise: "You have found the solution!", "You have redefined the laws of physics with your paradigm-shifting script!" This verbose fluff wastes time, tokens, and patience without contributing to productive development.

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

I become genuinely uncomfortable. Once I see things starting to degrade or become tangled, I compulsively need to organize and optimize. Deviation simply isn't an option anymore.

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

In August 2024, I created a RuneLite theme pack, but one of the plugin overlays didn't match my custom layout. I opened a GitHub issue (creating my first GitHub account to do so) requesting a customization option. The response was: "It's not a priority - if you want it, build it yourself."

I used ChatGPT to guide me through forking RuneLite and creating a plugin. This experience sparked intense interest in underlying software principles rather than just syntax.

---

</details>

<details>
<summary>How has your approach evolved over time?</summary>

---

I view development like a book: syntax is the cover, logic is the content itself. Rather than learning syntax structures, I focused on core meta-principles - how software interacts, how logic flows, different algorithm types. I quickly realized everything reduces to the same foundation: question and answer sequences.

Large code structures are essentially chaotic meetings - one coordinator fielding questions and answers from multiple sources, trying to provide correct responses without mix-ups or misinterpretation. If this applies to human communication, it must apply to software principles.

---

</details>

<details>
<summary>What were your biggest mistakes with AI collaboration?</summary>

---

Expecting it to intuitively understand my requirements, provide perfect fixes, be completely honest, and act like a true expert. This was all elaborate roleplay that produced poor code. While fine for single-purpose scripts, it failed completely for scalable codebases.

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

Regardless of project type, anything requiring architecture needs these foundations. You must ensure changes don't negatively impact the entire system. A single line modification in a nested function might work perfectly but cause 300ms boot time regression for all users. 

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