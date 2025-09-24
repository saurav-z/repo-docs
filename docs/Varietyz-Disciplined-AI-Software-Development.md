<div align="center">
  <img src="https://banes-lab.com/assets/images/banes_lab/700px_Main_Animated.gif" width="70" alt="Animated Logo"/>
</div>

# Disciplined AI Software Development: A Structured Approach

**Tired of AI code bloat and architectural drift?** [Disciplined AI Software Development](https://github.com/Varietyz/Disciplined-AI-Software-Development) offers a structured methodology for building robust and maintainable AI-assisted software. This approach, licensed under CC BY-SA 4.0, helps you overcome common challenges like code bloat and context dilution by enforcing systematic constraints and collaborative planning.

## Key Features:

*   **AI Behavioral Configuration:** Define custom instructions and persona frameworks for consistent AI behavior.
*   **Collaborative Planning:** Structure projects with AI using a shared methodology for clear scope and dependencies.
*   **Systematic Implementation:** Enforce file size limits and modular boundaries for manageable components.
*   **Data-Driven Iteration:** Utilize a benchmarking suite for performance optimization based on empirical data.
*   **Comprehensive Documentation:** Detailed Q&A documents and workflow visualizations for easy onboarding.

## Core Principles:

This methodology leverages four key stages to guide your AI-powered development:

### Stage 1: AI Behavioral Configuration

*   **Configure Custom Instructions:** Set up `AI-PREFERENCES.XML` to establish behavioral constraints and uncertainty flagging.
*   **Load Persona Framework (Recommended):** Use pre-built or custom personas (e.g., `GUIDE-PERSONA.json` for methodology enforcement) to maintain consistency.
*   **Activate Persona (Recommended):** Command the AI to "Simulate Persona" to initialize the framework.

### Stage 2: Collaborative Planning

*   Share `METHODOLOGY.XML` with the AI to collaboratively define project scope, components, and phases.
*   Structure tasks based on logical progression with measurable checkpoints.

### Stage 3: Systematic Implementation

*   Implement components one-by-one, focusing on a request like, "Can you implement [specific component]?"
*   Maintain file size under 150 lines for focused implementation and efficient context management.
*   Follow an implementation flow of "Request specific component → AI processes → Validate → Benchmark → Continue".

### Stage 4: Data-Driven Iteration

*   Utilize a benchmarking suite (built first) to gather performance data.
*   Feed data back to the AI for optimization decisions based on measurements.

## Example Projects:

*   [Discord Bot Template](https://github.com/Varietyz/discord-js-bot-template)
*   [PhiCode Runtime](https://github.com/Varietyz/phicode-runtime)
*   [PhiPipe](https://github.com/Varietyz/PhiPipe)

## Implementation Steps:

1.  Configure AI with `AI-PREFERENCES.XML` as custom instructions.
2.  RECOMMENDED: Share `CORE-PERSONA-FRAMEWORK.json` + selected `PERSONA.json`.
3.  RECOMMENDED: Issue command: "Simulate Persona".
4.  Share `METHODOLOGY.XML` for planning.
5.  Collaborate on project structure and phases.
6.  Generate a systematic development plan.

## Additional Resources:

*   **Project State Extraction Tool:** Use the Python script `scripts/project_extract.py` to generate structured snapshots of your codebase for AI analysis.  Configure options to specify what you want extracted.
*   **Prompt Formats:** Explore different prompt formats for different use cases at `prompt_formats/`.
*   **Model Evaluation:** View the Q&A Documentation for each AI model that was tested.
*   **Getting Started/Learning the Ropes:** Read more about the core workflow as well as experimental modification options.
*   **Frequently Asked Questions:** Get answers to common questions about the methodology.

## Workflow Visualization

```mermaid
graph TD
    A[Configure AI (Custom Instructions, Persona)] --> B(Collaborative Planning);
    B --> C{Systematic Implementation (Component by Component)};
    C --> D{Validation & Benchmarking};
    D --> C;
    C --> E{Data-Driven Iteration};
    E --> C;
```