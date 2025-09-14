# Disciplined AI Software Development: Build Robust AI-Powered Software with Structure 

This methodology provides a structured, collaborative approach to AI-driven software development, minimizing code bloat, architectural drift, and debugging time.  For more details, visit the original repository:  [Disciplined-AI-Software-Development](https://github.com/Varietyz/Disciplined-AI-Software-Development)

Key Features:

*   **Structured Workflow:**  A four-stage process with defined constraints and validation steps, ensuring consistent architecture.
*   **Context Management:** Employs file size limits (≤150 lines) and focused tasks to prevent AI from handling multiple concerns simultaneously.
*   **Data-Driven Decisions:** Leverages benchmarking and performance data for optimization, moving away from subjective assessments.
*   **Systematic Constraints:** Implements architectural checkpoints, file size limits, and dependency gates to promote consistent behavior.
*   **Collaborative Planning:**  Facilitates collaboration between you and the AI to define scope, identify dependencies, and structure phases.
*   **Automated Quality Assurance:** Includes tools for performance regression detection, architectural principle validation, and dependency boundary verification.
*   **Project Extraction Tool:** Uses a project extraction tool for structured snapshots of your codebase for easy sharing and architectural compliance.

## Methodology Overview

This methodology tackles common challenges in AI-assisted software development by establishing a structured, collaborative approach. It emphasizes upfront planning and systematic constraints to reduce debugging time and ensure code quality.

### The Core Problem

Traditional AI development often suffers from:

*   Unstructured and monolithic code
*   Architectural inconsistencies across sessions
*   Context dilution and output drift
*   Increased debugging time

### The Solution: A Four-Stage Approach

The methodology mitigates these issues through four iterative stages. Each stage incorporates systematic constraints, validation checkpoints, and leverages empirical data:

1.  **AI Configuration:** Set up the AI model with `AI-PREFERENCES.XML` to establish behavioral constraints.
2.  **Collaborative Planning:** Share `METHODOLOGY.XML` to jointly define scope, identify components, and plan phases.
3.  **Systematic Implementation:** Implement components systematically, with file size limits (≤150 lines) and focused objectives.
4.  **Data-Driven Iteration:**  Use benchmarking data to inform optimization decisions, improving performance.

## Example Projects

The methodology has been successfully applied to diverse projects, including:

*   **[Discord Bot Template](https://github.com/Varietyz/discord-js-bot-template)**:  A production-ready bot foundation with plugin architecture and security features.
*   **[PhiCode Runtime](https://github.com/Varietyz/phicode-runtime)**: A programming language runtime engine with transpilation and caching capabilities.
*   **[PhiPipe](https://github.com/Varietyz/PhiPipe)**: A CI/CD regression detection system for statistical analysis and integration.

## Getting Started

1.  **Configure AI:** Use `AI-PREFERENCES.XML` as custom instructions.
2.  **Collaborate:** Share `METHODOLOGY.XML` to structure your project.
3.  **Plan:** Work together to define phases and tasks.
4.  **Implement:** Follow the systematic implementation process, phase by phase.

### Implementation Steps

*   **Setup**: Follow the configurations steps as mentioned in the README.
*   **Execution**: Work through phases sequentially, implementing one component per interaction.
*   **Quality Assurance**: Use the performance regression detection, architecture principles validation and file size compliance checking.

For detailed information on the prompt formats, refer to the `prompt_formats` directory in the repository.

## Tools & Resources

*   **Project Extraction Tool:** Use the included `project_extract.py` script to generate structured project snapshots.
*   **Q&A Documentation:**  Explore detailed Q&A for different AI models in the [questions\_answers/](questions_answers/) directory.
*   **Learning the Ropes**: Share core and persona documents such as  `CORE-PERSONA-FRAMEWORK.json`, `GUIDE-PERSONA.json`, `AI-PREFERENCES.XML`, and `METHODOLOGY.XML` with your AI model.

## Frequently Asked Questions

*   **Origin & Development:**  Addresses the issues and inspirations that led to the methodology's creation.
*   **Personal Practice:**  Provides insights into the author's adherence and experience with the methodology.
*   **AI Development Journey:**  Details the evolution of the methodology.
*   **Methodology Specifics:**  Explains the rationale behind the key constraints and requirements.
*   **Practical Implementation:** Covers adaptability to different project types and advice for new users.

## Workflow Visualization

(See the original README for the mermaid diagram.)

---

This README provides a concise overview of the Disciplined AI Software Development methodology, highlighting its benefits and practical application. By following its principles, you can improve the quality, maintainability, and efficiency of your AI-powered software projects.