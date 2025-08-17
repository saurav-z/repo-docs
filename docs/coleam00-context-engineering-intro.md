# Context Engineering Template: Revolutionize AI Coding with Comprehensive Context

**Tired of unreliable AI coding assistants?  Context Engineering provides a robust system for providing comprehensive context to AI, leading to superior code generation and fewer errors!** ([View Original Repo](https://github.com/coleam00/context-engineering-intro))

## Key Features:

*   **Enhanced Accuracy:** Dramatically reduces AI failures by providing complete context.
*   **Consistency Guaranteed:** Ensures the AI adheres to your project's patterns and conventions.
*   **Complex Feature Enablement:** Empowers AI to handle multi-step implementations with ease.
*   **Self-Correcting Capabilities:** Uses validation loops to allow the AI to fix its own mistakes.
*   **PRP Workflow:** Automates the creation and execution of comprehensive implementation blueprints.

## Getting Started

1.  **Clone the Template:**

    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```

2.  **Set Up Your Project Rules:** (Optional, but recommended)

    *   Edit `CLAUDE.md` to define project-specific guidelines, coding style, testing requirements, and documentation standards.

3.  **Add Code Examples:** (Essential for Success)

    *   Place relevant code examples in the `examples/` folder to guide the AI.

4.  **Create Your Initial Feature Request:**

    *   Edit `INITIAL.md` to describe the desired feature or functionality.

5.  **Generate a Product Requirements Prompt (PRP):**

    *   Use the custom command within Claude Code: `/generate-prp INITIAL.md`

6.  **Execute the PRP to Implement Your Feature:**

    *   Use the custom command within Claude Code: `/execute-prp PRPs/your-feature-name.md`

## Template Structure Overview

```
context-engineering-intro/
├── .claude/         # Custom Claude Code commands and settings
├── PRPs/            # Generated Product Requirements Prompts
├── examples/        # Your code examples (critical!)
├── CLAUDE.md        # Global rules for AI assistant
├── INITIAL.md       # Template for feature requests
└── README.md        # This file
```

## Context Engineering vs. Prompt Engineering

Context Engineering offers a superior approach to AI coding assistance:

*   **Prompt Engineering:** Relies on clever phrasing, limiting the AI's understanding. It's like using a sticky note.
*   **Context Engineering:** Provides a complete system with comprehensive context, including documentation, examples, rules, and validation. It's like writing a full screenplay with all the details.

## Step-by-Step Guide

1.  **Define Global Rules (CLAUDE.md):**  Customize project-wide rules for the AI assistant, including code structure, testing, and style conventions.

2.  **Create Feature Requests (INITIAL.md):**  Clearly specify the desired functionality, required examples, documentation links, and other important considerations.  See `INITIAL_EXAMPLE.md` for a template and example.

3.  **Generate Product Requirements Prompt (PRP):**  The `/generate-prp` command reads your feature request, researches the codebase, gathers relevant documentation, and creates a comprehensive PRP in the `PRPs/` directory.  PRPs are similar to PRDs, but are tailored for AI assistants.

4.  **Execute the PRP:** The `/execute-prp` command instructs the AI to implement the feature based on the PRP, creating a detailed implementation plan, executing steps with validation, running tests, and iterating until all requirements are met.

## Writing Effective Feature Requests (INITIAL.md)

*   **FEATURE:** Be specific and comprehensive about the desired functionality and requirements.
*   **EXAMPLES:**  Reference code examples from the `examples/` folder and explain how they should be used.
*   **DOCUMENTATION:** Include links to relevant APIs, libraries, and documentation resources.
*   **OTHER CONSIDERATIONS:** Note authentication, rate limits, common pitfalls, and performance requirements.

## The PRP Workflow: How It Works

*   `/generate-prp` analyzes the codebase for patterns, gathers documentation, and creates a step-by-step implementation plan with validation and testing.
*   `/execute-prp` loads the PRP, creates a task list, implements components, validates the code, and iterates until the feature is complete.

## Maximizing Success with Examples

The `examples/` folder is **crucial** for guiding the AI.  Include examples that demonstrate:

*   Code Structure and Organization
*   Testing Patterns (unit tests, mocking)
*   Integration Patterns (API clients, database connections)
*   CLI Patterns (argument parsing, output formatting)

## Best Practices for Context Engineering

1.  **Be Explicit in INITIAL.md:** Clearly define requirements and constraints.
2.  **Provide Comprehensive Examples:**  The more examples, the better. Show what to do and what to avoid.
3.  **Utilize Validation Gates:**  PRPs include test commands to ensure working code.
4.  **Leverage Documentation:**  Include API docs and relevant resources.
5.  **Customize CLAUDE.md:**  Define your project's conventions and standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)