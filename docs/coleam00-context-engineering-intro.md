# Context Engineering: Build AI Applications with Confidence ([GitHub Repo](https://github.com/coleam00/context-engineering-intro))

**Unlock the power of AI by engineering comprehensive context, leading to more reliable and complex applications, outpacing traditional prompt engineering.**

This template provides a complete framework for Context Engineering, enabling you to build robust AI applications by providing AI assistants with the necessary context for end-to-end feature implementation. This approach is designed to reduce AI failures, ensure consistency, and handle complex features effectively.

## Key Features

*   **Comprehensive Context:** Provide AI with documentation, examples, rules, and validation to guide implementation.
*   **Simplified Workflow:** Streamlined steps for feature request, PRP generation, and execution.
*   **Customizable:** Adaptable template with options for project-specific rules and examples.
*   **PRP-Driven Implementation:** Generate Product Requirements Prompts (PRPs) that guide AI assistants through implementation.
*   **Validation & Iteration:** PRPs incorporate testing and validation gates to ensure code quality and functionality.

## Getting Started

Follow these steps to begin Context Engineering:

1.  **Clone the Template:**

    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```
2.  **Configure Project Rules (Optional):**
    *   Edit `CLAUDE.md` to define project-specific coding standards, testing requirements, and more.
3.  **Add Code Examples (Highly Recommended):**
    *   Place code examples in the `examples/` folder to demonstrate coding patterns.
    *   Use a clear `examples/README.md` to guide usage.
4.  **Create Feature Request:**
    *   Describe the desired feature in `INITIAL.md`, detailing requirements, examples, and documentation.
    *   Use `INITIAL_EXAMPLE.md` for guidance.
5.  **Generate PRP:**
    *   Use Claude Code to generate a PRP (Product Requirements Prompt):

        ```bash
        /generate-prp INITIAL.md
        ```

        This command creates a comprehensive implementation blueprint.
6.  **Execute PRP:**
    *   Run the PRP to implement the feature:

        ```bash
        /execute-prp PRPs/your-feature-name.md
        ```

        The AI assistant will use the context to create and execute a plan to implement the requested feature.

## Template Structure

```
context-engineering-intro/
├── .claude/
│   ├── commands/
│   │   ├── generate-prp.md    # Generates comprehensive PRPs
│   │   └── execute-prp.md     # Executes PRPs to implement features
│   └── settings.local.json    # Claude Code permissions
├── PRPs/
│   ├── templates/
│   │   └── prp_base.md       # Base template for PRPs
│   └── EXAMPLE_multi_agent_prp.md  # Example of a complete PRP
├── examples/                  # Your code examples (critical!)
├── CLAUDE.md                 # Global rules for AI assistant
├── INITIAL.md               # Template for feature requests
├── INITIAL_EXAMPLE.md       # Example feature request
└── README.md                # This file
```

## Detailed Guide

### What is Context Engineering?

Context Engineering surpasses traditional prompt engineering by providing AI with a complete system of context:

*   **Prompt Engineering:** Relies on carefully worded prompts, limiting implementation.
*   **Context Engineering:** Uses documentation, examples, rules, and validation for comprehensive instructions.

### Advantages of Context Engineering

1.  **Reduces AI Failures:** Provides necessary context, leading to fewer errors.
2.  **Ensures Consistency:** AI follows your project's coding standards.
3.  **Enables Complex Features:** Handles multi-step features efficiently.
4.  **Self-Correcting:** Validation loops allow AI to fix its mistakes.

### Step-by-Step Instructions

1.  **Setting Global Rules (CLAUDE.md):**
    *   Define project-wide rules for code structure, testing, style, and documentation.
2.  **Creating Feature Requests (INITIAL.md):**
    *   Use a structured format: `FEATURE`, `EXAMPLES`, `DOCUMENTATION`, and `OTHER CONSIDERATIONS`.
    *   Be specific and include all necessary context.
3.  **Generating PRPs:**
    *   The `/generate-prp` command:
        *   Analyzes your codebase.
        *   Gathers documentation.
        *   Creates a step-by-step implementation plan with validation.
4.  **Executing PRPs:**
    *   The `/execute-prp` command guides the AI through implementation, including testing and iteration.

### Effective Writing in `INITIAL.md`

*   **FEATURE:** Describe the functionality and requirements precisely.
*   **EXAMPLES:** Link to relevant files in the `examples/` folder and explain how they should be used.
*   **DOCUMENTATION:** Include links to documentation and other relevant resources.
*   **OTHER CONSIDERATIONS:** List any requirements, restrictions, or common pitfalls.

### The PRP Workflow

*   `/generate-prp`: Analyzes the codebase, gathers documentation, and creates an implementation plan.
*   `/execute-prp`: Loads the PRP context, plans the tasks, executes each step, validates, iterates, and ensures all requirements are met.

### Using Examples Effectively

The `examples/` folder is essential for providing the AI assistant with patterns to follow:

*   **Code Structure:** Module organization, import conventions.
*   **Testing:** Test file structure, mocking, and assertions.
*   **Integration:** API client implementations, database connections.
*   **CLI:** Argument parsing, output formatting, and error handling.
*   Include a clear `examples/README.md`.

### Best Practices

1.  Be specific in `INITIAL.md`.
2.  Provide comprehensive examples.
3.  Use validation gates within PRPs.
4.  Leverage documentation.
5.  Customize `CLAUDE.md`.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)