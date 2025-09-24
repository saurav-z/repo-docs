# Context Engineering: Revolutionizing AI Coding with Comprehensive Context

**Stop wrestling with prompt engineering and embrace Context Engineering, the future of AI-assisted coding!**

[View the original repo](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Comprehensive Context:** Provide AI assistants with all the information they need for end-to-end feature implementation.
*   **Reduced AI Failures:** Minimize errors by giving your AI assistant a complete understanding of your project.
*   **Consistent Code Quality:** Ensure adherence to your project's style, conventions, and standards.
*   **Complex Feature Implementation:** Empower AI to handle multi-step tasks through detailed planning and validation.
*   **Self-Correcting Capabilities:** Leverage validation loops for automated error detection and correction.

## Getting Started

Follow these steps to begin using the Context Engineering template:

1.  **Clone the Template:**
    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```

2.  **Configure Project Rules (Optional):** Customize `CLAUDE.md` to define your project's specific coding guidelines.
3.  **Add Code Examples (Highly Recommended):** Place relevant code examples in the `examples/` folder to guide the AI assistant.
4.  **Create a Feature Request:** Describe your desired feature in `INITIAL.md`.
5.  **Generate a Product Requirements Prompt (PRP):** Use Claude Code to create a comprehensive implementation plan.
    ```bash
    /generate-prp INITIAL.md
    ```

6.  **Execute the PRP:** Implement your feature using the generated plan.
    ```bash
    /execute-prp PRPs/your-feature-name.md
    ```

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering is a revolutionary approach to AI-assisted coding that surpasses the limitations of prompt engineering. It focuses on providing a complete and comprehensive context for the AI, enabling it to understand and implement complex features effectively.

### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering:** Relies on clever phrasing and is limited to the way you phrase the task.
*   **Context Engineering:** Provides a complete system including documentation, examples, rules, patterns, and validation.

### Why Context Engineering Matters

1.  **Reduced AI Failures:** Minimizes agent failures by ensuring the AI has the necessary context.
2.  **Consistency:** Ensures AI follows your project's patterns and conventions.
3.  **Complex Features:** Enables AI to handle multi-step implementations with correct context.
4.  **Self-Correction:** Utilizes validation loops to allow the AI to fix its own mistakes.

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

## Step-by-Step Guide

1.  **Set Up Global Rules (CLAUDE.md):**
    *   Define project-wide rules for your AI assistant, including:
        *   Project awareness
        *   Code structure
        *   Testing requirements
        *   Style conventions
        *   Documentation standards
    *   Customize the provided template or create your own.

2.  **Create Your Initial Feature Request:**
    *   Edit `INITIAL.md` to specify what you want to build.
    ```markdown
    ## FEATURE:
    [Describe what you want to build - be specific about functionality and requirements]

    ## EXAMPLES:
    [List any example files in the examples/ folder and explain how they should be used]

    ## DOCUMENTATION:
    [Include links to relevant documentation, APIs, or MCP server resources]

    ## OTHER CONSIDERATIONS:
    [Mention any gotchas, specific requirements, or things AI assistants commonly miss]
    ```
    *   Refer to `INITIAL_EXAMPLE.md` for a complete example.

3.  **Generate the PRP:**
    *   PRPs (Product Requirements Prompts) are detailed implementation blueprints.
    *   Run in Claude Code:
        ```bash
        /generate-prp INITIAL.md
        ```
        *   This command:
            *   Reads your feature request
            *   Researches the codebase
            *   Searches for relevant documentation
            *   Creates a PRP in `PRPs/your-feature-name.md`

4.  **Execute the PRP:**
    *   Implement your feature using the generated PRP.
    ```bash
    /execute-prp PRPs/your-feature-name.md
    ```
    *   The AI assistant will:
        *   Read all context from the PRP
        *   Create a detailed implementation plan
        *   Execute each step
        *   Run tests and fix any issues
        *   Ensure all success criteria are met

## Writing Effective INITIAL.md Files

### Key Sections Explained

*   **FEATURE:**  Be specific and comprehensive about the functionality and requirements.
    *   **Example:**  Instead of "Build a web scraper," use "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL."
*   **EXAMPLES:** Leverage code examples to guide the AI assistant.
    *   Reference specific files and patterns.
    *   Explain what aspects should be followed.
*   **DOCUMENTATION:** Include all relevant resources.
    *   API documentation URLs.
    *   Library guides.
    *   Server documentation.
    *   Database schemas.
*   **OTHER CONSIDERATIONS:** Capture important details.
    *   Authentication requirements
    *   Rate limits or quotas
    *   Common pitfalls
    *   Performance requirements

## The PRP Workflow

### How /generate-prp Works

The `/generate-prp` command:

1.  **Research Phase:** Analyzes your codebase, searches for similar implementations, and identifies conventions.
2.  **Documentation Gathering:** Fetches relevant API docs, and includes library documentation.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan, validation gates, and test requirements.
4.  **Quality Check:** Scores a confidence level (1-10) and ensures all context is included.

### How /execute-prp Works

The `/execute-prp` command:

1.  **Load Context:** Reads the entire PRP.
2.  **Plan:** Creates a detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes any issues found.
6.  **Complete:** Ensures all requirements are met.

## Using Examples Effectively

The `examples/` folder is crucial for AI success. AI coding assistants perform much better when they can see patterns to follow.

### What to Include in Examples

1.  **Code Structure Patterns:** Module organization, import conventions, class/function patterns.
2.  **Testing Patterns:** Test file structure, mocking approaches, assertion styles.
3.  **Integration Patterns:** API client implementations, database connections, authentication flows.
4.  **CLI Patterns:** Argument parsing, output formatting, error handling.

### Example Structure

```
examples/
├── README.md           # Explains what each example demonstrates
├── cli.py             # CLI implementation pattern
├── agent/             # Agent architecture patterns
│   ├── agent.py      # Agent creation pattern
│   ├── tools.py      # Tool implementation pattern
│   └── providers.py  # Multi-provider pattern
└── tests/            # Testing patterns
    ├── test_agent.py # Unit test patterns
    └── conftest.py   # Pytest configuration
```

## Best Practices

1.  **Be Explicit in INITIAL.md:** Include specific requirements, constraints, and reference examples.
2.  **Provide Comprehensive Examples:**  More examples lead to better implementations, including error handling patterns.
3.  **Use Validation Gates:**  PRPs incorporate test commands that must pass.
4.  **Leverage Documentation:** Include API docs, server resources, and specific documentation sections.
5.  **Customize CLAUDE.md:** Add your conventions, project-specific rules, and coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)