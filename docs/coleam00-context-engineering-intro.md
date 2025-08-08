# Context Engineering: The Future of AI Coding Assistants

**Revolutionize your AI coding with Context Engineering, a comprehensive framework for building robust, consistent, and complex applications with AI.**

[View the original repository](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **10x Better than Prompt Engineering:** Move beyond basic prompts and provide AI with everything it needs to succeed.
*   **Reduce AI Failures:** Equip your AI with the context it needs for consistent, reliable code generation.
*   **Ensure Consistency:** Enforce project-specific patterns, conventions, and documentation standards.
*   **Enable Complex Features:** Empower your AI to handle multi-step implementations with ease.
*   **Self-Correcting:** Leverage built-in validation loops for automated error detection and resolution.

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Quick Start](#quick-start)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering goes beyond traditional prompt engineering, providing a complete system for guiding AI coding assistants.

### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering:** Relies on clever wording and specific phrasing. It is limited to task phrasing, akin to a sticky note.
*   **Context Engineering:** Provides a complete context system, including documentation, examples, rules, patterns, and validation. It's like writing a full screenplay with all details.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses context failures, the leading cause of agent failures.
2.  **Ensures Consistency:** AI adheres to project-specific patterns and conventions.
3.  **Enables Complex Features:** Supports multi-step implementations with proper context.
4.  **Self-Correcting:** Uses validation loops for automated error correction.

## Quick Start

Get started with this template in a few easy steps:

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up project rules (optional)
# Edit CLAUDE.md to add project guidelines

# 3. Add examples (highly recommended)
# Place relevant code examples in the examples/ folder

# 4. Create your feature request
# Edit INITIAL.md with feature requirements

# 5. Generate a Product Requirements Prompt (PRP)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP to implement your feature
# In Claude Code, run:
/execute-prp PRPs/your-feature-name.md
```

## Template Structure

```
context-engineering-intro/
├── .claude/
│   ├── commands/
│   │   ├── generate-prp.md    # Generates PRPs
│   │   └── execute-prp.md     # Executes PRPs
│   └── settings.local.json    # Claude Code permissions
├── PRPs/
│   ├── templates/
│   │   └── prp_base.md       # Base template for PRPs
│   └── EXAMPLE_multi_agent_prp.md  # Example PRP
├── examples/                  # Your code examples
├── CLAUDE.md                 # Global rules for AI assistant
├── INITIAL.md               # Feature request template
├── INITIAL_EXAMPLE.md       # Example feature request
└── README.md                # This file
```

## Step-by-Step Guide

1.  **Set Up Global Rules (CLAUDE.md):** Define project-wide rules for the AI assistant. The template includes:

    *   Project awareness
    *   Code structure
    *   Testing requirements
    *   Style conventions
    *   Documentation standards

2.  **Create Your Initial Feature Request (INITIAL.md):** Describe your desired feature:

    ```markdown
    ## FEATURE:
    [Describe your desired functionality]

    ## EXAMPLES:
    [Reference example files]

    ## DOCUMENTATION:
    [Include relevant documentation]

    ## OTHER CONSIDERATIONS:
    [Note specific requirements or gotchas]
    ```

    See `INITIAL_EXAMPLE.md` for an example.
3.  **Generate the PRP:** Create a comprehensive implementation blueprint using the following command in Claude Code:

    ```bash
    /generate-prp INITIAL.md
    ```

    This command:
    *   Analyzes your codebase.
    *   Searches for relevant documentation.
    *   Creates a PRP in `PRPs/your-feature-name.md`.
4.  **Execute the PRP:** Implement your feature with the following command:

    ```bash
    /execute-prp PRPs/your-feature-name.md
    ```

    The AI coding assistant will:

    *   Read the PRP.
    *   Create an implementation plan.
    *   Execute each step with validation.
    *   Run tests and fix any issues.
    *   Ensure success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

*   **FEATURE**: Be specific about functionality.
    *   **Good:** "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"
*   **EXAMPLES**: Leverage the `examples/` folder, explaining how they should be used.
*   **DOCUMENTATION**: Include API docs, library guides, and server resources.
*   **OTHER CONSIDERATIONS**: Note authentication requirements, rate limits, common pitfalls, and performance.

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase:** Analyzes codebase, searches for implementations and conventions.
2.  **Documentation Gathering:** Fetches API docs, library documentation, and includes gotchas.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan with validation and tests.
4.  **Quality Check:** Scores confidence level and ensures context inclusion.

### How `/execute-prp` Works

1.  **Load Context**: Reads the entire PRP
2.  **Plan**: Creates detailed task list.
3.  **Execute**: Implements each component
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Fixes any issues.
6.  **Complete**: Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a PRP example.

## Using Examples Effectively

The `examples/` folder is crucial for the AI's performance.

### What to Include in Examples

1.  **Code Structure Patterns:** Module organization, import conventions, and class/function patterns.
2.  **Testing Patterns:** Test file structure, mocking approaches, and assertion styles.
3.  **Integration Patterns:** API client implementations, database connections, and authentication flows.
4.  **CLI Patterns:** Argument parsing, output formatting, and error handling.

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

1.  **Be Explicit in INITIAL.md:** Include specific requirements and reference examples.
2.  **Provide Comprehensive Examples:** More examples result in better implementations.
3.  **Use Validation Gates:** PRPs include test commands that must pass.
4.  **Leverage Documentation:** Include API docs and server resources.
5.  **Customize CLAUDE.md:** Add conventions, project-specific rules, and coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)