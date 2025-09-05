# Context Engineering: The Future of AI Coding 🚀

**Context Engineering revolutionizes AI coding, offering a 10x improvement over prompt engineering and a 100x improvement over traditional methods by providing AI assistants with comprehensive context.**

[Go to the original repository](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Comprehensive Context:** Provide AI with the information it needs to succeed.
*   **Reduced AI Failures:** Minimize errors by giving your AI the full picture.
*   **Consistency & Standardization:** Enforce project patterns and conventions effortlessly.
*   **Complex Feature Implementation:** Enables multi-step and intricate feature development.
*   **Self-Correction:** Integrated validation loops allow AI to fix its own mistakes.

## Quick Start Guide

1.  **Clone the Template:**

    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```

2.  **(Optional) Set Project Rules:** Customize `CLAUDE.md` with your project-specific guidelines.

3.  **Add Examples:** Place relevant code examples in the `examples/` directory to guide your AI assistant.

4.  **Create Feature Request:** Define your feature requirements in `INITIAL.md`.

5.  **Generate PRP (Product Requirements Prompt):** Run the following command in Claude Code:

    ```bash
    /generate-prp INITIAL.md
    ```

6.  **Execute the PRP:** Implement your feature using Claude Code:

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

## What is Context Engineering?

Context Engineering moves beyond prompt engineering by providing a complete system for giving comprehensive context.

### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering:** Focuses on clever phrasing and wording; acts like a sticky note.
*   **Context Engineering:** Provides comprehensive context through documentation, examples, rules, and validation; acts like a full screenplay.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses context-related failures.
2.  **Ensures Consistency:** Follows project patterns and conventions.
3.  **Enables Complex Features:** Manages multi-step implementations.
4.  **Self-Correcting:** Validation loops improve code quality.

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
├── INITIAL.md               # Template for feature requests
├── INITIAL_EXAMPLE.md       # Example feature request
└── README.md                # This file
```

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

Customize `CLAUDE.md` with project-wide rules for your AI assistant. It includes:

*   Project awareness, code structure, testing requirements, style, and documentation standards.

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to describe the desired feature:

```markdown
## FEATURE:
[Describe what you want to build - be specific]

## EXAMPLES:
[List relevant examples and how they should be used]

## DOCUMENTATION:
[Include links to documentation, APIs, or server resources]

## OTHER CONSIDERATIONS:
[Include any gotchas or specific requirements]
```

See `INITIAL_EXAMPLE.md` for an example.

### 3. Generate the PRP

PRPs (Product Requirements Prompts) provide detailed implementation blueprints:

*   Comprehensive context, implementation steps with validation, error handling, and test requirements.

Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

This will:

1.  Analyze your feature request.
2.  Research your codebase for patterns.
3.  Search for relevant documentation.
4.  Create a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI coding assistant will:

1.  Read the entire PRP.
2.  Create a detailed implementation plan.
3.  Execute each step with validation.
4.  Run tests and fix issues.
5.  Ensure all requirements are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

*   **FEATURE:** Be specific and comprehensive.
*   **EXAMPLES:** Leverage the `examples/` folder. Reference specific files and patterns to follow.
*   **DOCUMENTATION:** Include all relevant resources (API docs, library guides, database schemas).
*   **OTHER CONSIDERATIONS:** Capture important details (authentication, rate limits, performance).

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase:** Analyzes your codebase for patterns and conventions.
2.  **Documentation Gathering:** Fetches relevant API and library documentation.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan with validation.
4.  **Quality Check:** Scores confidence and ensures context is included.

### How `/execute-prp` Works

1.  Load Context
2.  Plan
3.  Execute
4.  Validate
5.  Iterate
6.  Complete

## Using Examples Effectively

The `examples/` directory is crucial for success.

### What to Include in Examples

1.  Code Structure Patterns
2.  Testing Patterns
3.  Integration Patterns
4.  CLI Patterns

### Example Structure

```
examples/
├── README.md           # Explains each example
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

### 1. Be Explicit in INITIAL.md
*   Include specific requirements and constraints.
*   Reference examples liberally.

### 2. Provide Comprehensive Examples
*   More examples lead to better implementations.
*   Show both what to do AND what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates
*   PRPs include test commands.
*   AI will iterate until all validations succeed.

### 4. Leverage Documentation
*   Include official API docs.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md
*   Add project-specific rules and coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)