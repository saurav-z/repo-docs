# Context Engineering: Revolutionize AI Coding with Comprehensive Context

**Unlock the power of AI coding assistants by providing them with comprehensive context, leading to faster, more accurate, and consistent results.**  Learn how to apply context engineering with this project. ([Original Repo](https://github.com/coleam00/context-engineering-intro))

## Key Features:

*   **Comprehensive Context:** Provide AI with the information it needs to succeed.
*   **Project-Specific Rules:** Customize project guidelines and standards.
*   **Example-Driven:** Leverage code examples for consistent implementations.
*   **Automated Workflow:** Generate Product Requirements Prompts (PRPs) and execute them for feature implementation.
*   **Validation & Iteration:** Ensure working code with built-in validation and automated correction.

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
    *   [Prompt Engineering vs. Context Engineering](#prompt-engineering-vs-context-engineering)
    *   [Why Context Engineering Matters](#why-context-engineering-matters)
*   [Getting Started: Quick Start](#getting-started-quick-start)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
    *   [1. Set Up Global Rules (CLAUDE.md)](#1-set-up-global-rules-claudemd)
    *   [2. Create Your Initial Feature Request](#2-create-your-initial-feature-request)
    *   [3. Generate the PRP](#3-generate-the-prp)
    *   [4. Execute the PRP](#4-execute-the-prp)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
    *   [Key Sections Explained](#key-sections-explained)
*   [The PRP Workflow](#the-prp-workflow)
    *   [How /generate-prp Works](#how-generate-prp-works)
    *   [How /execute-prp Works](#how-execute-prp-works)
*   [Using Examples Effectively](#using-examples-effectively)
    *   [What to Include in Examples](#what-to-include-in-examples)
    *   [Example Structure](#example-structure)
*   [Best Practices](#best-practices)
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering provides a more robust and effective way to utilize AI coding assistants, in contrast to traditional prompt engineering.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**  Focuses on crafting clever prompts.

**Context Engineering:** Provides a complete system for supplying all necessary information to the AI.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** By providing thorough context.
2.  **Ensures Consistency:** Aligns AI output with project standards.
3.  **Enables Complex Features:** Supports multi-step implementations.
4.  **Self-Correcting:** Uses validation to fix errors.

## Getting Started: Quick Start

1.  **Clone the template:**

    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```

2.  **Set up project rules (optional):** Edit `CLAUDE.md`.
3.  **Add examples (recommended):** Place code in the `examples/` folder.
4.  **Create your initial feature request:** Edit `INITIAL.md`.
5.  **Generate a PRP:** Run `/generate-prp INITIAL.md` in Claude Code.
6.  **Execute the PRP:** Run `/execute-prp PRPs/your-feature-name.md` in Claude Code.

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

### 1. Set Up Global Rules (CLAUDE.md)

Customize `CLAUDE.md` with your project-wide rules, including:

*   Project awareness
*   Code structure
*   Testing requirements
*   Style conventions
*   Documentation standards

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to describe your feature:

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

See `INITIAL_EXAMPLE.md` for a reference.

### 3. Generate the PRP

Create a Product Requirements Prompt (PRP) by running:

```bash
/generate-prp INITIAL.md
```

This command:

1.  Reads your feature request.
2.  Researches your codebase.
3.  Generates a comprehensive PRP.

### 4. Execute the PRP

Implement your feature with:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI will:

1.  Read the PRP context.
2.  Create an implementation plan.
3.  Execute each step.
4.  Run tests and fix issues.
5.  Ensure success.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE:** Be specific and detailed.

**EXAMPLES:** Use code examples from the `examples/` folder, specifying patterns.

**DOCUMENTATION:** Include all relevant documentation links.

**OTHER CONSIDERATIONS:** Add crucial details such as rate limits and authentication requirements.

## The PRP Workflow

### How /generate-prp Works

1.  **Research:** Analyzes codebase and identifies patterns.
2.  **Documentation:** Gathers relevant documentation.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan with validations.
4.  **Quality Check:** Evaluates the PRP's completeness.

### How /execute-prp Works

1.  Loads PRP context.
2.  Creates a detailed task list.
3.  Implements the components.
4.  Validates the implementation.
5.  Iterates to fix issues.
6.  Ensures all requirements are met.

## Using Examples Effectively

The `examples/` folder is vital. Provide examples of:

### What to Include in Examples

1.  Code Structure Patterns
2.  Testing Patterns
3.  Integration Patterns
4.  CLI Patterns

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

### 1. Be Explicit in INITIAL.md

### 2. Provide Comprehensive Examples

### 3. Use Validation Gates

### 4. Leverage Documentation

### 5. Customize CLAUDE.md

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)