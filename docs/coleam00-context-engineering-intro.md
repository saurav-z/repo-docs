# Context Engineering: Supercharge Your AI Coding with Comprehensive Context

Context Engineering is a revolutionary approach to AI coding, offering a superior experience compared to traditional prompt engineering. This template provides a robust framework for building AI-powered coding workflows that deliver reliable results.  **[View the original repository](https://github.com/coleam00/context-engineering-intro).**

**Key Features:**

*   **Comprehensive Context:** Provide AI assistants with all the necessary information to complete tasks end-to-end.
*   **Reduced AI Failures:** Significantly decrease the frequency of AI errors by addressing context-related issues.
*   **Consistent Output:** Ensure that your AI assistant consistently follows your project's patterns and conventions.
*   **Complex Feature Implementation:** Enable your AI to handle multi-step tasks with ease and accuracy.
*   **Self-Correction:** Implement validation loops to allow your AI to fix its own mistakes and improve its outputs.

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Quick Start Guide](#quick-start-guide)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective `INITIAL.md` Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering is a paradigm shift from prompt engineering, focusing on providing a complete system of context.

**Prompt Engineering vs. Context Engineering**

*   **Prompt Engineering:** Relies on clever phrasing and specific prompts, limiting task scope.
*   **Context Engineering:** Provides comprehensive context, including documentation, examples, rules, and validation.

**Why Context Engineering Matters**

*   **Reduces AI Failures:** Addresses the primary source of agent errors – context deficiencies.
*   **Ensures Consistency:** Enforces project-specific patterns and conventions.
*   **Enables Complex Features:** Allows AI to handle multi-step implementations.
*   **Self-Correcting:** Utilizes validation loops for AI-driven error correction.

## Quick Start Guide

1.  **Clone the Template:**
    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```
2.  **Set Up Project Rules (Optional):**
    Customize `CLAUDE.md` with your project guidelines.
3.  **Add Examples (Highly Recommended):**
    Place relevant code examples in the `examples/` folder.
4.  **Create Feature Request:**
    Edit `INITIAL.md` with your feature requirements.
5.  **Generate Product Requirements Prompt (PRP):**
    In Claude Code, run: `/generate-prp INITIAL.md`
6.  **Execute the PRP:**
    In Claude Code, run: `/execute-prp PRPs/your-feature-name.md`

## Template Structure

```
context-engineering-intro/
├── .claude/          # Claude Code commands & settings
├── PRPs/              # Generated Product Requirements Prompts
├── examples/          # Code examples for AI to learn from
├── CLAUDE.md          # Global project rules for the AI
├── INITIAL.md         # Template for feature requests
└── README.md          # This file
```

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

`CLAUDE.md` contains project-wide rules for the AI assistant, including:

*   Project awareness
*   Code structure guidelines
*   Testing requirements
*   Style and formatting conventions
*   Documentation standards

**Customize the template provided for your project.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to describe what you want to build.  Use the provided template:

```markdown
## FEATURE:
[Describe feature requirements and functionality]

## EXAMPLES:
[List example files and explain their usage]

## DOCUMENTATION:
[Include links to relevant resources]

## OTHER CONSIDERATIONS:
[Mention gotchas, specific requirements, etc.]
```

**See `INITIAL_EXAMPLE.md` for a complete example.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints for the AI assistant.

Run in Claude Code:
```bash
/generate-prp INITIAL.md
```

This command:

1.  Reads your feature request
2.  Researches the codebase for patterns
3.  Searches for relevant documentation
4.  Creates a comprehensive PRP in `PRPs/your-feature-name.md`

### 4. Execute the PRP

Execute the generated PRP to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read the entire PRP.
2.  Create an implementation plan.
3.  Execute steps with validation.
4.  Run tests and fix issues.
5.  Ensure all success criteria are met.

## Writing Effective `INITIAL.md` Files

### Key Sections Explained

**FEATURE**: Be specific and comprehensive, defining feature requirements and functionality.
**EXAMPLES**: Reference and describe the usage of relevant code patterns found in the `examples/` folder.
**DOCUMENTATION**: Include all relevant resources such as API documentation URLs, and library guides.
**OTHER CONSIDERATIONS**: Mention authentication requirements, rate limits, common pitfalls, and performance expectations.

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase:** Analyzes your codebase for patterns and identifies conventions.
2.  **Documentation Gathering:** Fetches relevant API docs and includes gotchas and quirks.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan, includes validation gates and test requirements.
4.  **Quality Check:** Scores confidence level and ensures all context is included.

### How `/execute-prp` Works

1.  Load Context
2.  Plan
3.  Execute
4.  Validate
5.  Iterate
6.  Complete

## Using Examples Effectively

The `examples/` folder is crucial for the AI's success, enabling it to learn and follow patterns.

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

1.  **Be Explicit in `INITIAL.md`:** Clearly state requirements and reference examples.
2.  **Provide Comprehensive Examples:** Include diverse examples showcasing both good and bad practices.
3.  **Use Validation Gates:**  PRPs include test commands for consistent results.
4.  **Leverage Documentation:** Include API docs and relevant resources.
5.  **Customize `CLAUDE.md`:** Tailor project-specific rules and coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)