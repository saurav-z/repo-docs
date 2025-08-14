# Context Engineering: Revolutionizing AI Coding with Comprehensive Context

**Tired of generic AI code? Context Engineering is the key to unlocking high-quality, consistent AI-generated code by providing AI assistants with the essential information they need.**  [Explore the original repository here](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Superior to Prompt Engineering:** Context Engineering surpasses prompt engineering by offering a complete system for guiding AI coding assistants.
*   **Reduced AI Failures:**  Minimizes agent failures by providing the necessary context.
*   **Consistent Results:**  Ensures AI follows project patterns and conventions.
*   **Complex Feature Enablement:**  Allows AI to handle multi-step implementations effectively.
*   **Self-Correcting Capabilities:** Leverages validation loops to enable AI to fix its own mistakes.

## Quick Start

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set project rules (optional, but recommended - edit CLAUDE.md)

# 3. Add code examples (essential - place in examples/)

# 4. Create your initial feature request (INITIAL.md)

# 5. Generate a PRP (Product Requirements Prompt)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP to implement your feature
# In Claude Code, run:
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

Context Engineering represents a paradigm shift from traditional prompt engineering, which focuses on wording and phrasing. Context Engineering offers a comprehensive system providing documentation, examples, rules, patterns, and validation.

### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering:** Uses clever wording and phrasing, limited to how a task is phrased, like giving someone a sticky note.
*   **Context Engineering:** Provides a complete system for comprehensive context, including documentation, examples, and rules.  Like writing a complete screenplay with all the details.

### Why Context Engineering Matters

1.  **Reduces AI Failures:**  Addresses context failures, which are a major cause of agent failures.
2.  **Ensures Consistency:**  Ensures AI follows project patterns and conventions.
3.  **Enables Complex Features:**  Allows AI to handle multi-step implementations effectively.
4.  **Self-Correcting:**  Utilizes validation loops, enabling AI to fix its mistakes.

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

1.  **Set Up Global Rules (CLAUDE.md):**  Define project-wide rules for the AI assistant.  Use the provided template and customize it for your project.  Includes project awareness, code structure, testing, style conventions, and documentation standards.

2.  **Create Your Initial Feature Request (INITIAL.md):**  Describe your feature, provide relevant examples, include links to documentation, and note any specific requirements.

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

3.  **Generate the PRP:**  Run the `/generate-prp INITIAL.md` command in Claude Code. This command analyzes your codebase, gathers documentation, and creates a comprehensive PRP in `PRPs/your-feature-name.md`.

4.  **Execute the PRP:**  Run the `/execute-prp PRPs/your-feature-name.md` command. The AI assistant will read the PRP, create an implementation plan, execute each step, run tests, and fix any issues until all requirements are met.

## Writing Effective INITIAL.md Files

**Key Sections Explained:**

*   **FEATURE:**  Be specific and comprehensive (e.g., "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL").
*   **EXAMPLES:**  Leverage the `examples/` folder, referencing specific files and patterns.
*   **DOCUMENTATION:** Include all relevant resources (API documentation, library guides, server documentation, database schemas).
*   **OTHER CONSIDERATIONS:** Capture important details (authentication, rate limits, common pitfalls, performance).

## The PRP Workflow

**How `/generate-prp` Works:**

1.  **Research Phase:** Analyzes your codebase and identifies conventions.
2.  **Documentation Gathering:** Fetches relevant documentation.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan with validation.
4.  **Quality Check:**  Scores confidence level and ensures all context is included.

**How `/execute-prp` Works:**

1.  Load Context
2.  Plan
3.  Execute
4.  Validate
5.  Iterate
6.  Complete

## Using Examples Effectively

The `examples/` folder is essential for success.  Provide patterns to guide the AI.

**What to Include in Examples:**

1.  Code Structure Patterns
2.  Testing Patterns
3.  Integration Patterns
4.  CLI Patterns

**Example Structure:**

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
2.  **Provide Comprehensive Examples:**  More examples lead to better implementations. Include error handling.
3.  **Use Validation Gates:**  PRPs include test commands to ensure working code.
4.  **Leverage Documentation:**  Include API docs and other resources.
5.  **Customize CLAUDE.md:**  Add your conventions and coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)