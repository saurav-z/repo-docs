# Context Engineering: Build AI-Powered Applications with Precision

**Context Engineering elevates AI development by providing comprehensive context, leading to more reliable and sophisticated applications.**  ([View the original repo](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **Comprehensive Context:**  Go beyond basic prompts with documentation, examples, and rules.
*   **Reduced AI Failures:**  Address the root cause of agent failures by providing all necessary information.
*   **Consistent Output:**  Ensure your AI follows your project's patterns and conventions.
*   **Complex Feature Support:**  Enable multi-step implementations with proper context and guidance.
*   **Self-Correcting Capabilities:**  Validation loops allow the AI to identify and fix errors automatically.

## Getting Started

This template provides a streamlined workflow for implementing features using Context Engineering.

```bash
# 1. Clone the repository
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Customize Project Rules (optional)
# Edit CLAUDE.md to establish your project-specific guidelines.

# 3. Add Code Examples (essential!)
# Place relevant code patterns and examples within the examples/ folder.

# 4. Craft Your Feature Request
# Edit INITIAL.md to specify your desired feature requirements.

# 5. Generate a Product Requirements Prompt (PRP)
# In Claude Code, use the following command:
/generate-prp INITIAL.md

# 6. Execute the PRP to Implement the Feature
# Within Claude Code, use:
/execute-prp PRPs/your-feature-name.md
```

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective `INITIAL.md` Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering is a superior approach to prompt engineering, offering a complete system for guiding AI assistants.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**
*   Focuses on clever wording and specific phrasing.
*   Limited by the way a task is phrased.

**Context Engineering:**
*   Provides comprehensive context, including documentation, examples, rules, and validation.
*   Like writing a full screenplay with all the details.

### The Benefits of Context Engineering:

1.  **Reduced AI Failures:** Address the root causes of agent failures by supplying comprehensive context.
2.  **Ensures Consistency:**  Ensure the AI follows your project patterns and conventions.
3.  **Enables Complex Features:**  Facilitate multi-step implementations with proper context.
4.  **Self-Correcting:**  Validation loops enable the AI to identify and correct its own mistakes.

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

1.  **Set Up Global Rules (CLAUDE.md)**:

    Customize `CLAUDE.md` with your project's rules.  This file governs the AI assistant's behavior and includes:

    *   Project awareness (planning docs, tasks)
    *   Code structure (file limits, module organization)
    *   Testing requirements (unit test patterns, coverage)
    *   Style conventions (language, formatting)
    *   Documentation standards (docstrings, comments)

2.  **Create Your Initial Feature Request**:

    Edit `INITIAL.md` to outline what you want to build. Be specific!

    ```markdown
    ## FEATURE:
    [Describe feature - functionality, requirements]

    ## EXAMPLES:
    [Reference relevant examples from the examples/ folder, and their purpose.]

    ## DOCUMENTATION:
    [Links to relevant documentation, APIs, or other resources]

    ## OTHER CONSIDERATIONS:
    [Mention any important details like authentication, limits, or common pitfalls.]
    ```

    See `INITIAL_EXAMPLE.md` for a comprehensive example.

3.  **Generate the PRP**:

    PRPs (Product Requirements Prompts) are detailed blueprints for AI implementation.  Use the `/generate-prp` command in Claude Code:

    ```bash
    /generate-prp INITIAL.md
    ```

    This command:

    1.  Reads your feature request.
    2.  Analyzes your codebase.
    3.  Searches for relevant documentation.
    4.  Creates a comprehensive PRP in `PRPs/your-feature-name.md`.

4.  **Execute the PRP**:

    Run the `/execute-prp` command to implement your feature:

    ```bash
    /execute-prp PRPs/your-feature-name.md
    ```

    The AI assistant will:

    1.  Read all context from the PRP.
    2.  Create a detailed implementation plan.
    3.  Execute each step with validation.
    4.  Run tests and resolve any issues.
    5.  Confirm all success criteria are met.

## Writing Effective `INITIAL.md` Files

### Key Sections Explained

*   **FEATURE**: Be precise and comprehensive, defining functionality and requirements.
*   **EXAMPLES**: Leverage the `examples/` folder to show code patterns and patterns that should be followed.
*   **DOCUMENTATION**: Include links to all relevant API documentation, guides, and server resources.
*   **OTHER CONSIDERATIONS**: Capture important details like authentication, rate limits, and any requirements or potential problems.

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase**: Analyzes codebase, identifies patterns and conventions.
2.  **Documentation Gathering**: Fetches relevant API docs and library information.
3.  **Blueprint Creation**:  Generates a step-by-step implementation plan with validation and test requirements.
4.  **Quality Check**:  Assesses the confidence level and context completeness.

### How `/execute-prp` Works

1.  **Load Context**: Reads the entire PRP.
2.  **Plan**: Creates detailed task list.
3.  **Execute**: Implements each component.
4.  **Validate**: Runs tests and checks for issues.
5.  **Iterate**:  Fixes issues until resolved.
6.  **Complete**: Ensures all requirements are met.

## Using Examples Effectively

The `examples/` folder is **essential** for success!  Examples guide the AI in pattern recognition.

### What to Include in Examples

1.  **Code Structure Patterns**: Module organization, import conventions, class/function patterns.
2.  **Testing Patterns**: Test file structure, mocking, and assertion styles.
3.  **Integration Patterns**: API client implementations, database connections, and authentication flows.
4.  **CLI Patterns**: Argument parsing, output formatting, and error handling.

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

### 1.  Be Explicit in `INITIAL.md`

*   Clearly state requirements and constraints.
*   Reference examples frequently.

### 2. Provide Comprehensive Examples

*   More examples generally lead to better implementations.
*   Include error handling patterns.

### 3. Use Validation Gates

*   PRPs include test commands that must pass.
*   The AI iterates until all validations are successful.

### 4. Leverage Documentation

*   Include official API docs.
*   Reference documentation sections.

### 5. Customize `CLAUDE.md`

*   Add project-specific conventions and standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)