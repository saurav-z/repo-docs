# Context Engineering Template: Build AI-powered applications with unparalleled precision.

[Get Started with Context Engineering](https://github.com/coleam00/context-engineering-intro) – Revolutionize your AI coding workflow by providing comprehensive context, leading to more reliable and efficient code generation.

**Key Features:**

*   **Context-Driven AI:** Move beyond simple prompts and provide your AI assistants with complete project context.
*   **Comprehensive PRPs:** Generate detailed Product Requirements Prompts (PRPs) that act as blueprints for your AI coding assistant.
*   **Example-Based Learning:** Leverage code examples to guide your AI and ensure consistent code style and patterns.
*   **Automated Validation:** Implement validation gates within your PRPs, ensuring the AI-generated code meets all requirements.
*   **Customizable Rules:**  Define global rules, coding standards, and project-specific guidelines for your AI assistant.

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Quick Start](#quick-start)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering is a revolutionary approach to AI-assisted coding, providing AI with the complete context needed for end-to-end implementation. This is a superior approach to traditional prompt engineering, which often falls short.

### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering:** Relies on crafting specific prompts and phrases, limited to how you phrase a task.
*   **Context Engineering:** Provides complete project context including documentation, examples, rules, patterns, and validation for complex, multi-step implementations.

### Benefits of Context Engineering:

1.  **Reduced AI Failures:** Addresses the primary cause of agent failures.
2.  **Consistent Results:**  Ensures AI adheres to your project patterns and conventions.
3.  **Enables Complex Features:**  Allows AI to tackle multi-step implementations with ease.
4.  **Self-Correcting:** Utilizes validation loops for automated error correction.

## Quick Start

Follow these steps to get started:

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Customize project rules (optional)
# Edit CLAUDE.md to specify project-specific guidelines

# 3. Add code examples (recommended)
# Place relevant code in the examples/ folder

# 4. Create an initial feature request
# Edit INITIAL.md with feature requirements

# 5. Generate a comprehensive PRP (Product Requirements Prompt)
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

This template focuses on providing a framework for Context Engineering and will soon include related tools.

## Step-by-Step Guide

1.  **Set Up Global Rules (CLAUDE.md):** Configure project-wide rules for your AI assistant within `CLAUDE.md`. This template includes an example, but customization is encouraged.

    *   Project awareness
    *   Code structure
    *   Testing requirements
    *   Style conventions
    *   Documentation standards

2.  **Create Your Initial Feature Request:** Describe your desired feature in `INITIAL.md`. The template includes sections for:

    *   **FEATURE:** Specific and comprehensive description.
    *   **EXAMPLES:** References to related code examples.
    *   **DOCUMENTATION:** Links to documentation and resources.
    *   **OTHER CONSIDERATIONS:**  Specific requirements and potential issues.  Refer to `INITIAL_EXAMPLE.md` for a sample.

3.  **Generate the PRP:**  Use the `/generate-prp INITIAL.md` command in Claude Code to generate a detailed Product Requirements Prompt.

    This command:

    *   Analyzes your feature request.
    *   Researches your codebase and relevant documentation.
    *   Generates a PRP in the `PRPs/` directory.
        *   The commands themselves can be viewed in `.claude/commands/`.

4.  **Execute the PRP:**  Implement your feature by running `/execute-prp PRPs/your-feature-name.md` in Claude Code.

    The AI will:

    *   Read the PRP and create an implementation plan.
    *   Execute each step and validate the results.
    *   Run tests and address any issues.
    *   Ensure all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections:

*   **FEATURE:** Be very specific about desired functionality and requirements.
*   **EXAMPLES:** Utilize code examples in the `examples/` folder.
*   **DOCUMENTATION:** Include relevant resources like API documentation.
*   **OTHER CONSIDERATIONS:** Note important details like authentication, rate limits, and performance goals.

## The PRP Workflow

### How /generate-prp Works:

1.  **Research Phase:** Codebase analysis and search for existing patterns.
2.  **Documentation Gathering:** Retrieval of relevant API documentation.
3.  **Blueprint Creation:** Development of a step-by-step plan, including validation and test requirements.
4.  **Quality Check:** Confidence scoring to ensure context inclusion.

### How /execute-prp Works:

1.  Load Context
2.  Plan
3.  Execute
4.  Validate
5.  Iterate (if needed)
6.  Complete

## Using Examples Effectively

The `examples/` directory is **crucial**. Good examples lead to good implementations.

### What to Include in Examples:

*   Code structure patterns
*   Testing patterns
*   Integration patterns
*   CLI patterns

### Example Structure:

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

1.  **Be Explicit in INITIAL.md:** Provide clear and comprehensive requirements and reference examples.
2.  **Provide Comprehensive Examples:** The more examples, the better. Include both what to do and what not to do.
3.  **Use Validation Gates:** Ensure working code through test commands within the PRP.
4.  **Leverage Documentation:** Integrate official API documentation and project resources.
5.  **Customize CLAUDE.md:** Adapt the global rules to your specific project needs.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)