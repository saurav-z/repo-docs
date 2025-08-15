# Context Engineering: The Future of AI Coding 🚀

**Tired of basic prompt engineering? This template provides a cutting-edge approach to Context Engineering, empowering AI assistants to build complex features with unparalleled accuracy and efficiency.**  [Check out the original repo](https://github.com/coleam00/context-engineering-intro) for a deep dive.

## Key Features

*   **Comprehensive Context Provision:** Move beyond simple prompts with a complete system that provides AI assistants with documentation, examples, rules, patterns, and validation mechanisms.
*   **Reduced AI Failures:** Minimize agent errors by providing the necessary context for successful feature implementation.
*   **Ensured Consistency:** Define and enforce project patterns and conventions, leading to more maintainable code.
*   **Facilitated Complex Features:** Enable AI to handle multi-step implementations through thorough context and guided execution.
*   **Self-Correcting Implementations:** Leverage validation loops that allow AI to identify and fix its own errors, ensuring quality code.

## Quick Start

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set Project Rules (optional, use CLAUDE.md)
# Edit CLAUDE.md to add project-specific guidelines

# 3. Add Code Examples (highly recommended)
# Place relevant code examples in the examples/ folder

# 4. Create Feature Request (INITIAL.md)
# Edit INITIAL.md with your feature requirements

# 5. Generate PRP (Product Requirements Prompt)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute PRP to Implement Feature
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

## What is Context Engineering?

Context Engineering is a paradigm shift, moving beyond prompt engineering to a more structured approach:

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Relies on clever wording and phrasing.
*   Limited by how you phrase a task.

**Context Engineering:**

*   Provides comprehensive context for AI assistants.
*   Includes documentation, examples, rules, patterns, and validation.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses context-related errors, the primary cause of AI agent failures.
2.  **Ensures Consistency:** Ensures code adheres to your project's established patterns and conventions.
3.  **Enables Complex Features:** Facilitates multi-step implementations with appropriate context.
4.  **Self-Correcting:** Validation loops allow AI to identify and fix errors automatically.

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

Customize `CLAUDE.md` with project-wide rules for your AI assistant:

*   Project awareness
*   Code structure
*   Testing requirements
*   Style conventions
*   Documentation standards

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to detail what you want built:

```markdown
## FEATURE:
[Specific description of the feature's functionality and requirements]

## EXAMPLES:
[List and explain how to use example files in the examples/ folder]

## DOCUMENTATION:
[Links to relevant documentation, APIs, or server resources]

## OTHER CONSIDERATIONS:
[Gotchas, specific requirements, or common AI assistant errors]
```

See `INITIAL_EXAMPLE.md` for an example.

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are implementation blueprints:

*   Complete context and documentation
*   Implementation steps with validation
*   Error handling patterns
*   Test requirements

Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

This command:

1.  Reads your feature request.
2.  Researches the codebase for patterns.
3.  Searches for relevant documentation.
4.  Creates a PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Once generated, execute the PRP:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read all context.
2.  Create an implementation plan.
3.  Execute steps with validation.
4.  Run tests and fix issues.
5.  Ensure requirements are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE:** Be specific and comprehensive. Provide a clear and detailed description of the feature, including desired functionality and any necessary constraints.

**EXAMPLES:** Leverage the `examples/` folder. Provide code examples and explain how they are to be used. Specify patterns to be followed.

**DOCUMENTATION:** Include all relevant resources: API documentation URLs, library guides, and server documentation.

**OTHER CONSIDERATIONS:** Capture important details, such as authentication, rate limits, and performance requirements.

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase:** Analyzes your codebase for patterns and searches for similar implementations.
2.  **Documentation Gathering:** Fetches relevant API documentation and includes gotchas and quirks.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan with validation gates and test requirements.
4.  **Quality Check:** Scores confidence and ensures all context is included.

### How `/execute-prp` Works

1.  **Load Context**: Reads the entire PRP.
2.  **Plan**: Creates a detailed task list.
3.  **Execute**: Implements each component.
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Fixes any issues found.
6.  **Complete**: Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a comprehensive example.

## Using Examples Effectively

The `examples/` folder is **critical** for success, as it allows AI assistants to learn and replicate patterns.

### What to Include

1.  **Code Structure Patterns:** How modules are organized, import conventions, and class/function patterns.
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

### 1. Be Explicit in INITIAL.md

*   Include specific requirements and constraints.
*   Reference examples liberally.

### 2. Provide Comprehensive Examples

*   More examples result in better implementations.
*   Show both what to do and what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates

*   PRPs contain test commands.
*   AI will iterate until all validations succeed.

### 4. Leverage Documentation

*   Include official API docs.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md

*   Add your conventions.
*   Include project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)