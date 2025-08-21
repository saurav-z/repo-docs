# Context Engineering: Revolutionizing AI Coding Assistants (10x Better than Prompt Engineering!)

**Tired of generic AI code? Context Engineering empowers AI assistants to deliver comprehensive, consistent, and complex code, all by providing them with the context they need to succeed.** Dive into the future of AI-driven development with this comprehensive template. ([See the original repo](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **Comprehensive Context:** Provide AI with all the information needed for end-to-end code generation.
*   **Consistent Results:** Ensure AI follows project patterns, conventions, and coding standards.
*   **Complex Feature Implementation:** Enable AI to handle multi-step implementations with validation.
*   **Self-Correcting:** Utilize validation loops for the AI to identify and resolve its own errors.
*   **10x Improvement:** Achieve superior results compared to traditional prompt engineering.

## Quick Start

Follow these steps to get started with Context Engineering:

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Customize Project Rules (optional - template provided)
# Edit CLAUDE.md to tailor project-specific guidelines.

# 3. Add Code Examples (highly recommended)
# Place relevant code patterns within the `examples/` directory.

# 4. Create a Feature Request
# Edit `INITIAL.md` to define feature requirements.

# 5. Generate a Product Requirements Prompt (PRP)
# Within Claude Code, execute:
/generate-prp INITIAL.md

# 6. Execute the PRP to implement the feature
# Within Claude Code, execute:
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

Context Engineering represents a paradigm shift from prompt engineering by providing a complete system for guiding AI code generation.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on clever wording and specific phrasing.
*   Limited to how you phrase a task.

**Context Engineering:**

*   A holistic system offering comprehensive context.
*   Includes documentation, code examples, rules, patterns, and validation.

### Why Context Engineering Matters

1.  **Reduced AI Failures:** Addresses context failures, the most common reason for AI agent failures.
2.  **Consistent Code:** Ensures AI adheres to your project's patterns and conventions.
3.  **Enhanced Complexity:** Enables AI to manage multi-step implementations successfully.
4.  **Self-Correcting Capabilities:** Leverages validation loops to empower AI to correct its own mistakes.

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

`CLAUDE.md` sets project-wide rules that the AI assistant uses in every conversation.

*   **Project Awareness**: Read planning docs, check tasks
*   **Code Structure**: File size limits, module organization
*   **Testing Requirements**: Unit test patterns, coverage expectations
*   **Style Conventions**: Language preferences, formatting rules
*   **Documentation Standards**: Docstring formats, commenting practices

**Use the template as-is or customize it for your project.**

### 2. Create Your Feature Request

Edit `INITIAL.md` to describe what you want to build:

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

**See `INITIAL_EXAMPLE.md` for a complete example.**

### 3. Generate the Product Requirements Prompt (PRP)

PRPs are comprehensive implementation blueprints for AI coding assistants.

Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

This command will:

1.  Read your feature request.
2.  Analyze your codebase for patterns.
3.  Search for relevant documentation.
4.  Create a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Once generated, execute the PRP to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read the entire PRP context.
2.  Create a detailed implementation plan.
3.  Execute each step with validation.
4.  Run tests and address any issues.
5.  Confirm all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific about the desired functionality and requirements.
*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**:  Leverage the `examples/` folder.
*   Include relevant code patterns in `examples/`.
*   Reference specific files and patterns.
*   Explain desired mimicking.

**DOCUMENTATION**: Provide all relevant resources.
*   Include API documentation URLs.
*   Provide links to library guides.
*   Include MCP server documentation.
*   Include database schemas.

**OTHER CONSIDERATIONS**: Capture important details.
*   Authentication requirements.
*   Rate limits or quotas.
*   Common pitfalls.
*   Performance requirements.

## The PRP Workflow

### How `/generate-prp` Works

The command follows this process:

1.  **Research Phase**:
    *   Analyzes your codebase.
    *   Searches for similar implementations.
    *   Identifies conventions.

2.  **Documentation Gathering**:
    *   Fetches relevant API documentation.
    *   Includes library documentation.
    *   Adds any gotchas.

3.  **Blueprint Creation**:
    *   Creates a step-by-step implementation plan.
    *   Includes validation gates.
    *   Adds test requirements.

4.  **Quality Check**:
    *   Scores confidence level.
    *   Ensures all context is included.

### How `/execute-prp` Works

1.  **Load Context**: Reads the entire PRP.
2.  **Plan**: Creates detailed task list.
3.  **Execute**: Implements each component.
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Fixes any issues found.
6.  **Complete**: Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example.

## Using Examples Effectively

The `examples/` folder is **crucial** for success.

### What to Include in Examples

1.  **Code Structure Patterns**:
    *   Module organization.
    *   Import conventions.
    *   Class/function patterns.

2.  **Testing Patterns**:
    *   Test file structure.
    *   Mocking approaches.
    *   Assertion styles.

3.  **Integration Patterns**:
    *   API client implementations.
    *   Database connections.
    *   Authentication flows.

4.  **CLI Patterns**:
    *   Argument parsing.
    *   Output formatting.
    *   Error handling.

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

*   Avoid assumptions; include specific requirements and constraints.
*   Reference examples generously.

### 2. Provide Comprehensive Examples

*   More examples lead to better implementations.
*   Show what to do *and* what not to do.
*   Include error handling patterns.

### 3. Utilize Validation Gates

*   PRPs incorporate test commands.
*   The AI iterates until all validations succeed.
*   Ensures working code on the first try.

### 4. Leverage Documentation

*   Include official API documentation.
*   Add MCP server resources.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md

*   Include your conventions and project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)