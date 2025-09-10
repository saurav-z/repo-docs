# Context Engineering: Build AI that Codes with Confidence

**Tired of generic prompt engineering? Context Engineering empowers your AI coding assistants with comprehensive context, resulting in more reliable and complex feature implementations.** ([Original Repo](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **Comprehensive Context:** Provide documentation, examples, rules, and validation for your AI assistant.
*   **Reduced AI Failures:** Minimize errors by giving the AI the information it needs.
*   **Consistent Code Quality:** Ensure adherence to your project's patterns and conventions.
*   **Enable Complex Features:** Tackle multi-step implementations with ease.
*   **Self-Correcting Capabilities:** Built-in validation loops allow AI to fix its own mistakes.

## Quick Start

Follow these steps to begin using the Context Engineering template:

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Customize Project Rules (optional)
# Edit CLAUDE.md to establish project-specific guidelines (e.g., coding style, testing standards).

# 3. Add Code Examples (crucial)
# Place relevant code examples within the examples/ directory to demonstrate desired patterns.

# 4. Create Feature Requests
# Define your desired features in INITIAL.md, specifying requirements, examples, and documentation.

# 5. Generate Product Requirements Prompts (PRPs)
# Within Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP
# Within Claude Code, run:
/execute-prp PRPs/your-feature-name.md
```

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
    *   [Prompt Engineering vs. Context Engineering](#prompt-engineering-vs-context-engineering)
    *   [Why Context Engineering Matters](#why-context-engineering-matters)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering elevates AI coding by providing it with a complete understanding of your project.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**
*   Relies on clever phrasing and precise wording.
*   Limited by the creativity of your prompts.
*   Like a sticky note with instructions.

**Context Engineering:**
*   Provides comprehensive context and documentation.
*   Includes examples, rules, and validation.
*   Like writing a detailed screenplay.

### Why Context Engineering Matters

1.  **Reduces AI Failures**: Addresses context failures, the primary cause of agent errors.
2.  **Ensures Consistency**: Enforces project-specific patterns and conventions.
3.  **Enables Complex Features**: Empowers AI to handle multi-step implementations.
4.  **Self-Correcting**: Validation mechanisms enable the AI to resolve its mistakes.

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

This template focuses on the core principles of Context Engineering.

## Step-by-Step Guide

Follow these steps to leverage the Context Engineering template:

### 1. Set Up Global Rules (CLAUDE.md)

`CLAUDE.md` defines project-wide rules for your AI assistant, including:

*   **Project Awareness**:  Instructions for reading documentation and checking tasks.
*   **Code Structure**:  File size constraints, module organization preferences.
*   **Testing Requirements**: Unit test patterns and coverage expectations.
*   **Style Conventions**: Language preferences and formatting rules.
*   **Documentation Standards**: Docstring formats and commenting practices.

**Customize the template to match your project's specific requirements.**

### 2. Create Your Initial Feature Request

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

**Refer to `INITIAL_EXAMPLE.md` for a comprehensive example.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints, similar to PRDs, specifically tailored for AI coding assistants.

Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

The `/generate-prp` command:
1.  Analyzes your feature request.
2.  Examines the codebase to identify patterns.
3.  Searches for relevant documentation.
4.  Creates a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Implement your feature by executing the generated PRP:

```bash
/execute-prp PRPs/your-feature-name.md
```

The `/execute-prp` command will:
1.  Read all context from the PRP.
2.  Create a detailed implementation plan.
3.  Execute each step with validation.
4.  Run tests and resolve any issues.
5.  Ensure all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Provide a clear and detailed description.
*   ❌ "Build a web scraper"
*   ✅ "Develop an asynchronous web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL."

**EXAMPLES**: Leverage the `examples/` directory.
*   Include relevant code patterns.
*   Reference files and patterns to be followed.
*   Explain the desired aspects to be mimicked.

**DOCUMENTATION**: Include all relevant resources.
*   API documentation URLs
*   Library guides
*   Server documentation
*   Database schemas

**OTHER CONSIDERATIONS**: Specify any important details.
*   Authentication requirements
*   Rate limits or quotas
*   Potential pitfalls
*   Performance requirements

## The PRP Workflow

### How /generate-prp Works

The command performs these steps:

1.  **Research Phase**
    *   Analyzes your codebase for patterns.
    *   Searches for similar implementations.
    *   Identifies conventions to follow.

2.  **Documentation Gathering**
    *   Fetches relevant API docs.
    *   Includes library documentation.
    *   Adds gotchas and quirks.

3.  **Blueprint Creation**
    *   Creates step-by-step implementation plan.
    *   Includes validation gates.
    *   Adds test requirements.

4.  **Quality Check**
    *   Scores confidence level (1-10).
    *   Ensures all context is included.

### How /execute-prp Works

1.  **Load Context**: Reads the entire PRP.
2.  **Plan**: Creates detailed task list using TodoWrite.
3.  **Execute**: Implements each component.
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Fixes any issues found.
6.  **Complete**: Ensures all requirements met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for an example PRP.

## Using Examples Effectively

The `examples/` directory is crucial for AI success.

### What to Include in Examples

1.  **Code Structure Patterns**
    *   How modules are organized
    *   Import conventions
    *   Class/function patterns

2.  **Testing Patterns**
    *   Test file structure
    *   Mocking approaches
    *   Assertion styles

3.  **Integration Patterns**
    *   API client implementations
    *   Database connections
    *   Authentication flows

4.  **CLI Patterns**
    *   Argument parsing
    *   Output formatting
    *   Error handling

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
*   Provide specific requirements and constraints.
*   Reference examples liberally.

### 2. Provide Comprehensive Examples
*   Show both what to do and what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates
*   PRPs include test commands that must pass.
*   This ensures working code on the first try.

### 4. Leverage Documentation
*   Include official API docs.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md
*   Add project-specific rules and coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)