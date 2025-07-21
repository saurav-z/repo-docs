# Context Engineering Template: Build AI-powered features with precision.

**Context Engineering** empowers you to build AI-driven features with comprehensive context, dramatically improving results compared to traditional prompt engineering, and offering a superior experience compared to "vibe coding." This template provides a complete framework for orchestrating AI coding assistants to deliver complex features end-to-end. Learn more and get started at the original repository: [Context Engineering Intro](https://github.com/coleam00/context-engineering-intro).

## Key Features

*   **Comprehensive Context:** Provide AI with documentation, examples, and rules.
*   **Reduced AI Failures:** Address context failures, a major cause of agent issues.
*   **Consistency & Standardization:** Ensure code follows project patterns and conventions.
*   **Multi-Step Implementation:** Enables AI to handle complex features with validation.
*   **Self-Correcting:** Validation loops allow AI to fix its own mistakes.

## Quick Start

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up project rules (optional - template provided)
# Edit CLAUDE.md for project-specific guidelines

# 3. Add examples (highly recommended)
# Place code examples in the examples/ folder

# 4. Create your initial feature request
# Edit INITIAL.md with your feature requirements

# 5. Generate a Product Requirements Prompt (PRP)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP to implement your feature
# In Claude Code, run:
/execute-prp PRPs/your-feature-name.md
```

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
    *   [Prompt Engineering vs Context Engineering](#prompt-engineering-vs-context-engineering)
    *   [Why Context Engineering Matters](#why-context-engineering-matters)
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

Context Engineering offers a superior approach to traditional prompt engineering by providing a complete system for guiding AI assistants. It's like writing a full screenplay with all the details, rather than giving someone a sticky note.

### Prompt Engineering vs Context Engineering

**Prompt Engineering:** Focuses on clever wording and specific phrasing, limiting the AI's understanding.

**Context Engineering:** Provides comprehensive context with documentation, examples, rules, and validation.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses context failures, a primary source of agent problems.
2.  **Ensures Consistency:** Enforces project patterns and conventions.
3.  **Enables Complex Features:** Allows AI to handle multi-step implementations.
4.  **Self-Correcting:** Validation loops enable AI to fix its mistakes automatically.

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

Follow these steps to leverage the power of Context Engineering.

### 1. Set Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file sets project-wide rules for your AI assistant, including:

*   **Project Awareness:** Reading planning docs, checking tasks
*   **Code Structure:** File size limits, module organization
*   **Testing Requirements:** Unit test patterns, coverage expectations
*   **Style Conventions:** Language preferences, formatting rules
*   **Documentation Standards:** Docstring formats, commenting practices

Customize the provided template to meet your project's specific needs.

### 2. Create Your Initial Feature Request

Describe your desired feature within `INITIAL.md`. Include:

```markdown
## FEATURE:
[Specific feature description, including functionality and requirements]

## EXAMPLES:
[List examples and explain how they should be used]

## DOCUMENTATION:
[Links to relevant documentation, APIs, or server resources]

## OTHER CONSIDERATIONS:
[Gotchas, specific requirements, and common AI assistant misses]
```

Refer to `INITIAL_EXAMPLE.md` for a complete example.

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive blueprints for implementation, including:

*   Complete context and documentation
*   Implementation steps with validation
*   Error handling patterns
*   Test requirements

Run this in Claude Code:

```bash
/generate-prp INITIAL.md
```

This command will:

1.  Analyze your feature request.
2.  Research the codebase.
3.  Gather relevant documentation.
4.  Create a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

After generating the PRP, execute it to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read all context from the PRP.
2.  Create a detailed implementation plan.
3.  Execute each step with validation.
4.  Run tests and fix any issues.
5.  Ensure all success criteria are met.

## Writing Effective INITIAL.md Files

Crafting well-structured `INITIAL.md` files is crucial for success.

### Key Sections Explained

**FEATURE:** Be specific and comprehensive in your description.

*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES:** Utilize the `examples/` folder.

*   Place code patterns in `examples/`.
*   Reference and explain how specific files and patterns should be used.

**DOCUMENTATION:** Include all relevant resources.

*   API documentation URLs
*   Library guides
*   Server documentation
*   Database schemas

**OTHER CONSIDERATIONS:** Capture important details.

*   Authentication requirements
*   Rate limits or quotas
*   Common pitfalls
*   Performance requirements

## The PRP Workflow

Understand the inner workings of PRP generation and execution.

### How /generate-prp Works

The command follows this process:

1.  **Research Phase:** Analyzes your codebase and searches for similar implementations.
2.  **Documentation Gathering:** Fetches API and library documentation.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan with validation gates and test requirements.
4.  **Quality Check:** Scores confidence level and ensures all context is included.

### How /execute-prp Works

1.  **Load Context:** Reads the entire PRP.
2.  **Plan:** Creates a detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes any issues found.
6.  **Complete:** Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a comprehensive example.

## Using Examples Effectively

The `examples/` folder is essential for guiding your AI assistant.

### What to Include in Examples

1.  **Code Structure Patterns:** Module organization, import conventions, and function patterns.
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

*   **Be Explicit in INITIAL.md:** Provide specific requirements and constraints.
*   **Provide Comprehensive Examples:** Demonstrate both what to do and what not to do.
*   **Use Validation Gates:** Leverage test commands for immediate feedback and validation.
*   **Leverage Documentation:** Include official API documentation and server resources.
*   **Customize CLAUDE.md:** Define coding standards and project-specific rules.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)