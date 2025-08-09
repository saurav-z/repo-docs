# Context Engineering Template: Revolutionize AI Coding with Comprehensive Context

**Unlock 10x better AI coding results by embracing Context Engineering – a comprehensive system designed to provide AI assistants with all the necessary information for end-to-end task completion.** [View the original repository](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Context-Driven AI:** Provides AI with project rules, examples, and documentation for accurate and consistent code generation.
*   **Automated PRP Generation:** Automatically creates comprehensive Product Requirements Prompts (PRPs) based on your feature requests.
*   **Streamlined Implementation:** Executes PRPs to implement features with validation, error handling, and test integration.
*   **Code Consistency:** Ensures your AI assistant follows your project's patterns, conventions, and style guidelines.
*   **Reduced AI Failures:** Minimizes AI agent failures by addressing the root cause: lack of sufficient context.

## Quick Start

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up project rules (optional - template provided)
# Edit CLAUDE.md to customize your project guidelines.

# 3. Add examples (highly recommended)
# Place relevant code examples in the examples/ folder.

# 4. Create your initial feature request
# Edit INITIAL.md with your feature requirements.

# 5. Generate a comprehensive PRP (Product Requirements Prompt)
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
*   [The PRP Workflow](#the-prp-workflow)
    *   [How /generate-prp Works](#how-generate-prp-works)
    *   [How /execute-prp Works](#how-execute-prp-works)
*   [Using Examples Effectively](#using-examples-effectively)
    *   [What to Include in Examples](#what-to-include-in-examples)
    *   [Example Structure](#example-structure)
*   [Best Practices](#best-practices)
    *   [1. Be Explicit in INITIAL.md](#1-be-explicit-in-initialmd)
    *   [2. Provide Comprehensive Examples](#2-provide-comprehensive-examples)
    *   [3. Use Validation Gates](#3-use-validation-gates)
    *   [4. Leverage Documentation](#4-leverage-documentation)
    *   [5. Customize CLAUDE.md](#5-customize-claudemd)
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering represents a paradigm shift from prompt engineering, providing a system for comprehensive context to AI assistants.

### Prompt Engineering vs Context Engineering

**Prompt Engineering:**
*   Focuses on clever wording and specific phrasing.
*   Limited to how you phrase a task.
*   Like giving someone a sticky note.

**Context Engineering:**
*   A complete system for providing comprehensive context.
*   Includes documentation, examples, rules, patterns, and validation.
*   Like writing a full screenplay with all the details.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Most agent failures are due to insufficient context, not model limitations.
2.  **Ensures Consistency:** AI follows your project patterns, conventions, and coding style.
3.  **Enables Complex Features:** AI can handle multi-step implementations with appropriate context.
4.  **Self-Correcting:** Validation loops allow AI to fix its own mistakes and produce reliable results.

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

This template prioritizes RAG and tools with context engineering, with more developments in store.

## Step-by-Step Guide

This template guides you through the process of using Context Engineering to enhance AI coding outcomes.

### 1. Set Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file defines project-wide rules for your AI assistant, including coding standards, testing, and documentation guidelines. The template provides initial rules that you can customize.

*   **Project awareness:** Reads planning docs, checks tasks.
*   **Code structure:** File size limits, module organization.
*   **Testing requirements:** Unit test patterns, coverage expectations.
*   **Style conventions:** Language preferences, formatting rules.
*   **Documentation standards:** Docstring formats, commenting practices.

**Customize this file to ensure AI alignment with your project's specific requirements.**

### 2. Create Your Initial Feature Request

Create or edit the `INITIAL.md` file to specify what you want to build. Provide a detailed description of the feature and its requirements:

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

**Refer to `INITIAL_EXAMPLE.md` for a complete example.**

### 3. Generate the PRP

Generate a comprehensive PRP (Product Requirements Prompt) using the command:

```bash
/generate-prp INITIAL.md
```

This command performs the following:
1.  **Reads Your Feature Request:** Understands what you want to build.
2.  **Researches the Codebase:** Identifies patterns and conventions.
3.  **Gathers Documentation:** Incorporates relevant API documentation and guides.
4.  **Creates a PRP:**  Generates a detailed, step-by-step implementation plan in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Once the PRP is generated, execute it to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  **Load Context:** Reads the entire PRP.
2.  **Plan:** Creates a detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes any issues found.
6.  **Complete:** Ensures all requirements are met.

## Writing Effective INITIAL.md Files

The `INITIAL.md` file is crucial for setting clear expectations for the AI assistant.

### Key Sections Explained

**FEATURE**: Provide a clear, specific, and comprehensive description of the feature.
*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Reference code examples in the `examples/` folder to illustrate patterns.
*   Specify files and patterns to follow.
*   Explain the aspects to be mimicked.

**DOCUMENTATION**: Include all relevant resources to guide the AI.
*   Provide API documentation URLs.
*   Include library guides and server documentation.
*   Specify database schemas.

**OTHER CONSIDERATIONS**: Capture important details and edge cases.
*   Outline authentication requirements.
*   Include information about rate limits.
*   Specify common pitfalls.
*   Detail performance requirements.

## The PRP Workflow

Understanding how the PRP generation and execution work will allow you to get the most from this template.

### How /generate-prp Works

1.  **Research Phase**: Analyzes your codebase for patterns, searches for similar implementations, and identifies existing conventions.
2.  **Documentation Gathering**: Fetches relevant API docs, includes library documentation, and adds gotchas and quirks.
3.  **Blueprint Creation**: Creates a step-by-step implementation plan, includes validation gates, and adds test requirements.
4.  **Quality Check**: Scores confidence level (1-10) and ensures all context is included.

### How /execute-prp Works

1.  **Load Context**: Reads the entire PRP.
2.  **Plan**: Creates detailed task list using TodoWrite.
3.  **Execute**: Implements each component.
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Fixes any issues found.
6.  **Complete**: Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example of what gets generated.

## Using Examples Effectively

The `examples/` folder is **critical** for success. AI coding assistants perform much better when they can see patterns to follow.

### What to Include in Examples

1.  **Code Structure Patterns:** Module organization, import conventions, and class/function patterns.
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

Implement these best practices for optimal results:

### 1. Be Explicit in INITIAL.md

Provide specific requirements, constraints, and preferences in `INITIAL.md`. Reference examples liberally.

### 2. Provide Comprehensive Examples

The more comprehensive the examples, the better the implementations. Demonstrate both what to do and what not to do, and include error handling patterns.

### 3. Use Validation Gates

PRPs include test commands that must pass. The AI assistant will iterate until all validations succeed, ensuring working code.

### 4. Leverage Documentation

Include official API docs, MCP server resources, and reference specific documentation sections.

### 5. Customize CLAUDE.md

Add your conventions, project-specific rules, and coding standards to the `CLAUDE.md` file.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)