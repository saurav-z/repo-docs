# Context Engineering: The Future of AI Coding

**Stop wrestling with prompts and start engineering the *context* your AI needs to succeed!** This template provides a comprehensive framework for Context Engineering, a revolutionary approach to building AI-powered applications.  Learn how to give your AI assistants the complete picture with documentation, examples, and rules to achieve 10x better results than prompt engineering.  For more information, check out the original repo: [Context Engineering Intro](https://github.com/coleam00/context-engineering-intro).

## Key Features

*   **Context-Driven Development:** Move beyond prompts with a complete system for providing context including documentation, examples, rules, patterns, and validation.
*   **Comprehensive Context:** Reduce AI failures and ensure consistent, complex feature implementation with self-correcting capabilities.
*   **Streamlined Workflow:**  Automate the development process from feature requests to implementation using Product Requirement Prompts (PRPs) and execution tools.
*   **Example-Driven Learning:** Leverage code examples to guide AI assistant behavior, including code structure, testing, and integration patterns.
*   **Customizable and Flexible:** Tailor the system to your project's needs with project-specific rules and coding standards.

## Quick Start

```bash
# 1. Clone this template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up your project rules (optional - template provided)
# Edit CLAUDE.md to add your project-specific guidelines

# 3. Add examples (highly recommended)
# Place relevant code examples in the examples/ folder

# 4. Create your initial feature request
# Edit INITIAL.md with your feature requirements

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

Context Engineering is a paradigm shift from prompt engineering, focusing on providing AI with all necessary information to solve a problem.

### Prompt Engineering vs Context Engineering

**Prompt Engineering:**

*   Focuses on clever wording and specific phrasing.
*   Limited to how you phrase a task.

**Context Engineering:**

*   A complete system for providing comprehensive context.
*   Includes documentation, examples, rules, patterns, and validation.

### Why Context Engineering Matters

1.  **Reduces AI Failures**: Most agent failures are due to lack of context.
2.  **Ensures Consistency**: Your project's patterns and conventions are followed.
3.  **Enables Complex Features**: AI can handle multi-step implementations with the right context.
4.  **Self-Correcting**: Validation loops allow AI to fix its own mistakes.

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

Follow these steps to harness the power of Context Engineering:

### 1. Set Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file defines project-wide rules that the AI assistant adheres to in every conversation. It includes settings for:

*   **Project Awareness**: Understanding project documentation and tasks.
*   **Code Structure**: Enforcing file size limits, and module organization.
*   **Testing Requirements**: Specifying unit test patterns and coverage expectations.
*   **Style Conventions**: Setting language preferences and formatting rules.
*   **Documentation Standards**: Defining docstring formats and commenting practices.

**Customize the provided template to fit your project.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to describe the feature you want to build:

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

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints:

*   Complete context and documentation.
*   Implementation steps with validation.
*   Error handling patterns.
*   Test requirements.

Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

**Note:** The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:

*   `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
*   `.claude/commands/execute-prp.md` - See how it implements features from PRPs

The `$ARGUMENTS` variable in these commands receives whatever you pass after the command name (e.g., `INITIAL.md` or `PRPs/your-feature.md`).

This command will:

1.  Read your feature request
2.  Research the codebase for patterns
3.  Search for relevant documentation
4.  Create a comprehensive PRP in `PRPs/your-feature-name.md`

### 4. Execute the PRP

Once generated, execute the PRP to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI coding assistant will:

1.  Read all context from the PRP
2.  Create a detailed implementation plan
3.  Execute each step with validation
4.  Run tests and fix any issues
5.  Ensure all success criteria are met

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific and comprehensive
    *   ❌ "Build a web scraper"
    *   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Leverage the `examples/` folder
    *   Place relevant code patterns in `examples/`
    *   Reference specific files and patterns to follow
    *   Explain what aspects should be mimicked

**DOCUMENTATION**: Include all relevant resources
    *   API documentation URLs
    *   Library guides
    *   MCP server documentation
    *   Database schemas

**OTHER CONSIDERATIONS**: Capture important details
    *   Authentication requirements
    *   Rate limits or quotas
    *   Common pitfalls
    *   Performance requirements

## The PRP Workflow

### How /generate-prp Works

The command follows this process:

1.  **Research Phase**
    *   Analyzes your codebase for patterns
    *   Searches for similar implementations
    *   Identifies conventions to follow

2.  **Documentation Gathering**
    *   Fetches relevant API docs
    *   Includes library documentation
    *   Adds gotchas and quirks

3.  **Blueprint Creation**
    *   Creates step-by-step implementation plan
    *   Includes validation gates
    *   Adds test requirements

4.  **Quality Check**
    *   Scores confidence level (1-10)
    *   Ensures all context is included

### How /execute-prp Works

1.  **Load Context**: Reads the entire PRP
2.  **Plan**: Creates detailed task list using TodoWrite
3.  **Execute**: Implements each component
4.  **Validate**: Runs tests and linting
5.  **Iterate**: Fixes any issues found
6.  **Complete**: Ensures all requirements met

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example of what gets generated.

## Using Examples Effectively

The `examples/` folder is **critical** for success. AI coding assistants perform much better when they can see patterns to follow.

### What to Include in Examples

1.  **Code Structure Patterns**
    *   How you organize modules
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
    *   Don't assume the AI knows your preferences
    *   Include specific requirements and constraints
    *   Reference examples liberally

### 2. Provide Comprehensive Examples
    *   More examples = better implementations
    *   Show both what to do AND what not to do
    *   Include error handling patterns

### 3. Use Validation Gates
    *   PRPs include test commands that must pass
    *   AI will iterate until all validations succeed
    *   This ensures working code on first try

### 4. Leverage Documentation
    *   Include official API docs
    *   Add MCP server resources
    *   Reference specific documentation sections

### 5. Customize CLAUDE.md
    *   Add your conventions
    *   Include project-specific rules
    *   Define coding standards

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)