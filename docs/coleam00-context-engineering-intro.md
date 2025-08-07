# Context Engineering Template: Revolutionize AI Coding with Comprehensive Context

**Unlock the power of AI coding assistants by providing them with the complete context they need to build features end-to-end.**

Learn how to use context engineering to get 10x better results than prompt engineering and 100x better results than vibe coding.

[Visit the original repository for more details](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Comprehensive Context:** Provide AI with documentation, examples, rules, patterns, and validation for complete project awareness.
*   **Reduced AI Failures:** Minimize agent errors by addressing context deficiencies, the leading cause of AI coding failures.
*   **Consistent & Complex Features:** Ensure AI follows project conventions, enabling the implementation of multi-step features.
*   **Self-Correcting Mechanism:** Employ validation loops that empower AI to autonomously rectify its own mistakes.
*   **Automated Product Requirement Prompts (PRPs):** Leverage the `/generate-prp` command to automatically create PRPs from your feature requests.
*   **Simplified Feature Implementation:** Use the `/execute-prp` command to guide AI in implementing features from PRPs with validation.

## Quick Start

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up project rules (optional - template provided)
# Edit CLAUDE.md to customize your project-specific guidelines

# 3. Add code examples (highly recommended)
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
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering goes beyond traditional prompt engineering by equipping AI coding assistants with a holistic understanding of your project, resulting in superior performance and reliability.

### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering:**
    *   Relies on clever wording and precise phrasing.
    *   Limited by the ability to frame a specific task.
    *   Analogous to a sticky note.
*   **Context Engineering:**
    *   A complete system for providing comprehensive context.
    *   Includes documentation, examples, rules, patterns, and validation.
    *   Analogous to writing a complete screenplay with all the details.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Most agent failures are due to context failures, not model limitations.
2.  **Ensures Consistency:** AI consistently follows project patterns and conventions.
3.  **Enables Complex Features:** AI can handle multi-step implementations with proper context.
4.  **Self-Correcting:** Validation loops allow AI to fix its own mistakes.

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

This template provides the essential framework for getting started with context engineering.

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file establishes project-wide rules for the AI assistant. The provided template includes:

*   **Project Awareness:** Guidance on reading planning documents and checking tasks.
*   **Code Structure:** Guidelines for file size limits and module organization.
*   **Testing Requirements:** Unit test patterns and coverage expectations.
*   **Style Conventions:** Language preferences and formatting rules.
*   **Documentation Standards:** Docstring formats and commenting practices.

**Customize the provided template to align with your project's specific requirements.**

### 2. Create Your Initial Feature Request

Use `INITIAL.md` to describe the feature you want to build:

```markdown
## FEATURE:
[Clearly describe what you want to build, focusing on functionality and requirements]

## EXAMPLES:
[List and explain relevant example files in the examples/ folder and how they should be used]

## DOCUMENTATION:
[Include links to relevant documentation, APIs, or server resources]

## OTHER CONSIDERATIONS:
[Mention any constraints, requirements, or common AI assistant pitfalls]
```

**Refer to `INITIAL_EXAMPLE.md` for a detailed example.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints that guide AI coding assistants:

*   Provide complete context and documentation.
*   Outline implementation steps with validation checks.
*   Incorporate error handling patterns.
*   Specify test requirements.

Run this command within Claude Code:

```bash
/generate-prp INITIAL.md
```

The command automatically:

1.  Reads your feature request.
2.  Analyzes the codebase for patterns.
3.  Searches for relevant documentation.
4.  Creates a comprehensive PRP in `PRPs/your-feature-name.md`.

**Note:** The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:
- `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
- `.claude/commands/execute-prp.md` - See how it implements features from PRPs

The `$ARGUMENTS` variable in these commands receives whatever you pass after the command name (e.g., `INITIAL.md` or `PRPs/your-feature.md`).

### 4. Execute the PRP

Once generated, execute the PRP to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI coding assistant will:

1.  Read all context from the PRP.
2.  Create a detailed implementation plan.
3.  Execute each step with validation.
4.  Run tests and fix any issues.
5.  Ensure all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE:** Be specific and comprehensive.
*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES:** Leverage the `examples/` folder to provide relevant code.

*   Place code patterns in `examples/`.
*   Reference specific files and patterns to be followed.
*   Explain how examples should be used.

**DOCUMENTATION:** Include essential resources.

*   API documentation URLs.
*   Library guides.
*   Server documentation.
*   Database schemas.

**OTHER CONSIDERATIONS:** Capture important details.

*   Authentication requirements.
*   Rate limits or quotas.
*   Common pitfalls.
*   Performance requirements.

## The PRP Workflow

### How /generate-prp Works

The `/generate-prp` command follows this process:

1.  **Research Phase**
    *   Analyzes your codebase for patterns.
    *   Searches for similar implementations.
    *   Identifies conventions to follow.

2.  **Documentation Gathering**
    *   Fetches relevant API documentation.
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
2.  **Plan**: Creates detailed task list.
3.  **Execute**: Implements each component.
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Fixes any issues found.
6.  **Complete**: Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example.

## Using Examples Effectively

The `examples/` folder is **critical** for success. AI coding assistants perform much better when they can see patterns to follow.

### What to Include in Examples

1.  **Code Structure Patterns**
    *   Module organization.
    *   Import conventions.
    *   Class/function patterns.

2.  **Testing Patterns**
    *   Test file structure.
    *   Mocking approaches.
    *   Assertion styles.

3.  **Integration Patterns**
    *   API client implementations.
    *   Database connections.
    *   Authentication flows.

4.  **CLI Patterns**
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

*   Provide specific requirements and constraints.
*   Reference examples liberally.
*   Don't assume the AI knows your preferences.

### 2. Provide Comprehensive Examples

*   More examples lead to better implementations.
*   Include error handling patterns.
*   Show what to do and what not to do.

### 3. Use Validation Gates

*   PRPs include test commands that must pass.
*   The AI will iterate until all validations succeed.
*   This ensures working code on the first try.

### 4. Leverage Documentation

*   Include official API documentation.
*   Reference specific documentation sections.
*   Add server resources.

### 5. Customize CLAUDE.md

*   Include project-specific rules.
*   Define coding standards.
*   Add your coding conventions.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)