# Context Engineering Template: Revolutionizing AI Coding with Comprehensive Context

**Tired of frustrating AI coding failures? Context Engineering offers a superior approach to prompt engineering, providing AI assistants with the complete context they need to succeed.**

[Link to original repo: https://github.com/coleam00/context-engineering-intro](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Comprehensive Context:** Provides documentation, examples, rules, and validation for AI assistants.
*   **Reduced AI Failures:** Addresses the root cause of AI agent failures – lack of context.
*   **Enhanced Consistency:** Ensures AI follows your project patterns and coding conventions.
*   **Enables Complex Features:** Empowers AI to handle multi-step implementations with ease.
*   **Self-Correcting:** Leverages validation loops, allowing AI to fix its own errors.

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
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering goes beyond prompt engineering, providing a holistic approach to AI coding.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**
*   Focuses on clever wording and specific phrasing.
*   Limited to how you phrase a task.
*   Analogous to a sticky note.

**Context Engineering:**
*   A complete system for providing comprehensive context.
*   Includes documentation, examples, rules, patterns, and validation.
*   Like writing a full screenplay with all the details.

### Why Context Engineering Matters

1.  **Minimizes AI Failures:** Most agent failures stem from a lack of sufficient context.
2.  **Ensures Consistency:** Enforces adherence to project patterns and conventions.
3.  **Enables Complex Features:** Facilitates AI's ability to handle multi-step tasks effectively.
4.  **Self-Correcting Capabilities:** Allows AI to iterate and correct its own mistakes through validation.

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

*   The `CLAUDE.md` file defines project-wide rules that the AI assistant will follow in every interaction.
*   The template includes guidelines for:
    *   Project awareness.
    *   Code structure.
    *   Testing requirements.
    *   Style conventions.
    *   Documentation standards.
*   **Customize `CLAUDE.md` to suit your project's specific needs.**

### 2. Create Your Initial Feature Request

*   Edit `INITIAL.md` to outline the desired feature:

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
*   **Refer to `INITIAL_EXAMPLE.md` for a detailed example.**

### 3. Generate the PRP

*   PRPs (Product Requirements Prompts) are comprehensive implementation blueprints.
*   Run in Claude Code:

    ```bash
    /generate-prp INITIAL.md
    ```
*   This command will:
    1.  Analyze your feature request.
    2.  Research the codebase.
    3.  Search for relevant documentation.
    4.  Create a comprehensive PRP in `PRPs/your-feature-name.md`.
*   **Note:** The slash commands are custom commands defined in `.claude/commands/`.  You can view their implementation:
    *   `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
    *   `.claude/commands/execute-prp.md` - See how it implements features from PRPs

### 4. Execute the PRP

*   After generating the PRP, execute it to implement your feature:

    ```bash
    /execute-prp PRPs/your-feature-name.md
    ```

    The AI assistant will:
    1.  Read the entire PRP context.
    2.  Generate a detailed implementation plan.
    3.  Execute each step with validation.
    4.  Run tests and resolve any issues.
    5.  Ensure all success criteria are met.
*   See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example of what gets generated.

## Writing Effective INITIAL.md Files

### Key Sections Explained

*   **FEATURE**: Be specific and include all requirements
    *   ❌ "Build a web scraper"
    *   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"
*   **EXAMPLES**: Leverage the `examples/` folder
    *   Place relevant code patterns in `examples/`.
    *   Reference specific files to guide implementation.
    *   Explain what to mimic.
*   **DOCUMENTATION**: Include all relevant resources
    *   API documentation URLs.
    *   Library guides.
    *   Server documentation.
    *   Database schemas.
*   **OTHER CONSIDERATIONS**: Capture important details
    *   Authentication requirements.
    *   Rate limits or quotas.
    *   Common pitfalls.
    *   Performance requirements.

## The PRP Workflow

### How /generate-prp Works

1.  **Research Phase**
    *   Analyzes your codebase for patterns.
    *   Searches for similar implementations.
    *   Identifies conventions to follow.
2.  **Documentation Gathering**
    *   Fetches relevant API docs.
    *   Includes library documentation.
    *   Adds gotchas and quirks.
3.  **Blueprint Creation**
    *   Creates a step-by-step implementation plan.
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
6.  **Complete**: Ensures all requirements met.

## Using Examples Effectively

The `examples/` directory is **crucial** for success. AI coding assistants perform better with readily available patterns.

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
*   Include specific requirements and constraints.
*   Reference examples frequently.

### 2. Provide Comprehensive Examples
*   More examples generally lead to better implementations.
*   Show both what to do *and* what *not* to do.
*   Include error handling patterns.

### 3. Use Validation Gates
*   PRPs include test commands that must pass.
*   AI will iterate until all validations succeed.
*   This ensures functional code on the first attempt.

### 4. Leverage Documentation
*   Incorporate official API documentation.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md
*   Include project-specific rules and conventions.
*   Define clear coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)