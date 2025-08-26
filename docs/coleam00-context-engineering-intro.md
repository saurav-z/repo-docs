# Context Engineering: Revolutionizing AI Coding with Comprehensive Context

**Tired of frustrating AI coding failures? Context Engineering provides a superior approach to prompt engineering, enabling AI assistants to build complex features reliably by providing rich context.** ([View the original repository](https://github.com/coleam00/context-engineering-intro))

## Key Features:

*   **Context-Driven AI:** Provide comprehensive context, including documentation, examples, and rules, for robust and reliable AI-assisted coding.
*   **Comprehensive Product Requirements Prompts (PRPs):** Automatically generate detailed implementation blueprints for your AI assistant, including implementation steps with validation and test requirements.
*   **Reduced AI Failures:** Address the root cause of most agent failures by providing the necessary context and ensuring consistent behavior.
*   **Consistent Code Quality:** Ensure your AI follows your project's patterns, conventions, and coding standards.
*   **Simplified Complex Implementations:** Enable AI to tackle multi-step projects with proper context, validation, and error handling.
*   **Self-Correcting Implementation:** Leverage built-in validation loops and AI-powered iteration to refine and ensure your code is production-ready.

## Quick Start

Get started with Context Engineering in just a few steps:

```bash
# 1. Clone the Template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Define Project Rules (Optional - Template Provided)
# Edit CLAUDE.md to establish project-specific guidelines.

# 3. Add Relevant Code Examples (Highly Recommended)
# Place code examples in the 'examples/' folder.

# 4. Create Your Feature Request
# Edit INITIAL.md to describe the feature you want to build.

# 5. Generate a Product Requirements Prompt (PRP)
# Run in Claude Code:
/generate-prp INITIAL.md

# 6. Execute the PRP to Implement Your Feature
# Run in Claude Code:
/execute-prp PRPs/your-feature-name.md
```

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
    *   [Prompt Engineering vs. Context Engineering](#prompt-engineering-vs-context-engineering)
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

Context Engineering moves beyond simple prompt engineering, offering a comprehensive system to guide AI coding assistants with the information they need to complete tasks.

### Prompt Engineering vs Context Engineering

**Prompt Engineering:**

*   Focuses on crafting precise prompts and nuanced phrasing.
*   Limited to the wording of a single task.
*   Like giving someone a sticky note.

**Context Engineering:**

*   Provides a complete and comprehensive context for AI understanding.
*   Includes documentation, examples, rules, patterns, and validation.
*   Like providing a detailed screenplay with all the supporting information.

### Why Context Engineering Matters

1.  **Reduces AI Failures**: Most agent failures result from a lack of context.
2.  **Ensures Consistency**: The AI adheres to your project's established patterns and conventions.
3.  **Enables Complex Features**: The AI can handle complex, multi-step implementations with the right context.
4.  **Self-Correcting**: Validation loops allow the AI to iterate and fix its own mistakes.

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

This template focuses on the fundamentals of context engineering and does not integrate RAG or tools. Future iterations will expand on these areas.

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file sets project-wide rules for the AI assistant. The provided template includes:

*   **Project Awareness:** Instructions to read planning documents and review tasks.
*   **Code Structure:** Guidelines on file size limits and module organization.
*   **Testing Requirements:** Unit test patterns and coverage expectations.
*   **Style Conventions:** Preferences for language and formatting.
*   **Documentation Standards:** Docstring formats and commenting best practices.

**Customize `CLAUDE.md` to fit your project's specific needs.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to detail your desired functionality:

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

**Review `INITIAL_EXAMPLE.md` for a comprehensive example of a feature request.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are detailed implementation blueprints, encompassing:

*   Comprehensive context and documentation.
*   Step-by-step implementation plans with validation.
*   Error-handling strategies.
*   Test requirements.

These are similar to PRDs (Product Requirements Documents) but are tailored for instructing an AI coding assistant.

Run the following command within Claude Code:

```bash
/generate-prp INITIAL.md
```

**Note:** The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:
- `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
- `.claude/commands/execute-prp.md` - See how it implements features from PRPs

The `$ARGUMENTS` variable in these commands receives whatever you pass after the command name (e.g., `INITIAL.md` or `PRPs/your-feature.md`).

This command will:

1.  Read your feature request.
2.  Analyze the codebase for patterns.
3.  Locate relevant documentation.
4.  Create a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Once generated, run this command to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI coding assistant will:

1.  Read all context from the PRP.
2.  Generate a detailed implementation plan.
3.  Execute each step, with validation at each stage.
4.  Run tests and resolve any identified issues.
5.  Verify all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific and comprehensive in describing your desired functionality.

*   **Incorrect:** "Build a web scraper."
*   **Correct:** "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL."

**EXAMPLES**: Leverage the `examples/` folder to guide the AI assistant.

*   Place relevant code patterns in `examples/`.
*   Refer to specific files and patterns the AI should follow.
*   Explain the aspects to be mimicked.

**DOCUMENTATION**: Provide all relevant resources to the AI.

*   Include API documentation URLs.
*   Link to library guides.
*   Provide MCP server documentation if applicable.
*   Include relevant database schemas.

**OTHER CONSIDERATIONS**: Capture important details and edge cases.

*   Mention authentication requirements.
*   Detail rate limits or quotas.
*   Note common pitfalls or issues.
*   Specify performance requirements.

## The PRP Workflow

### How /generate-prp Works

The `/generate-prp` command follows this process:

1.  **Research Phase**
    *   Analyzes your codebase for patterns.
    *   Searches for similar implementations.
    *   Identifies the coding conventions to follow.

2.  **Documentation Gathering**
    *   Fetches relevant API documentation.
    *   Includes relevant library documentation.
    *   Adds any "gotchas" and common quirks.

3.  **Blueprint Creation**
    *   Creates a step-by-step implementation plan.
    *   Includes validation gates at each step.
    *   Adds test requirements.

4.  **Quality Check**
    *   Assigns a confidence level score (1-10).
    *   Ensures that all relevant context has been included.

### How /execute-prp Works

1.  **Load Context**: Reads the entire PRP, including the feature requirements and supporting information.
2.  **Plan**: Creates a detailed task list using TodoWrite.
3.  **Execute**: Implements each component as specified.
4.  **Validate**: Runs tests and linters to ensure quality and functionality.
5.  **Iterate**: If there are any issues, it fixes them by iterating.
6.  **Complete**: Makes sure that all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for an example of a fully generated PRP.

## Using Examples Effectively

The `examples/` folder is **crucial** for maximizing the effectiveness of your AI coding assistant. It shows the AI what to do and what *not* to do.

### What to Include in Examples

1.  **Code Structure Patterns**
    *   How to organize your modules.
    *   How to handle import conventions.
    *   How to handle class and function patterns.

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

*   Don't assume the AI knows your preferences or any other context.
*   Include specific requirements and constraints.
*   Reference examples liberally.

### 2. Provide Comprehensive Examples

*   The more examples you provide, the better the implementations will be.
*   Show both what to do and what *not* to do.
*   Include error handling patterns.

### 3. Use Validation Gates

*   PRPs include test commands that must pass for validation.
*   The AI will iterate and correct the code until all validations pass.
*   This approach ensures working code on the first try.

### 4. Leverage Documentation

*   Include links to official API documentation.
*   Reference any relevant MCP server resources.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md

*   Add your own coding conventions.
*   Incorporate project-specific rules.
*   Define your coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)