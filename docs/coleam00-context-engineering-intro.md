# Context Engineering: Build AI Assistants That Code End-to-End

**Stop wrestling with prompts and start building powerful AI assistants by mastering Context Engineering, a comprehensive framework for providing AI with the context it needs to succeed.** ([Original Repository](https://github.com/coleam00/context-engineering-intro))

## Key Features:

*   **Comprehensive Context:** Provide AI with the information it needs – documentation, examples, rules, and validation – for robust, reliable coding.
*   **Reduce AI Failures:** Address the root cause of AI failures, which are often context failures, by providing comprehensive context.
*   **Consistent Code:** Define your project's patterns and conventions for predictable and maintainable code.
*   **Enable Complex Features:** Empower AI to handle multi-step implementations and build advanced features with confidence.
*   **Self-Correcting:** Implement validation loops so AI can identify and fix errors, streamlining your development process.

## Quick Start Guide

1.  **Clone the Template:**
    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```
2.  **Set Up Project Rules (Optional):**
    *   Edit `CLAUDE.md` to define project-specific guidelines.
3.  **Add Code Examples (Highly Recommended):**
    *   Place relevant code examples in the `examples/` folder.
4.  **Create Your Initial Feature Request:**
    *   Edit `INITIAL.md` to describe your feature requirements.
5.  **Generate a Product Requirements Prompt (PRP):**
    *   Use the custom Claude Code command:
        ```bash
        /generate-prp INITIAL.md
        ```
6.  **Execute the PRP:**
    *   Use the custom Claude Code command:
        ```bash
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

## What is Context Engineering?

Context Engineering surpasses prompt engineering by providing a comprehensive system for guiding AI.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**
*   Relies on crafting clever prompts.
*   Limited by the wording of the task.
*   Like a sticky note.

**Context Engineering:**
*   Provides comprehensive context, including documentation, examples, rules, and validation.
*   Like a complete screenplay.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses context failures, not just model failures.
2.  **Ensures Consistency:** Enforces project patterns and conventions.
3.  **Enables Complex Features:** Allows AI to handle multi-step implementations.
4.  **Self-Correcting:** Utilizes validation loops for AI to fix its own errors.

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

*   `CLAUDE.md` defines project-wide rules. The template covers:
    *   Project awareness
    *   Code structure
    *   Testing requirements
    *   Style conventions
    *   Documentation standards
*   Customize `CLAUDE.md` for your project.

### 2. Create Your Initial Feature Request

*   Edit `INITIAL.md` to describe your feature.

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
*   See `INITIAL_EXAMPLE.md` for a complete example.

### 3. Generate the PRP

*   PRPs (Product Requirements Prompts) are detailed implementation blueprints.
*   They include context, steps, validation, and test requirements.
*   Run this command in Claude Code:
    ```bash
    /generate-prp INITIAL.md
    ```
*   This command:
    1.  Reads your feature request.
    2.  Researches the codebase.
    3.  Searches for documentation.
    4.  Creates a PRP in `PRPs/your-feature-name.md`.
*   View the command implementations:
    *   `.claude/commands/generate-prp.md`
    *   `.claude/commands/execute-prp.md`

### 4. Execute the PRP

*   Implement your feature by executing the PRP:
    ```bash
    /execute-prp PRPs/your-feature-name.md
    ```
*   The AI assistant will:
    1.  Read the PRP context.
    2.  Create an implementation plan.
    3.  Execute steps with validation.
    4.  Run tests and fix issues.
    5.  Ensure all requirements are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be detailed and specific.
    *   ❌ "Build a web scraper"
    *   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Use the `examples/` folder.
    *   Include code patterns in `examples/`.
    *   Reference specific files and patterns.
    *   Explain what to mimic.

**DOCUMENTATION**: Include relevant resources.
    *   API documentation URLs.
    *   Library guides.
    *   Server documentation.
    *   Database schemas.

**OTHER CONSIDERATIONS**: Capture important details.
    *   Authentication.
    *   Rate limits.
    *   Common pitfalls.
    *   Performance.

## The PRP Workflow

### How /generate-prp Works

1.  **Research Phase**
    *   Analyzes codebase.
    *   Searches for implementations.
    *   Identifies conventions.
2.  **Documentation Gathering**
    *   Fetches API docs.
    *   Includes library documentation.
    *   Adds gotchas.
3.  **Blueprint Creation**
    *   Creates step-by-step plan.
    *   Includes validation gates.
    *   Adds test requirements.
4.  **Quality Check**
    *   Scores confidence level.
    *   Ensures context inclusion.

### How /execute-prp Works

1.  **Load Context**: Reads the PRP.
2.  **Plan**: Creates a detailed task list.
3.  **Execute**: Implements each component.
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Fixes issues.
6.  **Complete**: Ensures requirements are met.
*   See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example.

## Using Examples Effectively

*   The `examples/` folder is **critical**.
*   AI performs better with visible patterns.

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
*   Be specific and comprehensive.
*   Include requirements, constraints, and examples.

### 2. Provide Comprehensive Examples
*   More examples = better implementations.
*   Show what to do AND what not to do.
*   Include error handling.

### 3. Use Validation Gates
*   PRPs use test commands.
*   AI iterates until all validations pass.
*   Ensures working code on first try.

### 4. Leverage Documentation
*   Include official API docs.
*   Add server resources.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md
*   Add your conventions and project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)