# Context Engineering Template: Build AI-powered features with comprehensive context.

This template provides a robust framework for **Context Engineering**, a revolutionary approach to AI coding that surpasses traditional prompt engineering by providing AI assistants with all the information they need to generate accurate and complete code.

[View the original repository](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Comprehensive Context:** Provide detailed documentation, examples, and project rules for AI assistants.
*   **Automated Workflows:**  Generate and execute detailed implementation plans (PRPs) with `/generate-prp` and `/execute-prp`.
*   **Structured Template:** Leverage pre-defined `INITIAL.md` and example files for easy feature definition.
*   **Enhanced Consistency:** Ensure AI follows project-specific patterns, conventions, and coding standards.
*   **Reduced AI Failures:**  Minimize errors by providing the necessary context to AI coding assistants.
*   **Self-Correcting Code:**  Utilize validation loops to ensure code is tested and works.

## Getting Started

1.  **Clone the Template:**

    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```

2.  **Set Project Rules (Optional):** Customize `CLAUDE.md` with your project's specific guidelines and coding standards.

3.  **Add Examples (Recommended):** Place relevant code examples in the `examples/` folder to guide the AI assistant.

4.  **Create Feature Request:**  Define your desired feature in `INITIAL.md`.

5.  **Generate PRP:**  Use the `/generate-prp INITIAL.md` command within Claude Code to create a detailed Product Requirements Prompt (PRP).

6.  **Execute PRP:**  Implement the feature using `/execute-prp PRPs/your-feature-name.md` in Claude Code.

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering is a superior method to prompt engineering, allowing AI assistants to receive and use comprehensive context to write code:

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on carefully crafted prompts and phrasing.
*   Limited by the ability to express a task.
*   Think of this as a simple sticky note.

**Context Engineering:**

*   Employs a complete system for providing AI with thorough context.
*   Includes documentation, code examples, rules, patterns, and validation checks.
*   Think of this as a detailed screenplay with all the details necessary.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** AI failures stem from a lack of context more often than from model limitations.
2.  **Ensures Consistency:** AI adheres to your project's established patterns and conventions.
3.  **Enables Complex Features:**  AI can handle intricate, multi-step implementations with proper context.
4.  **Self-Correcting:** Validation loops allow AI to self-correct and fix mistakes.

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

### 1. Setting Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file houses project-wide rules, which the AI assistant will follow throughout conversations. The template includes:

*   **Project Awareness:** References planning documents and task definitions.
*   **Code Structure:** Adheres to file size limits and module organization guidelines.
*   **Testing Requirements:** Defines unit test patterns and code coverage expectations.
*   **Style Conventions:** Specifies preferred language and formatting rules.
*   **Documentation Standards:** Dictates docstring formats and commenting practices.

**Customize the provided template in `CLAUDE.md` to align with your project's requirements.**

### 2. Create Your Feature Request

In `INITIAL.md`, describe your desired feature:

```markdown
## FEATURE:
[Describe the desired feature, including its functionality and specific requirements]

## EXAMPLES:
[List code examples from the examples/ folder, along with their intended usage]

## DOCUMENTATION:
[Include relevant documentation, API references, or other resources]

## OTHER CONSIDERATIONS:
[Note any specifics, special needs, or common errors the AI assistant may encounter]
```

**Refer to `INITIAL_EXAMPLE.md` for a comprehensive example.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are like comprehensive blueprints for implementation. They include:

*   Complete context and documentation.
*   Implementation steps that include validation.
*   Error handling strategies.
*   Test requirements.

Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

**Note:** The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:
- `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
- `.claude/commands/execute-prp.md` - See how it implements features from PRPs

This command will:

1.  Analyze your feature request.
2.  Research the codebase for patterns.
3.  Search for related documentation.
4.  Generate a PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Once generated, execute the PRP to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI coding assistant will:

1.  Read all context from the PRP.
2.  Create a detailed implementation plan.
3.  Execute each step with validation.
4.  Run tests and address any issues.
5.  Ensure all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific and comprehensive.
*   ❌ "Build a web scraper"
*   ✅ "Build an asynchronous web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in a PostgreSQL database"

**EXAMPLES**: Utilize the `examples/` folder.

*   Include code patterns in `examples/`.
*   Reference specific files and coding patterns.
*   Explain what aspects the assistant should emulate.

**DOCUMENTATION**: Include all relevant resources.

*   Provide API documentation URLs.
*   Include library guides.
*   Provide database schemas.

**OTHER CONSIDERATIONS**: Capture important details.

*   Specify authentication requirements.
*   Note rate limits or quotas.
*   Highlight common pitfalls.
*   Specify performance requirements.

## The PRP Workflow

### How /generate-prp Works

The command follows this process:

1.  **Research Phase**

    *   Analyzes your codebase for patterns.
    *   Searches for similar implementations.
    *   Identifies conventions to follow.
2.  **Documentation Gathering**

    *   Fetches relevant API documentation.
    *   Includes library documentation.
    *   Adds gotchas and quirks.
3.  **Blueprint Creation**

    *   Creates a step-by-step implementation plan.
    *   Includes validation gates.
    *   Adds test requirements.
4.  **Quality Check**

    *   Scores a confidence level (1-10).
    *   Ensures all necessary context is included.

### How /execute-prp Works

1.  **Load Context:** Reads the entire PRP.
2.  **Plan:** Creates a detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes any problems found.
6.  **Complete:** Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for an example of what gets generated.

## Using Examples Effectively

The `examples/` folder is **critical** for ensuring success. AI coding assistants perform significantly better with pattern examples.

### What to Include in Examples

1.  **Code Structure Patterns**

    *   How you organize modules.
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

*   Clearly state your preferences and requirements.
*   Provide detailed constraints.
*   Reference examples frequently.

### 2. Provide Comprehensive Examples

*   More examples result in better implementations.
*   Demonstrate both what to do and what to avoid.
*   Include error handling patterns.

### 3. Use Validation Gates

*   PRPs include test commands that must pass.
*   The AI iterates until all validations succeed.
*   This ensures functional code from the beginning.

### 4. Leverage Documentation

*   Incorporate official API documentation.
*   Add MCP server resources.
*   Cite specific documentation sections.

### 5. Customize CLAUDE.md

*   Add your project conventions.
*   Incorporate project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)