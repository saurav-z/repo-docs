# Context Engineering: Revolutionize AI Coding with Comprehensive Context

**Unlock 10x better coding results by equipping AI with the complete context it needs to succeed.**

[View the original repository](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Comprehensive Context:** Provides AI with all necessary information: documentation, examples, rules, and validation.
*   **Reduced AI Failures:** Addresses the root cause of many AI agent failures: context gaps.
*   **Consistent & Reliable Code:** Ensures adherence to project patterns, conventions, and style.
*   **Enables Complex Implementations:** Allows AI to handle intricate, multi-step tasks with ease.
*   **Self-Correcting Workflow:** Integrated validation loops empower AI to identify and resolve its own errors.

## Getting Started: Context Engineering Template

This template provides a streamlined workflow for implementing Context Engineering, a superior approach to prompt engineering.

### 1. Clone the Template

```bash
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro
```

### 2. Define Project Rules (Optional)

Customize `CLAUDE.md` to specify project-wide guidelines:

*   Project awareness
*   Code structure
*   Testing requirements
*   Style conventions
*   Documentation standards

### 3. Provide Code Examples (Recommended)

Place relevant code examples in the `examples/` folder to guide the AI assistant.

### 4. Create Your Feature Request

Describe the desired feature in `INITIAL.md`.

### 5. Generate a Product Requirements Prompt (PRP)

Run the following command in Claude Code:

```bash
/generate-prp INITIAL.md
```

This command generates a comprehensive implementation blueprint for your feature.

### 6. Execute the PRP

Implement the feature using:

```bash
/execute-prp PRPs/your-feature-name.md
```

## Table of Contents

-   [What is Context Engineering?](#what-is-context-engineering)
    -   [Prompt Engineering vs. Context Engineering](#prompt-engineering-vs-context-engineering)
    -   [Why Context Engineering Matters](#why-context-engineering-matters)
-   [Template Structure](#template-structure)
-   [Step-by-Step Guide](#step-by-step-guide)
-   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
-   [The PRP Workflow](#the-prp-workflow)
    -   [How /generate-prp Works](#how-generate-prp-works)
    -   [How /execute-prp Works](#how-execute-prp-works)
-   [Using Examples Effectively](#using-examples-effectively)
-   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering moves beyond simple prompt engineering by providing the AI with a rich understanding of your project.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on crafting clever wording
*   Limited in scope to the prompt's phrasing
*   Analogous to a sticky note

**Context Engineering:**

*   A complete system for providing comprehensive context.
*   Includes documentation, examples, rules, patterns, and validation.
*   Like writing a detailed screenplay with all the necessary information.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses the cause of many AI agent failures.
2.  **Ensures Consistency:** AI adheres to your project's established patterns and conventions.
3.  **Enables Complex Features:** Empowers AI to handle complex tasks efficiently.
4.  **Self-Correcting:** Validation loops allows AI to fix its own mistakes.

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

Customize `CLAUDE.md` to define project-wide rules for your AI assistant. The template includes:

*   Project awareness
*   Code structure
*   Testing requirements
*   Style conventions
*   Documentation standards

**Customize the provided template to your project's needs.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to define your feature requirements:

```markdown
## FEATURE:
[Describe the desired functionality and its requirements]

## EXAMPLES:
[List and explain relevant files from the examples/ folder]

## DOCUMENTATION:
[Include links to relevant documentation, APIs, or server resources]

## OTHER CONSIDERATIONS:
[Note any special requirements or common pitfalls]
```

**See `INITIAL_EXAMPLE.md` for a comprehensive example.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints.

Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

This command:

1.  Analyzes your feature request
2.  Researches the codebase for patterns
3.  Searches for relevant documentation
4.  Creates a PRP in `PRPs/your-feature-name.md`

### 4. Execute the PRP

Implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read the PRP
2.  Create a detailed implementation plan
3.  Execute each step with validation
4.  Run tests and fix any issues
5.  Ensure all success criteria are met

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE:** Specify the exact desired functionality.
*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES:** Leverage the `examples/` folder to demonstrate code patterns.
*   Reference specific files and their use.
*   Explain patterns to be followed.

**DOCUMENTATION:** Include all relevant resources.
*   API documentation URLs
*   Library guides
*   Database schemas

**OTHER CONSIDERATIONS:** Capture important details.
*   Authentication details
*   Rate limits
*   Performance expectations

## The PRP Workflow

### How /generate-prp Works

This command follows these steps:

1.  **Research Phase:** Analyzes codebase for patterns, searches for similar implementations, and identifies conventions.

2.  **Documentation Gathering:** Fetches and includes relevant API documentation.

3.  **Blueprint Creation:** Creates a step-by-step implementation plan including validation gates and test requirements.

4.  **Quality Check:** Scores the confidence level and ensures all context is included.

### How /execute-prp Works

1.  **Load Context:** Reads the entire PRP.
2.  **Plan:** Creates detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes any issues.
6.  **Complete:** Ensures all requirements are met.

## Using Examples Effectively

The `examples/` folder is *essential* for success.

### What to Include in Examples

1.  **Code Structure Patterns:** Module organization, import conventions, and class/function patterns.
2.  **Testing Patterns:** Test file structure, mocking approaches, and assertion styles.
3.  **Integration Patterns:** API client implementations, database connections, and authentication flows.
4.  **CLI Patterns:** Argument parsing, output formatting, and error handling.

### Example Structure

```
examples/
├── README.md           # Explains each example
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
-   Include specific requirements and constraints.
-   Reference examples liberally.

### 2. Provide Comprehensive Examples
-   More examples = better implementations.
-   Demonstrate both what to do and *what not to do*.

### 3. Use Validation Gates
-   PRPs include test commands that must pass.

### 4. Leverage Documentation
-   Include official API docs and server resources.

### 5. Customize CLAUDE.md
-   Define your project-specific rules and standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)