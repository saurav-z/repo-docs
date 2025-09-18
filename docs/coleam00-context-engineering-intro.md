# Context Engineering: Revolutionizing AI Coding with Comprehensive Context

**Tired of frustrating AI coding assistant results? Context Engineering provides a superior approach to prompt engineering by equipping your AI with the complete information needed for success.** [Learn more and get started with this template!](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Comprehensive Context:** Provide AI with documentation, examples, rules, and validation to ensure accurate and consistent results.
*   **Reduced Failures:** Minimize AI errors by addressing context failures, not just model limitations.
*   **Project Consistency:** Enforce your coding standards and patterns with AI-driven adherence.
*   **Complex Feature Development:** Enable AI to handle multi-step implementations with ease.
*   **Self-Correcting:** Utilize validation loops to empower AI to fix its own mistakes and deliver working code.

## Getting Started

### 1. Clone the Template

```bash
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro
```

### 2. Set Project Rules (Optional, but Recommended)

Customize `CLAUDE.md` to define project-specific guidelines for your AI assistant. This file governs code structure, testing, style, and documentation standards.

### 3. Add Code Examples (Essential)

Place relevant code patterns in the `examples/` folder.  This is crucial for guiding the AI's implementation.

### 4. Create Feature Requests

Describe your desired features in `INITIAL.md`. Use the provided template as a starting point.

### 5. Generate a Product Requirements Prompt (PRP)

Use the custom `/generate-prp` command within Claude Code to create a detailed implementation plan based on your feature request.

```bash
/generate-prp INITIAL.md
```

### 6. Execute the PRP

Run the `/execute-prp` command to have the AI implement your feature based on the generated PRP.

```bash
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
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering transcends prompt engineering. It's a holistic approach that provides AI with comprehensive context, unlike the limited nature of prompt-based approaches.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on crafting clever prompts and specific phrasing.
*   Limited by the words used in the prompt.
*   Comparable to a sticky note with limited instructions.

**Context Engineering:**

*   Provides a complete system of context, including comprehensive documentation, code examples, rules, design patterns, and validation steps.
*   Enables AI to follow complex instructions with greater precision.
*   Comparable to a detailed screenplay with all the relevant details.

### Why Context Engineering Matters

1.  **Reduced AI Failures:** Address context gaps, which are the primary cause of failures, rather than relying solely on model improvements.
2.  **Consistency:** Ensure AI consistently adheres to project standards, conventions, and patterns.
3.  **Enable Complex Features:** Empower AI to implement multi-step tasks with accuracy.
4.  **Self-Correction:** Utilize validation loops to enable AI to fix its own mistakes.

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

This template focuses on a streamlined approach to context engineering, with future expansion planned for RAG and other tools.

## Step-by-Step Guide

### 1. Set Up Global Rules (`CLAUDE.md`)

`CLAUDE.md` contains project-wide rules that the AI assistant follows in every interaction. The template includes example rules for:

*   **Project Awareness:** Reading planning documents, checking tasks
*   **Code Structure:** File size limits, module organization
*   **Testing Requirements:** Unit test patterns, coverage expectations
*   **Style Conventions:** Language preferences, formatting rules
*   **Documentation Standards:** Docstring formats, commenting practices

**Customize the provided template to your project's needs.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to specify the feature you want to build:

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

**See `INITIAL_EXAMPLE.md` for an example feature request.**

### 3. Generate the PRP (Product Requirements Prompt)

PRPs are comprehensive implementation blueprints, similar to Product Requirements Documents, tailored for AI coding assistants. They include:

*   Complete context and documentation
*   Step-by-step implementation with validation gates
*   Error handling patterns
*   Test requirements

Run the following in Claude Code:

```bash
/generate-prp INITIAL.md
```

**Note:** The slash commands are custom commands within `.claude/commands/`. Their implementations are found in:
- `.claude/commands/generate-prp.md` - Creates PRPs by researching and collecting information.
- `.claude/commands/execute-prp.md` - Implements features using the PRPs.

The `$ARGUMENTS` variable after the command name (e.g., `INITIAL.md`) is used to pass in data.
This command reads your feature request, researches your codebase for patterns, searches for relevant documentation, and creates a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Implement your feature by executing the PRP:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI will:
1.  Read all context from the PRP
2.  Create a detailed implementation plan
3.  Execute each step with validation
4.  Run tests and fix any issues
5.  Ensure all success criteria are met

## Writing Effective `INITIAL.md` Files

### Key Sections Explained

**FEATURE:** Be specific and comprehensive in your description.

*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES:** Utilize the `examples/` folder effectively.

*   Place relevant code patterns in `examples/`.
*   Reference specific files and patterns to guide the AI.
*   Clearly explain which aspects to emulate.

**DOCUMENTATION:** Include all relevant resources.

*   API documentation URLs
*   Library guides
*   Server documentation
*   Database schema definitions

**OTHER CONSIDERATIONS:** Address important details.

*   Authentication requirements
*   Rate limits or quotas
*   Common pitfalls and potential issues
*   Performance requirements

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase:**
    *   Analyzes your codebase for code patterns.
    *   Searches for similar implementations.
    *   Identifies conventions to adhere to.

2.  **Documentation Gathering:**
    *   Fetches relevant API documentation.
    *   Includes documentation for libraries.
    *   Adds gotchas and relevant considerations.

3.  **Blueprint Creation:**
    *   Creates a step-by-step implementation plan.
    *   Includes validation gates to verify accuracy.
    *   Adds test requirements.

4.  **Quality Check:**
    *   Assigns a confidence level (1-10).
    *   Ensures that all the required context is included.

### How `/execute-prp` Works

1.  **Load Context:** Reads the complete PRP.
2.  **Plan:** Creates a task list using the TodoWrite tool.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and performs linting.
5.  **Iterate:** Fixes any issues that are found.
6.  **Complete:** Verifies that all requirements have been met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for an example of a generated PRP.

## Using Examples Effectively

The `examples/` folder is **critical** for success.  AI coding assistants learn best from seeing patterns to follow.

### What to Include in Examples

1.  **Code Structure Patterns:**
    *   How you organize modules.
    *   Import conventions.
    *   Class/function patterns.

2.  **Testing Patterns:**
    *   Test file structure.
    *   Mocking approaches.
    *   Assertion styles.

3.  **Integration Patterns:**
    *   API client implementations.
    *   Database connections.
    *   Authentication flows.

4.  **CLI Patterns:**
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

### 1. Be Explicit in `INITIAL.md`

*   Don't assume the AI knows your preferences.
*   Include specific requirements and constraints.
*   Reference examples frequently.

### 2. Provide Comprehensive Examples

*   More examples = better implementations.
*   Show both what to do AND what *not* to do.
*   Include error handling patterns.

### 3. Use Validation Gates

*   PRPs incorporate test commands that must pass.
*   The AI will iterate until all validations succeed.
*   This ensures working code on the first try.

### 4. Leverage Documentation

*   Include official API documentation.
*   Add server resources (if applicable).
*   Reference specific sections of the documentation.

### 5. Customize `CLAUDE.md`

*   Add your coding conventions.
*   Include project-specific rules.
*   Define your coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)