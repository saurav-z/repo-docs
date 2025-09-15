# Context Engineering: Supercharge Your AI Coding Assistant 🚀

**Unlock the power of AI coding assistants with Context Engineering – a comprehensive approach that surpasses prompt engineering and traditional methods.** Explore this innovative template for building advanced AI-powered workflows. For the original repository, check it out here: [Context Engineering Intro](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Enhanced AI Performance:** Significantly reduces AI failures by providing comprehensive context, leading to more reliable and accurate code generation.
*   **Consistent Code Generation:** Enforces project patterns, conventions, and best practices, ensuring uniformity across your codebase.
*   **Complex Feature Implementation:** Enables AI to handle multi-step implementations with ease, thanks to detailed context and guidance.
*   **Self-Correcting Capabilities:** Utilizes validation loops to empower AI to identify and correct its own mistakes, improving code quality.

## Quick Start Guide

1.  **Clone the Template:**
    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```

2.  **Set Up Project Rules (Optional):** Edit `CLAUDE.md` to define project-specific guidelines, coding standards, and conventions.
3.  **Add Code Examples (Highly Recommended):** Place relevant code examples in the `examples/` folder to guide the AI's implementation.
4.  **Create Feature Request:** Describe the desired functionality in `INITIAL.md`, including requirements, examples, and documentation.
5.  **Generate PRP (Product Requirements Prompt):**
    In Claude Code, run:
    ```bash
    /generate-prp INITIAL.md
    ```
6.  **Execute the PRP:**
    In Claude Code, run:
    ```bash
    /execute-prp PRPs/your-feature-name.md
    ```

## Table of Contents

-   [What is Context Engineering?](#what-is-context-engineering)
-   [Template Structure](#template-structure)
-   [Step-by-Step Guide](#step-by-step-guide)
-   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
-   [The PRP Workflow](#the-prp-workflow)
-   [Using Examples Effectively](#using-examples-effectively)
-   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering provides a superior approach to traditional prompt engineering, offering comprehensive context and guidance for AI coding assistants.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on crafting clever prompts.
*   Limited by the phrasing of the task.

**Context Engineering:**

*   Provides a complete system for comprehensive context.
*   Includes documentation, examples, rules, patterns, and validation.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses context-related failures, the primary cause of agent errors.
2.  **Ensures Consistency:** Ensures that AI adheres to your project's established patterns.
3.  **Enables Complex Features:** Facilitates the implementation of multi-step features with appropriate context.
4.  **Self-Correcting:** Utilizes validation loops for AI-driven error correction.

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

This template doesn't focus on RAG and tools with context engineering because I have a LOT more in store for that soon. ;)

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

Define project-wide rules in `CLAUDE.md`, covering areas like:

*   Project awareness
*   Code structure
*   Testing requirements
*   Style conventions
*   Documentation standards

Customize the provided template to align with your project's specific needs.

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to describe the desired feature, including:

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

Refer to `INITIAL_EXAMPLE.md` for a complete example.

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints designed for AI coding assistants. Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

This command:

1.  Reads your feature request.
2.  Researches the codebase.
3.  Searches for documentation.
4.  Generates a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Implement your feature by executing the generated PRP:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read context from the PRP.
2.  Create a detailed implementation plan.
3.  Execute each step, including validation.
4.  Run tests and fix any issues.
5.  Ensure success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE:** Be specific about functionality and requirements.

**EXAMPLES:** Leverage examples in the `examples/` folder and explain how to use them.

**DOCUMENTATION:** Include relevant API documentation, guides, and resources.

**OTHER CONSIDERATIONS:** Specify any unique requirements or potential challenges.

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase:** Analyzes your codebase for patterns, searches for similar implementations, and identifies conventions.
2.  **Documentation Gathering:** Fetches relevant API documentation and adds related information.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan with validation and test requirements.
4.  **Quality Check:** Assesses confidence level and ensures all context is included.

### How `/execute-prp` Works

1.  **Load Context:** Reads the PRP.
2.  **Plan:** Creates a detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes any identified issues.
6.  **Complete:** Ensures requirements are met.

## Using Examples Effectively

The `examples/` folder is essential for guiding the AI assistant.

### What to Include in Examples

1.  Code structure patterns.
2.  Testing patterns.
3.  Integration patterns.
4.  CLI patterns.

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

*   Clearly state requirements and constraints.
*   Reference relevant examples.

### 2. Provide Comprehensive Examples

*   Offer numerous examples.
*   Demonstrate both correct and incorrect approaches.
*   Include error-handling patterns.

### 3. Use Validation Gates

*   PRPs include test commands that must pass.
*   AI iterates until validation succeeds.
*   Ensures working code.

### 4. Leverage Documentation

*   Include official API docs.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md

*   Add your project's conventions.
*   Include project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)