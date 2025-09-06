# Context Engineering: Revolutionizing AI Coding with Comprehensive Context

**Unlock the power of AI coding assistants by providing them with a comprehensive understanding of your project, going beyond simple prompts with context engineering.** ([Original Repo](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **Comprehensive Context:** Provide documentation, examples, rules, and validation for AI assistants.
*   **Reduced AI Failures:** Minimize errors by providing the necessary context.
*   **Ensured Consistency:** Ensure your AI assistant follows your project patterns and conventions.
*   **Enables Complex Features:** Allows AI to handle complex, multi-step implementations.
*   **Self-Correcting:** Implement validation loops for AI to fix its own mistakes.

## Getting Started

1.  **Clone the Template:**

    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```

2.  **Customize Project Rules (Optional):**

    *   Edit `CLAUDE.md` to add your project-specific guidelines.

3.  **Add Code Examples (Highly Recommended):**

    *   Place relevant code examples in the `examples/` folder.

4.  **Create a Feature Request:**

    *   Edit `INITIAL.md` with your feature requirements.

5.  **Generate a Product Requirements Prompt (PRP):**

    *   In Claude Code, run: `/generate-prp INITIAL.md`

6.  **Execute the PRP to Implement Your Feature:**

    *   In Claude Code, run: `/execute-prp PRPs/your-feature-name.md`

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering is a paradigm shift from traditional prompt engineering, focusing on providing comprehensive context to AI coding assistants.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on clever wording and specific phrasing.
*   Limited by how a task is phrased.

**Context Engineering:**

*   A complete system for providing comprehensive context, including documentation, examples, rules, patterns, and validation.
*   Enables AI assistants to understand project specifics.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** AI failures often stem from context gaps.
2.  **Ensures Consistency:** AI follows your project patterns and conventions.
3.  **Enables Complex Features:** AI can handle multi-step implementations with proper context.
4.  **Self-Correcting:** Validation loops allow AI to fix its mistakes.

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

The `CLAUDE.md` file sets project-wide rules for the AI assistant, including:

*   Project awareness, code structure, testing requirements, style conventions, and documentation standards.

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

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints that include:

*   Complete context and documentation
*   Implementation steps with validation
*   Error handling patterns
*   Test requirements

Run in Claude Code:
```bash
/generate-prp INITIAL.md
```

This command:

1.  Reads your feature request
2.  Researches the codebase for patterns
3.  Searches for relevant documentation
4.  Creates a comprehensive PRP in `PRPs/your-feature-name.md`

### 4. Execute the PRP

Run in Claude Code:
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

*   **FEATURE:** Be specific and comprehensive about what you want to build.
*   **EXAMPLES:** Leverage the `examples/` folder to reference patterns.
*   **DOCUMENTATION:** Include links to all relevant resources (API docs, library guides, etc.).
*   **OTHER CONSIDERATIONS:** Capture important details (authentication, rate limits, performance).

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase:** Analyzes codebase for patterns, searches for similar implementations.
2.  **Documentation Gathering:** Fetches relevant API docs and includes gotchas.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan with validation gates.
4.  **Quality Check:** Assesses confidence level and ensures context inclusion.

### How `/execute-prp` Works

1.  **Load Context**: Reads the entire PRP
2.  **Plan**: Creates detailed task list using TodoWrite
3.  **Execute**: Implements each component
4.  **Validate**: Runs tests and linting
5.  **Iterate**: Fixes any issues found
6.  **Complete**: Ensures all requirements met

## Using Examples Effectively

The `examples/` folder is **critical** for success, as it helps AI assistants learn patterns.

### What to Include in Examples

1.  **Code Structure Patterns**
2.  **Testing Patterns**
3.  **Integration Patterns**
4.  **CLI Patterns**

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

*   Include specific requirements and constraints, and reference examples.

### 2. Provide Comprehensive Examples

*   Show both what to do and what not to do, including error handling patterns.

### 3. Use Validation Gates

*   PRPs include test commands to ensure working code on the first try.

### 4. Leverage Documentation

*   Include official API docs and reference specific documentation sections.

### 5. Customize CLAUDE.md

*   Add your conventions and project-specific rules.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)