# Context Engineering: Unlock AI Coding Efficiency 🚀

**Revolutionize your AI coding workflow with Context Engineering, a superior approach to prompt engineering that provides AI assistants with the complete context needed to build complex features efficiently.** ([View on GitHub](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **Comprehensive Context:** Provide AI with documentation, examples, rules, and validation to enable complex feature implementation.
*   **Reduced AI Failures:** Minimize agent failures by addressing context gaps, leading to more reliable results.
*   **Enhanced Consistency:** Ensure your AI assistant follows your project's specific patterns and conventions.
*   **Self-Correcting Workflow:** Utilize validation loops to allow the AI to automatically fix its mistakes.

## Getting Started

### 1. Clone the Template

```bash
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro
```

### 2. Set Project Rules (Optional)

Edit `CLAUDE.md` to establish your project's coding guidelines. (A template is provided.)

### 3. Add Code Examples (Recommended)

Place relevant code examples within the `examples/` folder to guide the AI assistant.

### 4. Create a Feature Request

Create a feature request in `INITIAL.md` to define your desired functionality.

### 5. Generate a Comprehensive PRP

Use the Claude Code command to generate a Product Requirements Prompt (PRP):

```bash
/generate-prp INITIAL.md
```

### 6. Execute the PRP

Implement your feature by running the following Claude Code command:

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

## What is Context Engineering?

Context Engineering is a superior alternative to prompt engineering.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Relies on clever wording and phrasing.
*   Limited by task phrasing.
*   Analogy: Giving someone a sticky note.

**Context Engineering:**

*   Provides a complete system of context.
*   Includes documentation, examples, rules, and validation.
*   Analogy: Writing a complete screenplay.

### Why Context Engineering Matters

1.  **Reduced AI Failures:** Addresses the primary causes of AI failures.
2.  **Consistency:** Enforces project-specific patterns and conventions.
3.  **Complex Features:** Enables the AI to handle multi-step implementations.
4.  **Self-Correction:** Leverages validation loops for AI self-correction.

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

The `CLAUDE.md` file defines project-wide rules. Customize this template to include:

*   Project awareness
*   Code structure guidelines
*   Testing requirements
*   Style conventions
*   Documentation standards

### 2. Create Your Initial Feature Request

Create a file named `INITIAL.md` describing the feature you want to build:

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

See `INITIAL_EXAMPLE.md` for an example.

### 3. Generate the PRP

Generate a Product Requirements Prompt (PRP) using Claude Code:

```bash
/generate-prp INITIAL.md
```

This command generates a comprehensive implementation blueprint.

### 4. Execute the PRP

Execute the PRP to implement the feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

This command reads the PRP, creates an implementation plan, and implements the feature.

## Writing Effective INITIAL.md Files

### Key Sections

*   **FEATURE:** Be clear and detailed about the desired functionality.
*   **EXAMPLES:** Leverage the `examples/` directory.
*   **DOCUMENTATION:** Include links to relevant resources.
*   **OTHER CONSIDERATIONS:** Note any critical requirements or constraints.

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase:** Analyzes the codebase and identifies patterns.
2.  **Documentation Gathering:** Fetches relevant API documentation.
3.  **Blueprint Creation:** Develops a step-by-step implementation plan.
4.  **Quality Check:** Evaluates confidence and ensures context is included.

### How `/execute-prp` Works

1.  Load Context.
2.  Plan.
3.  Execute.
4.  Validate.
5.  Iterate.
6.  Complete.

## Using Examples Effectively

The `examples/` directory is crucial for successful implementations.

### What to Include in Examples

1.  Code Structure Patterns
2.  Testing Patterns
3.  Integration Patterns
4.  CLI Patterns

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
### 2. Provide Comprehensive Examples
### 3. Use Validation Gates
### 4. Leverage Documentation
### 5. Customize `CLAUDE.md`

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)