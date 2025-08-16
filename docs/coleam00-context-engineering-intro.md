# Context Engineering: Build AI-Powered Code with Comprehensive Context

**Context Engineering empowers you to build complex AI-driven software faster and more reliably by providing your AI coding assistant with the complete context it needs.**  For the source code, please visit the [original repository](https://github.com/coleam00/context-engineering-intro).

## Key Features

*   **Comprehensive Context:** Provide your AI with documentation, examples, and project-specific rules for accurate code generation.
*   **Reduce AI Failures:** Minimize errors by giving your AI the context it needs to succeed.
*   **Consistency & Standardization:** Ensure consistent code style and adherence to project conventions.
*   **Automated Workflow:** Leverage automated tools to generate detailed implementation plans (PRPs) and execute them.
*   **Self-Correcting Code:** Implement validation and testing to ensure accurate and reliable results.

## Getting Started: Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Customize Project Rules (Optional)
# Edit CLAUDE.md with project-specific coding conventions and guidelines

# 3. Add Code Examples (Highly Recommended)
# Place relevant code snippets in the examples/ folder

# 4. Define Your Feature Request
# Edit INITIAL.md with your requirements

# 5. Generate a Product Requirements Prompt (PRP)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP to Implement Your Feature
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

Context Engineering moves beyond basic prompt engineering by providing your AI coding assistant with a complete understanding of your project, leading to more reliable and complex AI-assisted development.

### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering:** Focuses on clever wording and phrasing. Limited to how you phrase a task.
*   **Context Engineering:** Provides a comprehensive system including documentation, examples, rules, patterns, and validation.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses context failures, not just model limitations.
2.  **Ensures Consistency:** Enforces project patterns and conventions.
3.  **Enables Complex Features:** Allows AI to handle multi-step implementations.
4.  **Self-Correcting:** Uses validation loops for error correction.

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

Customize `CLAUDE.md` to establish project-wide rules for your AI assistant. The provided template includes:

*   Project awareness
*   Code structure
*   Testing requirements
*   Style conventions
*   Documentation standards

### 2. Create Your Initial Feature Request (INITIAL.md)

Describe your desired feature in detail within `INITIAL.md`.

```markdown
## FEATURE:
[Describe your desired feature with specific details]

## EXAMPLES:
[List relevant examples from the examples/ folder]

## DOCUMENTATION:
[Include relevant documentation, API links]

## OTHER CONSIDERATIONS:
[Note any special requirements or potential pitfalls]
```

See `INITIAL_EXAMPLE.md` for a complete example.

### 3. Generate the PRP

Generate a Product Requirements Prompt (PRP) using the `/generate-prp` command in Claude Code:

```bash
/generate-prp INITIAL.md
```

The `/generate-prp` command reads your feature request and generates a detailed PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Implement your feature by running the `/execute-prp` command:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read all context from the PRP.
2.  Create an implementation plan.
3.  Execute each step with validation.
4.  Run tests and fix any issues.
5.  Ensure success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific about functionality and requirements.

**EXAMPLES**: Include and explain relevant code patterns from the `examples/` directory.

**DOCUMENTATION**: Provide links to relevant resources (APIs, libraries, etc.).

**OTHER CONSIDERATIONS**: Address authentication, rate limits, and other important details.

## The PRP Workflow

### How /generate-prp Works

1.  **Research Phase:** Analyze the codebase for patterns and conventions.
2.  **Documentation Gathering:** Fetch relevant documentation.
3.  **Blueprint Creation:** Create a detailed implementation plan with validation and test requirements.
4.  **Quality Check:** Assess confidence and ensure all context is included.

### How /execute-prp Works

1.  Load Context
2.  Plan
3.  Execute
4.  Validate
5.  Iterate
6.  Complete

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example.

## Using Examples Effectively

The `examples/` folder is crucial. Use it to demonstrate code patterns:

### What to Include in Examples

1.  Code Structure
2.  Testing
3.  Integration
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

### 1. Be Explicit in INITIAL.md

Provide precise requirements and constraints.

### 2. Provide Comprehensive Examples

Include detailed examples to guide the AI assistant.

### 3. Use Validation Gates

PRPs include test commands that ensure code works.

### 4. Leverage Documentation

Include API documentation and other resources.

### 5. Customize CLAUDE.md

Add project-specific rules and coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)