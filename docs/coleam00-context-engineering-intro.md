# Context Engineering: Build AI-Powered Features with Precision

**Context Engineering is a superior approach to prompt engineering, providing a robust framework for building and deploying complex AI solutions.** Explore this template to revolutionize how you develop software with AI. ([Original Repo](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **Comprehensive Context:** Provide AI assistants with all necessary information for end-to-end feature implementation.
*   **Reduced AI Failures:** Minimize errors by equipping agents with complete context.
*   **Consistent Results:** Enforce project-specific patterns and coding standards.
*   **Simplified Complex Tasks:** Enable AI to manage multi-step processes with validation.
*   **Self-Correcting Capabilities:** Utilize validation loops to ensure accuracy and quality.

## 🚀 Quick Start

1.  **Clone the Template:**

    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```

2.  **Set Up Project Rules (Optional):**

    *   Edit `CLAUDE.md` to define your project's conventions.

3.  **Add Code Examples (Highly Recommended):**

    *   Place relevant code patterns in the `examples/` folder.

4.  **Create a Feature Request:**

    *   Edit `INITIAL.md` to describe your desired feature.

5.  **Generate a Product Requirements Prompt (PRP):**

    *   In Claude Code, run: `/generate-prp INITIAL.md`

6.  **Execute the PRP to Implement Your Feature:**

    *   In Claude Code, run: `/execute-prp PRPs/your-feature-name.md`

## 📚 Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering moves beyond prompt engineering to provide a complete system for instructing AI coding assistants.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:** Focuses on clever phrasing, like a sticky note.

**Context Engineering:** Provides comprehensive context including documentation, examples, rules, patterns, and validation, like writing a full screenplay.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses context failures, which are the leading cause of agent failures.
2.  **Ensures Consistency:** Enforces project patterns and conventions.
3.  **Enables Complex Features:** Allows AI to implement multi-step features with proper context.
4.  **Self-Correcting:** Uses validation loops for self-improvement.

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

This template focuses on providing a foundation for context engineering, and future improvements regarding RAG and tool integration are planned.

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

Define project-wide rules for the AI assistant in `CLAUDE.md`, including:

*   Project awareness
*   Code structure
*   Testing requirements
*   Style conventions
*   Documentation standards

Customize the provided template to meet your project's needs.

### 2. Create Your Initial Feature Request

Describe what you want to build in `INITIAL.md`.

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

See `INITIAL_EXAMPLE.md` for an example feature request.

### 3. Generate the PRP

PRPs are detailed implementation blueprints that includes:

*   Complete context and documentation
*   Implementation steps with validation
*   Error handling patterns
*   Test requirements

Run in Claude Code:
```bash
/generate-prp INITIAL.md
```

The `/generate-prp` command:

1.  Reads your feature request
2.  Researches the codebase for patterns
3.  Searches for relevant documentation
4.  Creates a comprehensive PRP in `PRPs/your-feature-name.md`

### 4. Execute the PRP

Implement your feature by executing the PRP.

```bash
/execute-prp PRPs/your-feature-name.md
```

The `/execute-prp` command:

1.  Reads all context from the PRP
2.  Creates a detailed implementation plan
3.  Executes each step with validation
4.  Runs tests and fixes any issues
5.  Ensures all success criteria are met

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE:** Clearly define the feature's functionality and requirements.

**EXAMPLES:** Reference and explain relevant code patterns in the `examples/` folder.

**DOCUMENTATION:** Include links to all essential resources.

**OTHER CONSIDERATIONS:** Capture any special requirements or details the AI assistant should consider.

## The PRP Workflow

### How /generate-prp Works

1.  **Research Phase:** Analyzes your codebase for patterns.
2.  **Documentation Gathering:** Fetches and includes relevant documentation.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan with validation and testing.
4.  **Quality Check:** Scores confidence and ensures all context is included.

### How /execute-prp Works

1.  **Load Context:** Reads the PRP.
2.  **Plan:** Creates a detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes any issues.
6.  **Complete:** Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example.

## Using Examples Effectively

The `examples/` folder is **critical** for the success of your implementation.

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

### 1. Be Explicit in INITIAL.md

*   Provide specific requirements and constraints.
*   Reference examples.

### 2. Provide Comprehensive Examples

*   Include both what to do and what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates

*   Include test commands that must pass.
*   AI iterates until validations succeed.

### 4. Leverage Documentation

*   Include official API documentation.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md

*   Add your conventions and project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)