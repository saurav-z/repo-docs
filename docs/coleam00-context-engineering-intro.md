# Context Engineering: The Future of AI Coding

**Unlock the power of AI coding assistants with Context Engineering, providing comprehensive context for faster, more reliable development.** 🚀 [View the original repository](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Improved Accuracy:** Significantly reduces AI failures by providing complete context.
*   **Consistent Code:** Enforces project-specific patterns, conventions, and standards.
*   **Complex Implementations:** Enables AI to handle multi-step projects with ease.
*   **Self-Correcting:** Leverages validation loops for automatic error fixing.

## Getting Started

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set project rules (optional, template provided)
# Customize CLAUDE.md with your project guidelines.

# 3. Add code examples (highly recommended)
# Place relevant examples in the examples/ folder.

# 4. Create your feature request
# Edit INITIAL.md with your desired feature specifications.

# 5. Generate a comprehensive PRP (Product Requirements Prompt)
# Run in Claude Code: /generate-prp INITIAL.md

# 6. Implement your feature using the PRP
# Run in Claude Code: /execute-prp PRPs/your-feature-name.md
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

Context Engineering moves beyond prompt engineering by providing AI with comprehensive context for superior results.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**
*   Focuses on precise wording.
*   Limited to how you phrase the task.

**Context Engineering:**
*   Provides a complete system for context.
*   Includes documentation, examples, rules, and validation.

### Why Context Engineering Matters

1.  **Reduced AI Failures**: Addresses context gaps, not just model limitations.
2.  **Consistency**: Ensures AI follows project standards and conventions.
3.  **Complex Features**: Empowers AI to handle intricate multi-step implementations.
4.  **Self-Correction**: Uses validation loops to enable AI to correct its work.

## Template Structure

```
context-engineering-intro/
├── .claude/
│   ├── commands/
│   │   ├── generate-prp.md    # Generates PRPs
│   │   └── execute-prp.md     # Executes PRPs
│   └── settings.local.json    # Claude Code permissions
├── PRPs/
│   ├── templates/
│   │   └── prp_base.md       # PRP base template
│   └── EXAMPLE_multi_agent_prp.md  # Example PRP
├── examples/                  # Your code examples (critical!)
├── CLAUDE.md                 # Global rules
├── INITIAL.md               # Feature request template
├── INITIAL_EXAMPLE.md       # Example feature request
└── README.md                # This file
```

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

Customize `CLAUDE.md` with your project's global rules:

*   Project Awareness: Planning documents and task specifications.
*   Code Structure: File size limits and module organization guidelines.
*   Testing Requirements: Unit testing patterns and coverage expectations.
*   Style Conventions: Coding language preferences and formatting rules.
*   Documentation Standards: Docstring formats and commenting practices.

### 2. Create Your Initial Feature Request

Define what you want to build in `INITIAL.md`:

```markdown
## FEATURE:
[Detailed description of the desired functionality and requirements.]

## EXAMPLES:
[List and explain the role of example files in the examples/ folder.]

## DOCUMENTATION:
[Include links to relevant documentation, APIs, or server resources.]

## OTHER CONSIDERATIONS:
[Mention any constraints, specific requirements, or common challenges.]
```

Refer to `INITIAL_EXAMPLE.md` for a comprehensive example.

### 3. Generate the PRP (Product Requirements Prompt)

Generate a comprehensive PRP with a single Claude Code command:

```bash
/generate-prp INITIAL.md
```

The `/generate-prp` command:

1.  Analyzes the feature request.
2.  Researches the codebase for patterns.
3.  Gathers relevant documentation.
4.  Creates a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Implement your feature using the generated PRP:

```bash
/execute-prp PRPs/your-feature-name.md
```

The `/execute-prp` command:

1.  Reads the entire PRP.
2.  Creates an implementation plan.
3.  Executes each step, with validation.
4.  Runs tests and fixes any issues.
5.  Ensures that all requirements are met.

## Writing Effective INITIAL.md Files

### Key Sections

**FEATURE**:  Be specific and detailed about functionality and requirements.
**EXAMPLES**:  Reference code patterns from the `examples/` folder, clarifying their use.
**DOCUMENTATION**:  Include relevant API documentation, library guides, and server resources.
**OTHER CONSIDERATIONS**:  Address special requirements, authentication, and potential pitfalls.

## The PRP Workflow

### How /generate-prp Works

1.  **Research:** Analyzes your codebase and identifies relevant conventions.
2.  **Documentation:** Collects API documentation and includes potential gotchas.
3.  **Blueprint Creation:** Builds a step-by-step implementation plan with validations.
4.  **Quality Check:** Assesses confidence and ensures all context is present.

### How /execute-prp Works

1.  Load Context
2.  Plan
3.  Execute
4.  Validate
5.  Iterate
6.  Complete

## Using Examples Effectively

The `examples/` folder is essential for clear communication with the AI.

### What to Include in Examples

1.  Code Structure Patterns (module organization, imports).
2.  Testing Patterns (test file structure, mocking, assertions).
3.  Integration Patterns (API clients, database connections, authentication).
4.  CLI Patterns (argument parsing, output formatting, error handling).

### Example Structure

```
examples/
├── README.md           # Explains each example.
├── cli.py             # CLI implementation pattern.
├── agent/             # Agent architecture patterns.
│   ├── agent.py      # Agent creation pattern.
│   ├── tools.py      # Tool implementation pattern.
│   └── providers.py  # Multi-provider pattern.
└── tests/            # Testing patterns.
    ├── test_agent.py # Unit test patterns.
    └── conftest.py   # Pytest configuration.
```

## Best Practices

### 1. Be Explicit in INITIAL.md
Include specific requirements, constraints, and liberally reference your examples.

### 2. Provide Comprehensive Examples
More examples lead to better, more accurate implementations.

### 3. Use Validation Gates
PRPs contain test commands to ensure the code works correctly.

### 4. Leverage Documentation
Include all API documentation and internal resources.

### 5. Customize CLAUDE.md
Define project conventions and coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)