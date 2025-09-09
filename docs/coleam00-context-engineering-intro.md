# Context Engineering: Build AI-Powered Applications with Precision

**Unlock the power of AI by providing comprehensive context for your coding assistants, leading to more reliable, consistent, and complex implementations.**  ([View Original Repo](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **10x Improvement:** Context Engineering surpasses prompt engineering in effectiveness.
*   **Reduced AI Failures:** Minimize errors with robust context and validation.
*   **Consistent Code:** Ensure your AI follows your project's style and conventions.
*   **Complex Implementations:** Enables multi-step features through detailed instructions.
*   **Self-Correcting:** Validation loops allow AI to resolve its own mistakes.

## Quick Start

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set Project Rules (optional, in CLAUDE.md)
# Customize your project guidelines in CLAUDE.md

# 3. Add Code Examples (highly recommended!)
# Place your relevant examples in the examples/ directory

# 4. Create Feature Request (INITIAL.md)
# Describe your feature requirements in INITIAL.md

# 5. Generate a Product Requirements Prompt (PRP)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP
# In Claude Code, run:
/execute-prp PRPs/your-feature-name.md
```

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective Feature Requests (INITIAL.md)](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering provides a complete system for giving coding assistants the information needed to get the job done. It goes beyond prompt engineering by including:

*   **Documentation**
*   **Code Examples**
*   **Project Rules**
*   **Coding Patterns**
*   **Validation Mechanisms**

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on clever phrasing.
*   Limited by wording choices.

**Context Engineering:**

*   Provides comprehensive context for coding assistants.
*   Enables complex features and consistent code.

### Why Context Engineering Matters

1.  **Reduce AI Failures:** Address context failures for more successful implementations.
2.  **Ensure Consistency:** Enforce project patterns and conventions.
3.  **Enable Complex Features:** Handle multi-step implementations efficiently.
4.  **Self-Correction:** Validation loops enable the AI to correct its mistakes.

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

### 1.  Set Up Global Project Rules (CLAUDE.md)

Customize `CLAUDE.md` with your project-wide rules:

*   **Project Awareness:** Reading documentation and checking tasks.
*   **Code Structure:** File size limits, module organization.
*   **Testing Requirements:** Unit test patterns, coverage expectations.
*   **Style Conventions:** Language preferences, formatting rules.
*   **Documentation Standards:** Docstring formats, commenting practices.

### 2.  Create Your Feature Request (INITIAL.md)

Define what you want to build in `INITIAL.md`:

```markdown
## FEATURE:
[Describe your desired functionality and requirements specifically.]

## EXAMPLES:
[List examples and how they should be used.]

## DOCUMENTATION:
[Include links to relevant documentation.]

## OTHER CONSIDERATIONS:
[Mention any special requirements or considerations.]
```

See `INITIAL_EXAMPLE.md` for a sample feature request.

### 3.  Generate the Product Requirements Prompt (PRP)

PRPs are detailed implementation blueprints used to instruct the AI. In Claude Code, run:

```bash
/generate-prp INITIAL.md
```

This command:

1.  Reads your feature request.
2.  Analyzes your codebase for patterns.
3.  Searches for relevant documentation.
4.  Creates a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4.  Execute the PRP

Once generated, execute the PRP to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read all context from the PRP.
2.  Create a detailed implementation plan.
3.  Execute each step and validate.
4.  Run tests and address any issues.
5.  Ensure success criteria are met.

## Writing Effective Feature Requests (INITIAL.md)

### Key Sections Explained

**FEATURE:** Be clear and detailed. Example: "*Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL.*"

**EXAMPLES:** Utilize the `examples/` directory to show code patterns and how they should be used.

**DOCUMENTATION:** Include links to all relevant API documentation, library guides, and database schemas.

**OTHER CONSIDERATIONS:** Include authentication requirements, rate limits, common pitfalls, and performance needs.

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase:** Analyzes codebase for patterns.
2.  **Documentation Gathering:** Fetches relevant API documentation.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan.
4.  **Quality Check:** Ensures all context is included.

### How `/execute-prp` Works

1.  **Load Context:** Reads the PRP.
2.  **Plan:** Creates detailed task list using TodoWrite
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes any issues found.
6.  **Complete:** Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example of what gets generated.

## Using Examples Effectively

The `examples/` folder is **crucial** for the success of your projects.

### What to Include in Examples

1.  **Code Structure Patterns:** Module organization, import conventions, etc.
2.  **Testing Patterns:** Test file structure, mocking approaches, assertion styles.
3.  **Integration Patterns:** API client implementations, database connections, authentication flows.
4.  **CLI Patterns:** Argument parsing, output formatting, error handling.

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

*   Include specific requirements and constraints.
*   Reference examples often.

### 2. Provide Comprehensive Examples

*   Include both what to do and what *not* to do.
*   Show error handling.

### 3. Use Validation Gates

*   PRPs use test commands.
*   Iterate until validations pass.

### 4. Leverage Documentation

*   Include API docs.
*   Add server resources.

### 5. Customize CLAUDE.md

*   Add your conventions.
*   Include project-specific rules.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)