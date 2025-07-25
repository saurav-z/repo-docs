# Context Engineering Template: Build AI-Powered Solutions with Precision

Tired of frustrating AI coding assistants? **Context Engineering provides a robust framework to empower AI coding assistants, enabling them to deliver complete, accurate solutions by giving them the context they need.** 

[View the original repo](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Comprehensive Context:** Provide AI with documentation, examples, and rules.
*   **Reduced AI Failures:** Minimize agent errors through detailed context.
*   **Consistent Output:** Ensure adherence to project patterns and conventions.
*   **Complex Feature Implementation:** Enable multi-step implementations.
*   **Self-Correcting:** Validate and fix errors automatically.

## Getting Started

```bash
# 1. Clone the Template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set Project Rules (Optional)
# Edit CLAUDE.md to add project-specific guidelines (code style, testing)

# 3. Add Code Examples (Highly Recommended)
# Place relevant code examples in the examples/ directory

# 4. Create a Feature Request
# Edit INITIAL.md to describe desired features

# 5. Generate a PRP (Product Requirements Prompt)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP to Implement the Feature
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
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering is a more effective approach compared to prompt engineering. It provides comprehensive context and improves the performance of AI.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on carefully crafted prompts.
*   Limited in scope; relies on the user's ability to describe the task.
*   Like a sticky note.

**Context Engineering:**

*   Provides a complete system with all necessary context.
*   Includes documentation, examples, rules, and validation.
*   Like writing a detailed screenplay.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses context-related failures.
2.  **Ensures Consistency:** Adheres to project patterns and conventions.
3.  **Enables Complex Features:** Allows AI to handle multi-step implementations.
4.  **Self-Correcting:** Validation mechanisms allow AI to fix mistakes.

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
│   │   └── prp_base.md       # Base template for PRPs
│   └── EXAMPLE_multi_agent_prp.md  # Example of a complete PRP
├── examples/                  # Code Examples
├── CLAUDE.md                 # Global Rules
├── INITIAL.md               # Feature Request Template
├── INITIAL_EXAMPLE.md       # Example Feature Request
└── README.md                # This File
```

*This template focuses on core Context Engineering principles, with more advanced features in development.*

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file defines project-wide rules. Edit this file to include your project's:

*   Project Awareness
*   Code Structure
*   Testing Requirements
*   Style Conventions
*   Documentation Standards

**Use the template provided and customize it as needed.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to describe your desired feature.

```markdown
## FEATURE:
[Describe your desired feature, being as specific as possible]

## EXAMPLES:
[List examples and how they should be used]

## DOCUMENTATION:
[Include relevant documentation links]

## OTHER CONSIDERATIONS:
[Include any special requirements or common pitfalls]
```

**See `INITIAL_EXAMPLE.md` for a complete example.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints.

Run this command in Claude Code:

```bash
/generate-prp INITIAL.md
```

This command will:

1.  Analyze the codebase for patterns.
2.  Search for relevant documentation.
3.  Create a detailed PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Implement your feature by running this command in Claude Code:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI will:

1.  Read the PRP.
2.  Create an implementation plan.
3.  Execute each step with validation.
4.  Run tests and fix any issues.
5.  Ensure all requirements are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific and detailed.

*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Leverage code examples in the `examples/` folder.

*   Reference specific files and patterns.
*   Explain the aspects to mimic.

**DOCUMENTATION**: Include all relevant resources.

*   API documentation URLs
*   Library guides
*   Server documentation
*   Database schemas

**OTHER CONSIDERATIONS**: Capture all essential details.

*   Authentication
*   Rate limits
*   Pitfalls
*   Performance requirements

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase:** Analyze codebase, find patterns, identify conventions.
2.  **Documentation Gathering:** Fetch relevant API docs and add gotchas.
3.  **Blueprint Creation:** Create a step-by-step plan with validation and tests.
4.  **Quality Check:** Verify and ensure all necessary context is included.

### How `/execute-prp` Works

1.  Load Context
2.  Plan (using TodoWrite)
3.  Execute
4.  Validate
5.  Iterate
6.  Complete

See `PRPs/EXAMPLE_multi_agent_prp.md` for an example.

## Using Examples Effectively

The `examples/` folder is **critical** for your success. Well-crafted examples significantly improve the AI's performance.

### What to Include in Examples

1.  **Code Structure Patterns** (Modules, imports)
2.  **Testing Patterns** (Test structure, mocking, assertions)
3.  **Integration Patterns** (API clients, database connections, auth)
4.  **CLI Patterns** (Argument parsing, output, error handling)

### Example Structure

```
examples/
├── README.md           # Explains each example
├── cli.py             # CLI pattern
├── agent/             # Agent architecture
│   ├── agent.py      # Agent creation
│   ├── tools.py      # Tool implementation
│   └── providers.py  # Multi-provider
└── tests/            # Testing patterns
    ├── test_agent.py # Unit test
    └── conftest.py   # Pytest config
```

## Best Practices

### 1. Be Explicit in `INITIAL.md`
-   Specify requirements, constraints, and reference examples.

### 2. Provide Comprehensive Examples
-   More examples lead to better implementations.

### 3. Use Validation Gates
-   PRPs contain test commands for successful code.

### 4. Leverage Documentation
-   Include API docs and reference specific documentation sections.

### 5. Customize `CLAUDE.md`
-   Add your conventions and project-specific rules.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)