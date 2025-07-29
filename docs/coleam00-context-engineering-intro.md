# Context Engineering Template: Build AI-Powered Features with Ease

**Revolutionize your AI coding with Context Engineering, providing comprehensive context for AI assistants, resulting in more successful and complex feature implementations.**  Learn more and contribute to the original repo [here](https://github.com/coleam00/context-engineering-intro).

## Key Features:

*   **Comprehensive Context:** Provides AI assistants with the necessary information (documentation, examples, rules) for end-to-end feature implementation.
*   **Reduced AI Failures:** Addresses the primary cause of AI agent failures - lack of context.
*   **Ensured Consistency:** Enforces project patterns and conventions for standardized code.
*   **Enables Complex Features:** Facilitates multi-step implementations with proper context.
*   **Self-Correcting:** Integrates validation loops for AI to identify and correct errors.

## Getting Started

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set Project Rules (Optional but Recommended)
# Edit CLAUDE.md to tailor project-specific guidelines

# 3. Add Relevant Examples (Crucial)
# Place code examples in the examples/ directory

# 4. Create a Feature Request
# Edit INITIAL.md to detail feature requirements

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
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering surpasses traditional Prompt Engineering by offering a holistic system for providing AI coding assistants with the context they need.

### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering:** Focuses on prompt wording, limited by phrasing capabilities. Like giving someone a sticky note.
*   **Context Engineering:** Provides complete context via documentation, examples, rules, patterns, and validation. Like writing a full screenplay with all the details.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses context failures, the leading cause of AI agent errors.
2.  **Ensures Consistency:** Enforces project patterns and conventions.
3.  **Enables Complex Features:** Facilitates multi-step implementations.
4.  **Self-Correcting:** Implements validation loops allowing for AI-driven error correction.

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
├── examples/                  # Your code examples
├── CLAUDE.md                 # Global rules for AI assistant
├── INITIAL.md               # Template for feature requests
├── INITIAL_EXAMPLE.md       # Example feature request
└── README.md                # This file
```

*Note: This template will be updated with more features soon.*

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

Customize the `CLAUDE.md` file to include your project-specific rules for consistent AI assistant behavior.  The template includes sections for:

*   Project Awareness
*   Code Structure
*   Testing Requirements
*   Style Conventions
*   Documentation Standards

**Customize the provided template to align with your project's needs.**

### 2. Create Your Initial Feature Request (INITIAL.md)

Define what you want to build by editing `INITIAL.md`:

```markdown
## FEATURE:
[Describe what you want to build - be specific]

## EXAMPLES:
[List and explain the use of example files in examples/ folder]

## DOCUMENTATION:
[Include links to documentation, APIs, or resources]

## OTHER CONSIDERATIONS:
[Include requirements, and gotchas]
```

**See `INITIAL_EXAMPLE.md` for a comprehensive example.**

### 3. Generate the PRP

Generate a Product Requirements Prompt (PRP) with the following command:

```bash
/generate-prp INITIAL.md
```

PRPs are comprehensive blueprints that guide AI assistants in implementing features.  The `/generate-prp` command will:

1.  Analyze your codebase for patterns.
2.  Search for relevant documentation.
3.  Create a PRP in `PRPs/your-feature-name.md`.

**Note:** Custom commands defined in `.claude/commands/`.

### 4. Execute the PRP

Implement your feature by running:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read all context from the PRP.
2.  Create an implementation plan.
3.  Implement steps, with validation.
4.  Run tests and fix issues.
5.  Ensure all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections

**FEATURE:** Be explicit and detailed.
*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES:** Reference relevant code patterns in the `examples/` folder. Explain the desired behaviors.

**DOCUMENTATION:** Provide relevant API documentation URLs, guides, and database schemas.

**OTHER CONSIDERATIONS:** Capture essential details like authentication requirements, rate limits, common pitfalls, and performance requirements.

## The PRP Workflow

### How /generate-prp Works

1.  **Research Phase:** Analyzes codebase for patterns and conventions.
2.  **Documentation Gathering:** Retrieves documentation, APIs, and any gotchas.
3.  **Blueprint Creation:** Generates a step-by-step implementation plan with validation and test requirements.
4.  **Quality Check:** Assesses confidence level and ensures context inclusion.

### How /execute-prp Works

1.  Load Context.
2.  Plan.
3.  Execute.
4.  Validate.
5.  Iterate.
6.  Complete.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example.

## Using Examples Effectively

The `examples/` folder is essential for success.  Examples enable the AI assistant to learn and apply your coding patterns.

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

*   Include specific requirements and constraints.
*   Reference examples liberally.

### 2. Provide Comprehensive Examples

*   Show both what to do and what not to do.
*   Include error-handling patterns.

### 3. Use Validation Gates

*   PRPs include test commands.
*   The AI iterates until all validations succeed.

### 4. Leverage Documentation

*   Include official API docs.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md

*   Add your conventions.
*   Include project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)