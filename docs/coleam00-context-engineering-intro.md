# Context Engineering: Revolutionizing AI Coding with Comprehensive Context

**Tired of frustrating AI coding assistant results? Context Engineering provides the structured context your AI needs, leading to 10x better results than prompt engineering.**  [View the original repository](https://github.com/coleam00/context-engineering-intro)

## Key Features of Context Engineering

*   **Improved AI Performance:** Provide comprehensive context for consistent and reliable AI code generation.
*   **Project Consistency:** Enforce coding standards and project-specific rules for uniform output.
*   **Complex Feature Development:** Enable AI assistants to handle multi-step implementations effectively.
*   **Self-Correcting Code:** Implement validation loops and testing for improved code quality and debugging.
*   **Streamlined Workflow:** Leverage templates and automation to simplify the AI coding process.

## Quick Start

Follow these steps to get started with Context Engineering:

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Customize Project Rules (Optional)
# Edit CLAUDE.md to add project-specific guidelines, such as file size limits or code organization.

# 3. Add Code Examples (Highly Recommended)
# Place relevant code examples in the examples/ folder to help the AI assistant follow your patterns.

# 4. Create a Feature Request
# Edit INITIAL.md to describe the desired feature, including required functionality, documentation, and considerations.

# 5. Generate a PRP (Product Requirements Prompt)
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

Context Engineering moves beyond simple prompts, providing AI with a complete understanding of your project:

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Relies on carefully crafted prompts.
*   Limited in scope and effectiveness.

**Context Engineering:**

*   Provides a comprehensive system for context, including documentation, examples, and rules.
*   Enables AI to understand and follow project-specific patterns and conventions.

### Why Context Engineering Matters

1.  **Reduces AI Failures:**  Address the common causes of AI agent failures through comprehensive context.
2.  **Ensures Consistency:**  Enforce project patterns and conventions for uniform results.
3.  **Enables Complex Features:**  Empower AI to handle multi-step implementations with proper context.
4.  **Self-Correcting:**  Use validation and testing to enable AI to fix its own mistakes.

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

Customize `CLAUDE.md` to include project-wide rules that the AI assistant will follow:

*   Project awareness: Reading planning docs, checking tasks
*   Code structure: File size limits, module organization
*   Testing requirements: Unit test patterns, coverage expectations
*   Style conventions: Language preferences, formatting rules
*   Documentation standards: Docstring formats, commenting practices

Use the provided template as a starting point, and adapt it to your project's needs.

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to describe your desired feature, using the following structure:

```markdown
## FEATURE:
[Describe the feature with specific functionality and requirements]

## EXAMPLES:
[List and describe example files in the examples/ folder]

## DOCUMENTATION:
[Include links to relevant API documentation, libraries, or server resources]

## OTHER CONSIDERATIONS:
[Include considerations, such as authentication requirements, limitations, or common pitfalls]
```

See `INITIAL_EXAMPLE.md` for a complete example.

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

**Note:** The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:

*   `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
*   `.claude/commands/execute-prp.md` - See how it implements features from PRPs

This command will:

1.  Read your feature request.
2.  Research the codebase.
3.  Search for relevant documentation.
4.  Create a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Once generated, execute the PRP to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read all context from the PRP.
2.  Create a detailed implementation plan.
3.  Execute each step with validation.
4.  Run tests and fix any issues.
5.  Ensure all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Describe the feature with specific and comprehensive details:

*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Leverage the `/examples` folder:

*   Place relevant code patterns in `/examples`.
*   Reference specific files and patterns to follow.
*   Explain how each example applies.

**DOCUMENTATION**: Include all relevant resources:

*   API documentation URLs
*   Library guides
*   Database schemas
*   MCP server documentation

**OTHER CONSIDERATIONS**: Capture essential details:

*   Authentication requirements
*   Rate limits or quotas
*   Common pitfalls
*   Performance requirements

## The PRP Workflow

### How /generate-prp Works

The command follows this process:

1.  **Research Phase**

    *   Analyzes your codebase for patterns
    *   Searches for similar implementations
    *   Identifies conventions to follow
2.  **Documentation Gathering**

    *   Fetches relevant API docs
    *   Includes library documentation
    *   Adds gotchas and quirks
3.  **Blueprint Creation**

    *   Creates step-by-step implementation plan
    *   Includes validation gates
    *   Adds test requirements
4.  **Quality Check**

    *   Scores confidence level (1-10)
    *   Ensures all context is included

### How /execute-prp Works

1.  **Load Context**: Reads the entire PRP
2.  **Plan**: Creates detailed task list using TodoWrite
3.  **Execute**: Implements each component
4.  **Validate**: Runs tests and linting
5.  **Iterate**: Fixes any issues found
6.  **Complete**: Ensures all requirements met

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example of what gets generated.

## Using Examples Effectively

The `/examples/` folder is **critical** for success. Well-structured examples significantly improve AI coding results.

### What to Include in Examples

1.  **Code Structure Patterns**

    *   Module organization
    *   Import conventions
    *   Class/function patterns
2.  **Testing Patterns**

    *   Test file structure
    *   Mocking approaches
    *   Assertion styles
3.  **Integration Patterns**

    *   API client implementations
    *   Database connections
    *   Authentication flows
4.  **CLI Patterns**

    *   Argument parsing
    *   Output formatting
    *   Error handling

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

*   Be clear about your preferences.
*   Include specific requirements and constraints.
*   Reference examples liberally.

### 2. Provide Comprehensive Examples

*   More examples lead to better implementations.
*   Show both what to do AND what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates

*   PRPs include test commands that must pass.
*   The AI will iterate until all validations succeed.
*   This ensures working code on the first try.

### 4. Leverage Documentation

*   Include official API docs.
*   Add server resources.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md

*   Add your conventions.
*   Include project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)