# Context Engineering Template: Revolutionize AI Coding with Comprehensive Context

**Tired of subpar AI code? Context Engineering is your key to robust and reliable AI-assisted software development!**  Check out the original repo [here](https://github.com/coleam00/context-engineering-intro).

Context Engineering empowers AI coding assistants by providing them with the complete information needed to deliver end-to-end, high-quality solutions, leading to significantly improved results over prompt engineering.

## Key Features

*   **Comprehensive Context:** Provide AI with rules, examples, documentation, and more for precise guidance.
*   **Reduced AI Failures:** Significantly decrease agent errors by addressing context gaps.
*   **Consistency & Adherence:** Enforce project-specific coding standards and patterns automatically.
*   **Enable Complex Features:** Facilitate the implementation of intricate, multi-step features.
*   **Self-Correction & Validation:** Implement built-in validation loops for automated error fixing.

## Quick Start

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set project rules (optional) - Edit CLAUDE.md
# Add project-specific guidelines in CLAUDE.md to guide the AI assistant

# 3. Add examples (recommended)
# Place your code examples in the examples/ folder

# 4. Create a feature request
# Edit INITIAL.md with your feature requirements

# 5. Generate a Product Requirements Prompt (PRP)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP to implement your feature
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

Context Engineering transforms AI coding by providing a complete system for comprehensive context.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on clever wording and phrasing.
*   Limited to how you phrase a task.

**Context Engineering:**

*   A complete system for providing comprehensive context, like writing a full screenplay.
*   Includes documentation, examples, rules, patterns, and validation.

### Why Context Engineering Matters

1.  **Reduce AI Failures**: Address context failures, not just model failures.
2.  **Ensure Consistency**: Enforce your project's established patterns and standards.
3.  **Enable Complex Features**: Facilitate the implementation of multi-step implementations.
4.  **Self-Correcting**: Implement validation loops for AI-driven error fixing.

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
│   └── EXAMPLE_multi_agent_prp.md  # Example PRP
├── examples/                  # Your code examples (critical!)
├── CLAUDE.md                 # Global project rules
├── INITIAL.md               # Feature request template
├── INITIAL_EXAMPLE.md       # Example feature request
└── README.md                # This file
```

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

`CLAUDE.md` contains project-wide rules that guide the AI assistant. The template includes:

*   Project awareness: Reading planning docs, checking tasks
*   Code structure: File size limits, module organization
*   Testing requirements: Unit test patterns, coverage expectations
*   Style conventions: Language preferences, formatting rules
*   Documentation standards: Docstring formats, commenting practices

**Customize this template for your project.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to describe the feature:

```markdown
## FEATURE:
[Describe what you want to build - be specific about functionality and requirements]

## EXAMPLES:
[List example files and explain how they should be used]

## DOCUMENTATION:
[Include links to relevant docs, APIs, or server resources]

## OTHER CONSIDERATIONS:
[Mention any gotchas, specific requirements, or things AI assistants commonly miss]
```

**See `INITIAL_EXAMPLE.md` for an example.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive blueprints that include:

*   Complete context and documentation
*   Implementation steps with validation
*   Error handling patterns
*   Test requirements

Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

This command will:
1.  Read your feature request
2.  Research the codebase for patterns
3.  Search for relevant documentation
4.  Create a comprehensive PRP in `PRPs/your-feature-name.md`

### 4. Execute the PRP

Execute the PRP to implement the feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read all context from the PRP
2.  Create a detailed implementation plan
3.  Execute each step with validation
4.  Run tests and fix any issues
5.  Ensure all success criteria are met

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific and comprehensive.

*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Leverage examples in the `examples/` folder.

*   Reference specific files and patterns to follow.
*   Explain what aspects should be mimicked.

**DOCUMENTATION**: Include all relevant resources.

*   API documentation URLs
*   Library guides
*   Server documentation
*   Database schemas

**OTHER CONSIDERATIONS**: Capture important details.

*   Authentication requirements
*   Rate limits or quotas
*   Common pitfalls
*   Performance requirements

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase**: Analyze your codebase for patterns, search for similar implementations, and identify conventions.
2.  **Documentation Gathering**: Fetch relevant API docs and include library documentation.
3.  **Blueprint Creation**: Create a step-by-step implementation plan, include validation gates, and add test requirements.
4.  **Quality Check**: Score confidence levels (1-10) and ensure all context is included.

### How `/execute-prp` Works

1.  **Load Context**: Reads the entire PRP.
2.  **Plan**: Creates a detailed task list.
3.  **Execute**: Implements each component.
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Fixes any issues found.
6.  **Complete**: Ensures all requirements are met.

## Using Examples Effectively

The `examples/` folder is **critical** for success.

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
├── README.md           # Explains each example
├── cli.py             # CLI implementation
├── agent/             # Agent architecture
│   ├── agent.py      # Agent creation
│   ├── tools.py      # Tool implementation
│   └── providers.py  # Multi-provider
└── tests/            # Testing
    ├── test_agent.py # Unit test patterns
    └── conftest.py   # Pytest configuration
```

## Best Practices

### 1. Be Explicit in INITIAL.md

*   Provide specific requirements and constraints.
*   Reference examples frequently.

### 2. Provide Comprehensive Examples

*   More examples = better implementations.
*   Show both what to do AND what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates

*   PRPs include test commands that must pass.
*   AI will iterate until all validations succeed.

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