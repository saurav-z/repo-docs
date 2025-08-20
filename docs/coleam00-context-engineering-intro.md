# Context Engineering: Revolutionize AI Coding with Comprehensive Context

**Stop wrestling with prompts and embrace Context Engineering – the future of AI-powered software development, ensuring consistent, complex features with fewer failures.** 
[Check out the original repo here!](https://github.com/coleam00/context-engineering-intro)

## Key Features & Benefits

*   **Comprehensive Context:** Provide AI assistants with all necessary information, including documentation, examples, rules, and validation.
*   **Reduced AI Failures:** Minimize agent errors by addressing context gaps.
*   **Consistent Code:** Ensure AI follows project patterns and conventions.
*   **Complex Feature Implementation:** Enable AI to handle multi-step implementations effectively.
*   **Self-Correcting Code:** Employ validation loops for AI to fix its own mistakes.

## Quick Start Guide

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up project rules (optional, template provided)
# Edit CLAUDE.md to add your project-specific guidelines

# 3. Add code examples (highly recommended)
# Place code examples in the examples/ folder

# 4. Create your initial feature request
# Edit INITIAL.md with feature requirements

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

Context Engineering is a superior approach to prompt engineering:

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**
*   Focuses on clever wording.
*   Limited by how a task is phrased.

**Context Engineering:**
*   Provides complete, comprehensive context.
*   Includes documentation, examples, rules, patterns, and validation.

### Why Context Engineering Matters

1.  **Reduced Failures:** Addresses context failures, the root cause of many AI agent errors.
2.  **Consistency:** Ensures the AI adheres to your project's patterns and conventions.
3.  **Complex Features:** Empowers AI to handle multi-step implementations.
4.  **Self-Correction:** Utilizes validation loops for AI to fix its own errors.

## Template Structure

```
context-engineering-intro/
├── .claude/
│   ├── commands/
│   │   ├── generate-prp.md    # Generates PRPs
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

*   `CLAUDE.md` defines project-wide rules for the AI assistant.
*   The template includes rules for:
    *   Project awareness
    *   Code structure
    *   Testing requirements
    *   Style conventions
    *   Documentation standards
*   Customize `CLAUDE.md` to fit your project's requirements.

### 2. Create Your Initial Feature Request

*   Edit `INITIAL.md` to detail what you want to build.
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
*   Refer to `INITIAL_EXAMPLE.md` for a complete example.

### 3. Generate the PRP

*   PRPs are comprehensive implementation blueprints.
*   Run the command in Claude Code:
    ```bash
    /generate-prp INITIAL.md
    ```
*   The command will:
    1.  Read your feature request.
    2.  Research the codebase.
    3.  Search for documentation.
    4.  Create a PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

*   Once the PRP is generated, run:
    ```bash
    /execute-prp PRPs/your-feature-name.md
    ```
*   The AI assistant will:
    1.  Read the entire PRP context.
    2.  Create an implementation plan.
    3.  Execute each step with validation.
    4.  Run tests and fix issues.
    5.  Ensure all requirements are met.

## Writing Effective INITIAL.md Files

### Key Sections

**FEATURE:** Be detailed and specific.
*   Good: "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES:** Leverage the `examples/` folder.
*   Reference and explain code patterns.

**DOCUMENTATION:** Include all relevant resources.
*   Include API documentation, library guides, and server resources.

**OTHER CONSIDERATIONS:** Capture important details.
*   Address authentication, rate limits, pitfalls, and performance requirements.

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research:** Analyze the codebase, search for similar implementations, and identify conventions.
2.  **Documentation Gathering:** Fetch relevant API documentation, include library documentation, and add considerations.
3.  **Blueprint Creation:** Create a step-by-step implementation plan, including validation gates and test requirements.
4.  **Quality Check:** Assess confidence level and ensure all context is included.

### How `/execute-prp` Works

1.  Load Context
2.  Plan
3.  Execute
4.  Validate
5.  Iterate
6.  Complete

## Using Examples Effectively

*   The `examples/` folder is crucial for success.
*   Examples help the AI assistant understand patterns.

### What to Include in Examples

1.  Code Structure
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
*   Specify requirements and constraints.
*   Reference examples extensively.

### 2. Provide Comprehensive Examples
*   Include both what to do and what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates
*   PRPs use test commands to validate code.
*   AI iterates until all validations pass.

### 4. Leverage Documentation
*   Include official API docs and server resources.

### 5. Customize CLAUDE.md
*   Add your own project conventions.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)