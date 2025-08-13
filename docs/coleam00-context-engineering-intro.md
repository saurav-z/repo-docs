# Context Engineering Template: Revolutionizing AI Coding with Comprehensive Context

**Tired of generic prompt engineering? Context Engineering provides a powerful framework for building AI coding assistants that consistently deliver high-quality, complex features.** ([See Original Repo](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **Improved Accuracy**:  Context Engineering reduces AI failures by providing the necessary information to the AI.
*   **Consistent Code Quality**: Ensures AI-generated code adheres to your project's patterns, conventions, and style.
*   **Complex Feature Implementation**: Enables AI to handle multi-step tasks and advanced functionality.
*   **Self-Correcting Workflow**: Leverages validation loops for AI to fix its own mistakes, leading to reliable output.

## Quick Start

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up your project rules (Optional - Customize CLAUDE.md)
# Edit CLAUDE.md to add your project-specific guidelines

# 3. Add code examples (Highly Recommended)
# Place relevant code patterns in the examples/ folder

# 4. Create your feature request
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

Context Engineering moves beyond simple prompts to provide a comprehensive system for guiding AI coding assistants. It's the difference between a sticky note and a complete screenplay for your AI.

### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering**: Focuses on clever wording and phrasing; limited in scope.
*   **Context Engineering**: A complete system including documentation, examples, rules, and validation.

### Why Context Engineering Matters

1.  **Reduces AI Failures**:  Addresses context failures, the primary cause of AI assistant errors.
2.  **Ensures Consistency**:  Enforces project patterns, conventions, and standards.
3.  **Enables Complex Features**:  Allows AI to implement multi-step implementations effectively.
4.  **Self-Correcting**: Validation loops ensure AI fixes its own mistakes.

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

This template focuses on the core principles of Context Engineering.

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

Customize `CLAUDE.md` to establish project-wide rules for your AI assistant. The template includes sections for:

*   Project awareness
*   Code structure
*   Testing requirements
*   Style conventions
*   Documentation standards

**Modify the template in `CLAUDE.md` or create your own global guidelines.**

### 2. Create Your Initial Feature Request

Describe what you want to build in `INITIAL.md`. Use the following format:

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

**See `INITIAL_EXAMPLE.md` for a complete example.**

### 3. Generate the PRP

Product Requirements Prompts (PRPs) are blueprints for implementation, including:

*   Complete context and documentation
*   Implementation steps with validation
*   Error handling patterns
*   Test requirements

Run the following command in Claude Code:

```bash
/generate-prp INITIAL.md
```

This command performs the following steps:

1.  Analyzes codebase for patterns
2.  Searches for relevant documentation
3.  Creates a comprehensive PRP in `PRPs/your-feature-name.md`

### 4. Execute the PRP

Execute the generated PRP to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI coding assistant will:

1.  Read all context from the PRP
2.  Create a detailed implementation plan
3.  Execute each step with validation
4.  Run tests and fix any issues
5.  Ensure all success criteria are met

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific and comprehensive about the desired functionality and requirements.

**EXAMPLES**: Reference and describe the patterns demonstrated within code patterns located in the `examples/` folder.

**DOCUMENTATION**: Provide URLs to APIs, libraries, and relevant resources.

**OTHER CONSIDERATIONS**: Note any authentication, rate limits, common pitfalls, and performance requirements.

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase**: Analyzes codebase, searches for implementations and conventions.
2.  **Documentation Gathering**: Retrieves relevant API docs, and adds gotchas and quirks.
3.  **Blueprint Creation**: Creates step-by-step implementation plan, including validation and test requirements.
4.  **Quality Check**: Scores confidence and ensures all context is included.

### How `/execute-prp` Works

1.  **Load Context**: Reads the entire PRP
2.  **Plan**: Creates detailed task list using TodoWrite
3.  **Execute**: Implements each component
4.  **Validate**: Runs tests and linting
5.  **Iterate**: Fixes any issues found
6.  **Complete**: Ensures all requirements met

See `PRPs/EXAMPLE_multi_agent_prp.md` for a comprehensive example of what a PRP looks like.

## Using Examples Effectively

The `examples/` directory is *critical* for successful implementations. Code examples provide clear patterns for the AI assistant to follow.

### What to Include in Examples

1.  **Code Structure Patterns**: Module organization, import conventions, class/function patterns.
2.  **Testing Patterns**: Test file structure, mocking approaches, assertion styles.
3.  **Integration Patterns**: API client implementations, database connections, authentication flows.
4.  **CLI Patterns**: Argument parsing, output formatting, error handling.

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
-   Clearly define requirements and constraints.
-   Reference examples liberally.

### 2. Provide Comprehensive Examples
-   More examples lead to better implementations.
-   Include error handling patterns.

### 3. Use Validation Gates
-   PRPs include test commands that must pass.
-   AI iterates until validation succeeds.

### 4. Leverage Documentation
-   Include API documentation, server resources, and relevant documentation sections.

### 5. Customize CLAUDE.md
-   Add your coding conventions and project-specific rules.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)