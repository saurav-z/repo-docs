# Context Engineering: Revolutionizing AI Code Generation 🚀

**Tired of basic prompt engineering? Context Engineering provides a complete system for guiding AI coding assistants, resulting in more reliable, complex, and consistent code.** Explore the future of AI-assisted development with this powerful template. [See the original repo](https://github.com/coleam00/context-engineering-intro) for the source code and further information.

## Key Features

*   **Comprehensive Context:** Provide AI assistants with all the information they need, from documentation to examples and project-specific rules.
*   **Reduced AI Failures:** Minimize errors by giving your AI the context it needs to succeed.
*   **Consistent Code:** Ensure your AI follows your project's established patterns and conventions.
*   **Simplified Complex Tasks:** Tackle intricate, multi-step implementations with ease.
*   **Self-Correcting Implementation:** Leverage validation loops to allow the AI to fix its own mistakes.

## Quick Start

Get up and running with Context Engineering in a few simple steps:

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up your project rules (optional - template provided)
# Edit CLAUDE.md to add your project-specific guidelines

# 3. Add examples (highly recommended)
# Place relevant code examples in the examples/ folder

# 4. Create your initial feature request
# Edit INITIAL.md with your feature requirements

# 5. Generate a comprehensive PRP (Product Requirements Prompt)
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

Context Engineering transforms AI code generation by providing complete context, unlike traditional prompt engineering.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:** Focuses on clever wording and specific phrasing, limiting its scope.  It's like giving someone a sticky note.

**Context Engineering:**  Provides a comprehensive system, including documentation, examples, rules, and validation.  It's like writing a full screenplay with all the details.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Most agent failures are due to context failures, not model failures.
2.  **Ensures Consistency:** AI follows project patterns and conventions.
3.  **Enables Complex Features:** AI handles multi-step implementations with proper context.
4.  **Self-Correcting:** Validation loops allow the AI to fix mistakes.

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

This template currently focuses on core Context Engineering principles, with more advanced features coming soon.

## Step-by-Step Guide

Follow these steps to start leveraging Context Engineering:

### 1. Set Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file contains project-wide rules that the AI assistant will follow in every conversation. Configure things such as:

*   **Project Awareness:** Reading planning docs, checking tasks
*   **Code Structure:** File size limits, module organization
*   **Testing Requirements:** Unit test patterns, coverage expectations
*   **Style Conventions:** Language preferences, formatting rules
*   **Documentation Standards:** Docstring formats, commenting practices

Customize the provided template to fit your project's needs.

### 2. Create Your Initial Feature Request

Create and edit `INITIAL.md` to describe what you want to build. Include:

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

See `INITIAL_EXAMPLE.md` for a complete example.

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints for your AI assistant, including:

*   Complete context and documentation
*   Implementation steps with validation
*   Error handling patterns
*   Test requirements

They are similar to PRDs (Product Requirements Documents) but are crafted more specifically to instruct an AI coding assistant.

Run in Claude Code:
```bash
/generate-prp INITIAL.md
```

**Note:** The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:
-   `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
-   `.claude/commands/execute-prp.md` - See how it implements features from PRPs

The `$ARGUMENTS` variable in these commands receives whatever you pass after the command name (e.g., `INITIAL.md` or `PRPs/your-feature.md`).

This command will:

1.  Read your feature request
2.  Research the codebase for patterns
3.  Search for relevant documentation
4.  Create a comprehensive PRP in `PRPs/your-feature-name.md`

### 4. Execute the PRP

Implement your feature by executing the generated PRP:

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

Follow these guidelines for optimal feature request creation:

### Key Sections Explained

**FEATURE**: Be specific and comprehensive
    -   ❌ "Build a web scraper"
    -   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Leverage the `examples/` folder
    -   Place relevant code patterns in `examples/`
    -   Reference specific files and patterns to follow
    -   Explain what aspects should be mimicked

**DOCUMENTATION**: Include all relevant resources
    -   API documentation URLs
    -   Library guides
    -   MCP server documentation
    -   Database schemas

**OTHER CONSIDERATIONS**: Capture important details
    -   Authentication requirements
    -   Rate limits or quotas
    -   Common pitfalls
    -   Performance requirements

## The PRP Workflow

Understand how `/generate-prp` and `/execute-prp` operate for effective use.

### How /generate-prp Works

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

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example of a generated PRP.

## Using Examples Effectively

The `examples/` directory is crucial for Context Engineering success. Well-crafted examples provide a clear blueprint for your AI coding assistant.

### What to Include in Examples

1.  **Code Structure Patterns**
    *   How you organize modules
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

Follow these recommendations for the best results:

### 1. Be Explicit in INITIAL.md
    -   Don't assume the AI knows your preferences
    -   Include specific requirements and constraints
    -   Reference examples liberally

### 2. Provide Comprehensive Examples
    -   More examples = better implementations
    -   Show both what to do AND what not to do
    -   Include error handling patterns

### 3. Use Validation Gates
    -   PRPs include test commands that must pass
    -   AI will iterate until all validations succeed
    -   This ensures working code on first try

### 4. Leverage Documentation
    -   Include official API docs
    -   Add MCP server resources
    -   Reference specific documentation sections

### 5. Customize CLAUDE.md
    -   Add your conventions
    -   Include project-specific rules
    -   Define coding standards

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)