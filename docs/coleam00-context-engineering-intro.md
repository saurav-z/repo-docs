# Context Engineering Template: Revolutionize AI Coding with Comprehensive Context

**Unlock the full potential of AI coding assistants by providing the context they need to succeed, achieving results far beyond simple prompt engineering.** This template empowers you to build complex features with consistency, reliability, and efficiency.  [View the original repository here](https://github.com/coleam00/context-engineering-intro).

**Key Features:**

*   ✅ **Comprehensive Context:** Provide AI with all the information it needs - documentation, examples, rules, and validation.
*   ✅ **Reduced AI Failures:** Significantly minimize errors by addressing context failures.
*   ✅ **Consistent Code:** Ensure your AI assistant adheres to your project's patterns and conventions.
*   ✅ **Complex Feature Implementation:** Enable AI to handle intricate, multi-step implementations effectively.
*   ✅ **Self-Correcting Workflow:** Leverage validation loops for AI to identify and fix its own mistakes.

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Quick Start Guide](#quick-start)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering moves beyond traditional prompt engineering by providing comprehensive context.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on clever wording.
*   Limited by how you phrase a task.
*   Like giving someone a sticky note.

**Context Engineering:**

*   A complete system for providing comprehensive context.
*   Includes documentation, examples, rules, patterns, and validation.
*   Like writing a full screenplay with all the details.

### Why Context Engineering Matters

1.  **Reduce AI Failures:** Most agent failures stem from context limitations, not model shortcomings.
2.  **Ensure Consistency:** Align AI-generated code with your project's established patterns and conventions.
3.  **Enable Complex Features:** Empower AI to handle multi-step implementations.
4.  **Self-Correcting:** Validation loops allow the AI to fix its own mistakes.

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

This template doesn't focus on RAG and tools with context engineering because I have a LOT more in store for that soon. ;)

## Quick Start Guide

```bash
# 1. Clone this template
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

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file defines project-wide rules for your AI assistant. The template includes:

*   **Project Awareness:** Directs the AI to read planning documents and check tasks.
*   **Code Structure:** Defines file size limits and module organization guidelines.
*   **Testing Requirements:** Sets unit test patterns and coverage expectations.
*   **Style Conventions:** Specifies language preferences and formatting rules.
*   **Documentation Standards:** Outlines docstring formats and commenting practices.

**Customize `CLAUDE.md` to align with your project's specific needs.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to detail what you want to build:

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

**Refer to `INITIAL_EXAMPLE.md` for a comprehensive example.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are detailed implementation blueprints:

*   Complete context and documentation.
*   Step-by-step implementation with validation.
*   Error-handling patterns.
*   Test requirements.

They function like PRDs (Product Requirements Documents) but are tailored for AI coding assistants.

Run in Claude Code:
```bash
/generate-prp INITIAL.md
```

**Note:** The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:
- `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
- `.claude/commands/execute-prp.md` - See how it implements features from PRPs

The `$ARGUMENTS` variable in these commands receives whatever you pass after the command name (e.g., `INITIAL.md` or `PRPs/your-feature.md`).

This command will:
1.  Read your feature request.
2.  Analyze the codebase for existing patterns.
3.  Search for relevant documentation.
4.  Generate a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

After generating the PRP, execute it to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI coding assistant will:

1.  Load all context from the PRP.
2.  Create a detailed implementation plan.
3.  Execute each step with built-in validation.
4.  Run tests and rectify any issues.
5.  Ensure all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be explicit and provide detailed requirements.
*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Leverage the `examples/` folder.
*   Place code patterns in `examples/`.
*   Refer to specific files and the patterns to follow.
*   Explain the aspects the AI should emulate.

**DOCUMENTATION**: Include all relevant resources.
*   API documentation URLs.
*   Library guides.
*   MCP server documentation.
*   Database schemas.

**OTHER CONSIDERATIONS**: Capture important details.
*   Authentication requirements.
*   Rate limits and quotas.
*   Common pitfalls.
*   Performance requirements.

## The PRP Workflow

### How /generate-prp Works

The command follows this process:

1.  **Research Phase**
    *   Analyzes your codebase for patterns.
    *   Searches for similar implementations.
    *   Identifies conventions to follow.

2.  **Documentation Gathering**
    *   Fetches relevant API docs.
    *   Includes library documentation.
    *   Adds gotchas and quirks.

3.  **Blueprint Creation**
    *   Creates step-by-step implementation plan.
    *   Includes validation gates.
    *   Adds test requirements.

4.  **Quality Check**
    *   Scores confidence level (1-10).
    *   Ensures all context is included.

### How /execute-prp Works

1.  **Load Context**: Reads the entire PRP.
2.  **Plan**: Creates detailed task list using TodoWrite.
3.  **Execute**: Implements each component.
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Fixes any issues found.
6.  **Complete**: Ensures all requirements met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example of what gets generated.

## Using Examples Effectively

The `examples/` folder is **crucial** for successful AI implementation. AI coding assistants perform best when they have clear patterns to follow.

### What to Include in Examples

1.  **Code Structure Patterns**
    *   How you organize modules.
    *   Import conventions.
    *   Class/function patterns.

2.  **Testing Patterns**
    *   Test file structure.
    *   Mocking approaches.
    *   Assertion styles.

3.  **Integration Patterns**
    *   API client implementations.
    *   Database connections.
    *   Authentication flows.

4.  **CLI Patterns**
    *   Argument parsing.
    *   Output formatting.
    *   Error handling.

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
*   Don't assume the AI knows your preferences.
*   Include specific requirements and constraints.
*   Reference examples liberally.

### 2. Provide Comprehensive Examples
*   More examples lead to better implementations.
*   Show both what to do AND what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates
*   PRPs include test commands that must pass.
*   AI will iterate until all validations succeed.
*   This ensures working code on the first try.

### 4. Leverage Documentation
*   Include official API docs.
*   Add MCP server resources.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md
*   Add your coding conventions.
*   Include project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)