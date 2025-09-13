# Context Engineering: The Future of AI Code Generation

Context Engineering revolutionizes how AI assistants code, providing comprehensive context to eliminate failures and deliver high-quality, consistent results.  [Learn more at the original repository](https://github.com/coleam00/context-engineering-intro).

## Key Features

*   **Comprehensive Context:** Go beyond basic prompting and provide detailed documentation, examples, and rules.
*   **Reduced AI Failures:** Minimize errors by giving your AI the context it needs to succeed.
*   **Consistent Code:** Enforce project-specific patterns, conventions, and coding standards.
*   **Complex Feature Implementation:** Enables AI to handle multi-step implementations with built-in validation and self-correction.
*   **PRP Workflow:** Utilize Product Requirements Prompts to generate comprehensive implementation plans tailored for AI assistants.

## Quick Start

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
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering is a powerful approach to AI code generation that surpasses traditional prompt engineering by providing a complete system for delivering comprehensive context, leading to more robust and reliable AI-assisted development.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on crafting precise prompts.
*   Limited in scope, relying on clever phrasing.
*   Analogy: Giving someone a sticky note.

**Context Engineering:**

*   Provides extensive context including documentation, examples, rules, and validation.
*   Enables comprehensive, multi-step implementations.
*   Analogy: Writing a full screenplay.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses the root causes of most agent failures by providing thorough context.
2.  **Ensures Consistency:** Enforces your project's coding standards and conventions.
3.  **Enables Complex Features:** Allows AI to handle multi-step implementations effectively.
4.  **Self-Correcting:** Built-in validation mechanisms allow AI to iterate and fix its own mistakes.

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

This template provides a foundation for context engineering, and I have even more in store in the future.

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

`CLAUDE.md` sets project-wide rules for the AI assistant:

*   **Project Awareness:** Rules on reading documentation and completing tasks.
*   **Code Structure:** Guidelines on file size limits and module organization.
*   **Testing Requirements:** Instructions on unit tests and coverage expectations.
*   **Style Conventions:** Preferences for language, formatting, and other style choices.
*   **Documentation Standards:** Expectations for docstring formatting and commenting.

Customize the provided template in `CLAUDE.md` to tailor it to your project.

### 2. Create Your Initial Feature Request (INITIAL.md)

Describe what you want to build in `INITIAL.md`:

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

See `INITIAL_EXAMPLE.md` for a detailed example.

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints:

*   Complete context and documentation
*   Step-by-step implementation with validation
*   Error handling guidelines
*   Testing requirements

Run the following command in Claude Code:

```bash
/generate-prp INITIAL.md
```

The `/generate-prp` command:

1.  Reads your feature request
2.  Researches code for patterns
3.  Searches for relevant documentation
4.  Creates a comprehensive PRP in `PRPs/your-feature-name.md`

**Note:** The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:
- `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
- `.claude/commands/execute-prp.md` - See how it implements features from PRPs

### 4. Execute the PRP

Execute the PRP to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The `/execute-prp` command will:

1.  Read all context from the PRP
2.  Create a detailed implementation plan
3.  Execute each step with validation
4.  Run tests and fix any issues
5.  Ensure all success criteria are met

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE:** Be specific and comprehensive.

*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES:** Leverage the `examples/` folder.

*   Place code patterns in `examples/`
*   Reference specific files to follow
*   Explain the aspects to be emulated

**DOCUMENTATION:** Include relevant resources.

*   API documentation URLs
*   Library guides
*   MCP server documentation
*   Database schemas

**OTHER CONSIDERATIONS:** Capture crucial details.

*   Authentication requirements
*   Rate limits or quotas
*   Common pitfalls
*   Performance requirements

## The PRP Workflow

### How /generate-prp Works

1.  **Research Phase:**
    *   Analyzes your codebase for patterns.
    *   Searches for similar implementations.
    *   Identifies conventions to follow.
2.  **Documentation Gathering:**
    *   Fetches relevant API docs.
    *   Includes library documentation.
    *   Adds gotchas and quirks.
3.  **Blueprint Creation:**
    *   Creates a step-by-step implementation plan.
    *   Includes validation gates.
    *   Adds test requirements.
4.  **Quality Check:**
    *   Scores confidence level (1-10).
    *   Ensures all context is included.

### How /execute-prp Works

1.  **Load Context:** Reads the entire PRP.
2.  **Plan:** Creates detailed task list using TodoWrite.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes any issues found.
6.  **Complete:** Ensures all requirements met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example.

## Using Examples Effectively

The `examples/` folder is **critical** for success. AI coding assistants perform much better when they can see patterns to follow.

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

*   Don't make assumptions about what the AI knows.
*   Specify requirements and constraints.
*   Reference relevant examples.

### 2. Provide Comprehensive Examples

*   More examples lead to better implementations.
*   Show both what to do and what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates

*   PRPs include test commands that must pass.
*   The AI will iterate until all validations succeed.
*   This ensures working code on the first try.

### 4. Leverage Documentation

*   Include official API documentation.
*   Add relevant resources.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md

*   Add your project-specific conventions.
*   Include project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)