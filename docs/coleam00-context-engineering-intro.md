# Context Engineering: The Future of AI Code Assistants

**Tired of clunky prompt engineering? Context Engineering provides a complete system for building AI-powered coding assistants that understand your project's nuances, leading to more consistent and complex feature implementations.** ([Original Repo](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **Comprehensive Context:** Move beyond prompts by providing documentation, examples, rules, and validation for your AI assistant.
*   **Reduced AI Failures:** Minimize model failures by providing the necessary information.
*   **Enhanced Consistency:** Ensure AI follows your project's patterns and conventions.
*   **Enabled Complexity:** Handle multi-step implementations with proper context.
*   **Self-Correcting:** Validation loops allow the AI to fix its own mistakes.

## Quick Start

1.  **Clone the Template:**
    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```
2.  **(Optional) Set Project Rules:** Edit `CLAUDE.md` to define project-specific guidelines.
3.  **Add Examples (Highly Recommended):** Place relevant code examples in the `examples/` folder.
4.  **Create Feature Request:** Edit `INITIAL.md` with your feature requirements.
5.  **Generate a PRP:** In Claude Code, run:
    ```bash
    /generate-prp INITIAL.md
    ```
6.  **Execute the PRP:** In Claude Code, run:
    ```bash
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

Context Engineering empowers AI coding assistants by providing a complete and structured system for understanding your project.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on clever wording and specific phrasing.
*   Limited to how you phrase a task.
*   Analogy: Giving someone a sticky note.

**Context Engineering:**

*   A complete system for providing comprehensive context.
*   Includes documentation, examples, rules, patterns, and validation.
*   Analogy: Writing a full screenplay with all the details.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Most agent failures are context failures.
2.  **Ensures Consistency:** AI follows your project patterns and conventions.
3.  **Enables Complex Features:** AI can handle multi-step implementations with proper context.
4.  **Self-Correcting:** Validation loops allow AI to fix its own mistakes.

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

*This template focuses on the core principles of context engineering and will expand to include features for RAG and tools in the future.*

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

`CLAUDE.md` contains project-wide rules for your AI assistant, including:

*   Project awareness (reading docs, checking tasks)
*   Code structure (file size limits, module organization)
*   Testing requirements (unit test patterns, coverage)
*   Style conventions (language preferences, formatting)
*   Documentation standards (docstring formats, commenting)

**Customize the provided template in `CLAUDE.md` for your project.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to describe your feature. Use the `INITIAL_EXAMPLE.md` example as a guide.

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

### 3. Generate the PRP (Product Requirements Prompt)

PRPs are comprehensive implementation blueprints. In Claude Code, run:

```bash
/generate-prp INITIAL.md
```

This command:
1.  Reads your feature request.
2.  Researches your codebase for patterns.
3.  Searches for relevant documentation.
4.  Creates a PRP in `PRPs/your-feature-name.md`.

The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:
- `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
- `.claude/commands/execute-prp.md` - See how it implements features from PRPs

### 4. Execute the PRP

Implement your feature by running:

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

**FEATURE**: Be specific and comprehensive.

*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Leverage the `examples/` folder.

*   Place code patterns in `examples/`.
*   Reference specific files and patterns to follow.
*   Explain aspects to be mimicked.

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

1.  **Research:** Analyzes codebase for patterns, searches for similar implementations, identifies conventions.
2.  **Documentation Gathering:** Fetches API docs, includes library documentation, adds gotchas.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan with validation gates and test requirements.
4.  **Quality Check:** Scores confidence level and ensures all context is included.

### How `/execute-prp` Works

1.  **Load Context:** Reads the entire PRP.
2.  **Plan:** Creates a detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes any issues.
6.  **Complete:** Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example.

## Using Examples Effectively

The `examples/` folder is **critical** for success. Examples enable your AI to learn your project's style.

### What to Include in Examples

1.  **Code Structure Patterns:** How you organize modules, import conventions, class/function patterns.
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

*   Don't assume the AI knows your preferences.
*   Include specific requirements and constraints.
*   Reference examples liberally.

### 2. Provide Comprehensive Examples

*   More examples = better implementations.
*   Show both what to do AND what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates

*   PRPs include test commands that must pass.
*   AI will iterate until all validations succeed.
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