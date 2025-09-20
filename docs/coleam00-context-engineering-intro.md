# Context Engineering: Build AI-Powered Code with Precision and Efficiency

Context Engineering revolutionizes AI code generation by providing comprehensive context, resulting in more reliable and complex features than traditional prompt engineering. [View the original repository](https://github.com/coleam00/context-engineering-intro) for a deeper dive into this powerful approach.

## Key Features:

*   **Enhanced AI Reliability:** Reduce AI failures by providing the necessary context for successful code generation.
*   **Consistent Code Quality:** Enforce project-specific patterns, conventions, and standards for uniform code.
*   **Complex Feature Implementation:** Enable your AI to handle multi-step processes with detailed implementation plans and validation.
*   **Self-Correcting Capabilities:** Incorporate validation loops, so your AI can fix its own mistakes during development.

## Getting Started: A Quick Guide

1.  **Clone the Template:**
    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```

2.  **Customize Project Rules (Optional):** Edit `CLAUDE.md` to define your project-specific coding standards.

3.  **Add Code Examples (Highly Recommended):** Place relevant code examples in the `examples/` folder to guide the AI.

4.  **Create a Feature Request:** Describe your desired feature in `INITIAL.md`.

5.  **Generate a Comprehensive PRP (Product Requirements Prompt):**
    ```bash
    /generate-prp INITIAL.md
    ```

6.  **Execute the PRP to Implement:**
    ```bash
    /execute-prp PRPs/your-feature-name.md
    ```

## Table of Contents:

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering shifts from prompt engineering by providing complete context. Unlike prompt engineering, which relies on clever phrasing, Context Engineering offers a complete system.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on wording and specific phrasing.
*   Limited by the ability to phrase a task.
*   Think of it as giving a sticky note.

**Context Engineering:**

*   A complete system that provides comprehensive context.
*   Includes documentation, examples, rules, patterns, and validation.
*   Think of it as writing a full screenplay.

### Why Context Engineering Matters:

1.  **Reduces AI Failures:** Most agent failures are due to context failures, not model issues.
2.  **Ensures Consistency:** The AI adheres to your project's patterns and standards.
3.  **Enables Complex Features:** AI can handle multi-step implementations effectively with the proper context.
4.  **Self-Correcting:** Validation loops enable the AI to correct its mistakes.

## Template Structure:

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

## Step-by-Step Guide:

### 1. Set Up Global Rules (CLAUDE.md)

`CLAUDE.md` defines project-wide rules for the AI. The template includes:

*   Project awareness: Reading planning docs, checking tasks.
*   Code structure: File size limits, module organization.
*   Testing requirements: Unit test patterns, coverage expectations.
*   Style conventions: Language preferences, formatting rules.
*   Documentation standards: Docstring formats, commenting practices.

Customize the provided template to match your project's needs.

### 2. Create Your Initial Feature Request

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

See `INITIAL_EXAMPLE.md` for an example.

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints:

*   Complete context and documentation.
*   Implementation steps with validation.
*   Error handling patterns.
*   Test requirements.

Run in Claude Code:
```bash
/generate-prp INITIAL.md
```

This command will:

1.  Read your feature request.
2.  Research your codebase for patterns.
3.  Search for relevant documentation.
4.  Create a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

After generation, execute the PRP:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI will:

1.  Read all context from the PRP.
2.  Create a detailed implementation plan.
3.  Execute each step with validation.
4.  Run tests and fix issues.
5.  Ensure all success criteria are met.

## Writing Effective INITIAL.md Files:

### Key Sections Explained

**FEATURE**: Be specific and comprehensive:

*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Utilize the `examples/` folder:

*   Place relevant code patterns in `examples/`.
*   Reference specific files and patterns.
*   Explain what should be emulated.

**DOCUMENTATION**: Include all relevant resources:

*   API documentation URLs.
*   Library guides.
*   MCP server documentation.
*   Database schemas.

**OTHER CONSIDERATIONS**: Capture important details:

*   Authentication requirements.
*   Rate limits or quotas.
*   Common pitfalls.
*   Performance requirements.

## The PRP Workflow:

### How /generate-prp Works:

The command process:

1.  **Research Phase**: Analyzes codebase, searches for implementations, identifies conventions.
2.  **Documentation Gathering**: Fetches relevant API docs, includes library documentation, adds gotchas.
3.  **Blueprint Creation**: Creates a step-by-step implementation plan, including validation and test requirements.
4.  **Quality Check**: Scores confidence, ensures all context is included.

### How /execute-prp Works:

1.  **Load Context**: Reads the entire PRP.
2.  **Plan**: Creates a detailed task list.
3.  **Execute**: Implements each component.
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Fixes any issues found.
6.  **Complete**: Ensures all requirements met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a PRP example.

## Using Examples Effectively:

The `examples/` folder is **crucial** for success.

### What to Include in Examples:

1.  **Code Structure Patterns**: How you organize modules, import conventions, class/function patterns.
2.  **Testing Patterns**: Test file structure, mocking approaches, assertion styles.
3.  **Integration Patterns**: API client implementations, database connections, authentication flows.
4.  **CLI Patterns**: Argument parsing, output formatting, error handling.

### Example Structure:

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

## Best Practices:

### 1. Be Explicit in INITIAL.md:
- Don't make assumptions.
- Include specific requirements and constraints.
- Reference examples.

### 2. Provide Comprehensive Examples:
- The more examples, the better the implementations.
- Show both what to do and what not to do.
- Include error handling patterns.

### 3. Use Validation Gates:
- PRPs include test commands that must pass.
- AI will iterate until all validations succeed.
- This ensures working code.

### 4. Leverage Documentation:
- Include official API docs.
- Add resources and examples.
- Reference specific documentation.

### 5. Customize CLAUDE.md:
- Add your project conventions.
- Include specific rules.
- Define coding standards.

## Resources:

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)