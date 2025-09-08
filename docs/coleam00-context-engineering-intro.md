# Context Engineering: Supercharge AI Coding with Comprehensive Context

**Stop wrestling with basic prompt engineering and unlock 10x the power of AI coding assistants with Context Engineering!**  This template provides a comprehensive framework for engineering context, dramatically improving AI's ability to understand and execute complex coding tasks.

🔗 **[View the original repo on GitHub](https://github.com/coleam00/context-engineering-intro)**

## Key Features:

*   **Comprehensive Context:**  Provide AI assistants with everything they need to succeed, including documentation, examples, rules, and validation, going far beyond simple prompt engineering.
*   **Reduced AI Failures:**  Minimize errors by addressing context failures, which are the leading cause of AI agent issues.
*   **Consistency & Standards:** Ensure your AI follows your project's patterns, conventions, and coding standards.
*   **Complex Feature Implementation:**  Empower AI to handle multi-step implementations and intricate features with ease.
*   **Self-Correcting Workflows:**  Utilize built-in validation loops, allowing your AI assistant to identify and correct its own mistakes.

## Getting Started: Quick Steps

Follow these steps to begin leveraging the power of context engineering:

1.  **Clone the Template:**
    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```

2.  **Set Up Project Rules (Optional, but Recommended):**
    *   Edit `CLAUDE.md` to define project-specific guidelines (e.g., coding style, file structure, testing requirements). The template provides a solid starting point.

3.  **Add Code Examples (Essential for Success):**
    *   Populate the `examples/` folder with relevant code patterns to guide the AI assistant.

4.  **Create Your Initial Feature Request:**
    *   Edit `INITIAL.md` to clearly define the feature you want to build, outlining requirements, dependencies, and desired functionality.

5.  **Generate a Comprehensive PRP (Product Requirements Prompt):**
    *   Utilize the `/generate-prp` command in Claude Code:
        ```bash
        /generate-prp INITIAL.md
        ```

6.  **Execute the PRP to Implement Your Feature:**
    *   Execute the generated PRP using the `/execute-prp` command in Claude Code:
        ```bash
        /execute-prp PRPs/your-feature-name.md
        ```

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#getting-started-quick-steps)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering is a revolutionary approach that shifts the focus from simple prompt engineering to a complete system for providing comprehensive context to AI assistants.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on crafting precise wording and phrasing.
*   Limited by the phrasing of your task.
*   Effectively, a sticky note for the AI.

**Context Engineering:**

*   Provides a complete context system.
*   Includes documentation, code examples, rules, and validation processes.
*   Enables complex features and self-correction.
*   Equivalent to giving the AI a complete screenplay with all the details.

### Why Context Engineering Matters

1.  **Reduces AI Failures:**  Most AI agent errors stem from context failures.
2.  **Ensures Consistency:**  Guarantees your AI adheres to your project patterns and conventions.
3.  **Enables Complex Features:** Allows the AI to handle intricate multi-step implementations with proper context.
4.  **Self-Correcting:** Validation loops enable the AI to fix its mistakes, delivering working code.

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

*   **Note:**  This template doesn't focus on RAG and tools with context engineering because I have a LOT more in store for that soon. ;)

## Writing Effective INITIAL.md Files

The `INITIAL.md` file is crucial for defining the desired functionality. Here's a breakdown:

### Key Sections Explained

**FEATURE:**  Be specific, clear, and detailed.

*   **Bad:** "Build a web scraper"
*   **Good:** "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in a PostgreSQL database."

**EXAMPLES:**  Leverage the `examples/` folder.

*   Reference specific code files and patterns within the `examples/` directory.
*   Clearly indicate what aspects of the example the AI should follow.

**DOCUMENTATION:**  Include relevant resources.

*   Include URLs to API documentation.
*   Provide links to library guides and references.
*   Include documentation for your project or the MCP server.

**OTHER CONSIDERATIONS:** Capture important details.

*   Mention authentication requirements.
*   Detail rate limits or quotas.
*   Highlight common pitfalls or edge cases.
*   Specify performance requirements.

## The PRP Workflow

PRPs (Product Requirements Prompts) are detailed implementation blueprints.  Here’s how the commands work:

### How `/generate-prp` Works

1.  **Research Phase:**
    *   Analyzes your codebase for existing patterns.
    *   Searches for similar implementations.
    *   Identifies project conventions.

2.  **Documentation Gathering:**
    *   Fetches relevant API documentation.
    *   Includes library documentation.
    *   Adds any relevant gotchas and quirks.

3.  **Blueprint Creation:**
    *   Creates a step-by-step implementation plan.
    *   Includes validation gates (e.g., tests).
    *   Adds testing requirements.

4.  **Quality Check:**
    *   Assigns a confidence level (1-10).
    *   Ensures all necessary context is included.

### How `/execute-prp` Works

1.  **Load Context:**  The AI reads the entire PRP.
2.  **Plan:** The AI creates a detailed task list.
3.  **Execute:** The AI implements each component.
4.  **Validate:** The AI runs tests and performs linting.
5.  **Iterate:** The AI fixes any identified issues.
6.  **Complete:**  The AI ensures all requirements are met.

## Using Examples Effectively

The `examples/` folder is **critical** for achieving the best results.  AI coding assistants excel when they can learn from practical patterns.

### What to Include in Examples

1.  **Code Structure Patterns:**
    *   Module organization examples
    *   Import conventions
    *   Class and function patterns

2.  **Testing Patterns:**
    *   Test file structure
    *   Mocking approaches
    *   Assertion styles

3.  **Integration Patterns:**
    *   API client implementations
    *   Database connection examples
    *   Authentication flows

4.  **CLI Patterns:**
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

### 1. Be Explicit in `INITIAL.md`

*   Don't assume the AI knows your preferences.
*   Include specific requirements and constraints.
*   Reference examples liberally.

### 2. Provide Comprehensive Examples

*   The more examples, the better the implementation.
*   Show what to do AND what *not* to do.
*   Include error handling patterns.

### 3. Utilize Validation Gates

*   PRPs incorporate test commands.
*   The AI iterates until all validations succeed.
*   This leads to functional code on the first attempt.

### 4. Leverage Documentation

*   Include official API documentation.
*   Add documentation resources.
*   Reference specific sections of relevant documentation.

### 5. Customize `CLAUDE.md`

*   Add your project's conventions.
*   Include project-specific rules and guidelines.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)