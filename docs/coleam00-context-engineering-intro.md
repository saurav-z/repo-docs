# Context Engineering: The Future of AI-Powered Coding (Improve & Summarize)

**Unlock the full potential of AI coding assistants with Context Engineering – a revolutionary approach that provides the complete context your AI needs to build complex features, consistently and reliably.**

> **Context Engineering surpasses prompt engineering and vibe coding by providing a comprehensive system for delivering relevant context to your AI coding assistant, resulting in more accurate and efficient code generation.**

[![GitHub Repo stars](https://img.shields.io/github/stars/coleam00/context-engineering-intro?style=social)](https://github.com/coleam00/context-engineering-intro)

## 🚀 Key Features

*   **10x Improvement Over Prompt Engineering:** Context Engineering focuses on providing comprehensive context, leading to higher-quality code.
*   **Reduce AI Failures:** Significantly decreases errors by providing all necessary information.
*   **Ensure Consistency:** Enforces project patterns, conventions, and coding standards.
*   **Enable Complex Features:** Facilitates multi-step implementations through detailed context.
*   **Self-Correcting:** Implements validation loops that allow the AI to fix its own mistakes.
*   **Comprehensive Template:** Includes a structured template with key components and best practices.
*   **Seamless Workflow:** Simplifies feature implementation through a streamlined PRP (Product Requirements Prompt) process.

## 🛠️ Getting Started

Follow these simple steps to get started with Context Engineering:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```

2.  **Customize Project Rules (Optional):**
    *   Edit `CLAUDE.md` to set project-specific guidelines, coding standards, and testing requirements.

3.  **Add Code Examples (Highly Recommended):**
    *   Place relevant code examples in the `examples/` folder.  These examples are crucial for guiding the AI.

4.  **Create a Feature Request:**
    *   Edit `INITIAL.md` to describe the feature you want to build.

5.  **Generate a PRP (Product Requirements Prompt):**
    *   In Claude Code, run: `/generate-prp INITIAL.md` to generate a comprehensive implementation blueprint.

6.  **Execute the PRP:**
    *   In Claude Code, run: `/execute-prp PRPs/your-feature-name.md` to implement the feature.

## 📚 Key Concepts and Structure

### What is Context Engineering?

Context Engineering goes beyond traditional prompt engineering by providing a complete system for the AI to understand the project's context.

**Key Differences:**

*   **Prompt Engineering:** Focuses on precise wording.
*   **Context Engineering:** Provides comprehensive context, including documentation, examples, rules, and validation.

**Benefits of Context Engineering:**

1.  **Reduced Failures:** AI receives all necessary information to minimize errors.
2.  **Consistency:** Ensures consistent adherence to project standards.
3.  **Complex Features:** Enables AI to handle intricate, multi-step implementations.
4.  **Self-Correction:** Implements validation and testing to allow the AI to learn from and correct its mistakes.

### Template Structure

```
context-engineering-intro/
├── .claude/          # Claude Code configuration
│   ├── commands/    # Custom commands
│   │   ├── generate-prp.md # Generates PRPs
│   │   └── execute-prp.md  # Executes PRPs
│   └── settings.local.json # Claude Code permissions
├── PRPs/             # Product Requirements Prompts (Implementation Blueprints)
│   ├── templates/
│   │   └── prp_base.md    # PRP base template
│   └── EXAMPLE_multi_agent_prp.md # Example PRP
├── examples/         # Code Examples (Essential!)
├── CLAUDE.md         # Global Rules for the AI Assistant
├── INITIAL.md        # Feature Request Template
├── INITIAL_EXAMPLE.md  # Example Feature Request
└── README.md         # This File
```

## 🛠️ Step-by-Step Guide

1.  **Set Up Global Rules (CLAUDE.md):**
    *   Customize `CLAUDE.md` with project-wide rules, including:
        *   Project awareness
        *   Code structure guidelines
        *   Testing and documentation requirements
        *   Style conventions

2.  **Create Your Initial Feature Request (INITIAL.md):**
    *   Describe the desired feature in detail:

    ```markdown
    ## FEATURE:
    [Specific description of the feature and its requirements]

    ## EXAMPLES:
    [References to code examples in the examples/ folder]

    ## DOCUMENTATION:
    [Links to relevant documentation, APIs, etc.]

    ## OTHER CONSIDERATIONS:
    [Important details like authentication, rate limits, or potential issues]
    ```

    *   Use `INITIAL_EXAMPLE.md` as a reference.

3.  **Generate the PRP (Product Requirements Prompt):**
    *   Run `/generate-prp INITIAL.md` in Claude Code.  This command generates a comprehensive implementation plan by:
        *   Analyzing your codebase for patterns
        *   Searching for relevant documentation
        *   Creating a step-by-step implementation plan with validation and test requirements.

4.  **Execute the PRP:**
    *   Run `/execute-prp PRPs/your-feature-name.md` in Claude Code.  This executes the PRP by:
        *   Reading the PRP context
        *   Creating a detailed implementation plan
        *   Executing each step with validation
        *   Running tests and fixing any issues
        *   Ensuring all success criteria are met.

## 📝 Writing Effective Feature Requests (INITIAL.md)

*   **FEATURE:** Be specific and include functionality and requirements.
    *   *Example:* Build an async web scraper that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL.
*   **EXAMPLES:** Point to specific files and code patterns in the `examples/` folder.
*   **DOCUMENTATION:** Include links to API documentation, library guides, and database schemas.
*   **OTHER CONSIDERATIONS:** Capture details like authentication, rate limits, and potential pitfalls.

## ⚙️ The PRP Workflow: `/generate-prp` and `/execute-prp`

### How /generate-prp Works:

1.  **Research Phase:** Analyzes codebase, searches for similar implementations, and identifies conventions.
2.  **Documentation Gathering:** Fetches API docs, library documentation, and adds relevant gotchas.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan with validation gates and test requirements.
4.  **Quality Check:** Scores confidence level and ensures all context is included.

### How /execute-prp Works:

1.  **Load Context:** Reads the PRP.
2.  **Plan:** Creates a detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes issues.
6.  **Complete:** Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a comprehensive example of a generated PRP.

## 💡 Maximizing Success: Using Examples Effectively

The `examples/` folder is a **critical** component. Provide patterns for the AI to follow.

### Example Content:

1.  **Code Structure Patterns:** Module organization, import conventions, class/function patterns.
2.  **Testing Patterns:** Test file structure, mocking approaches, and assertion styles.
3.  **Integration Patterns:** API client implementations, database connections, and authentication flows.
4.  **CLI Patterns:** Argument parsing, output formatting, and error handling.

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

## ✅ Best Practices

1.  **Be Explicit in INITIAL.md:** Include specific requirements and constraints.
2.  **Provide Comprehensive Examples:** More examples lead to better implementations, including error handling patterns.
3.  **Use Validation Gates:** PRPs include test commands that must pass. The AI will iterate until all tests succeed.
4.  **Leverage Documentation:** Include API docs, server resources, and relevant documentation sections.
5.  **Customize CLAUDE.md:** Add conventions, project-specific rules, and coding standards.

## 🔗 Resources

*   [Original Repository](https://github.com/coleam00/context-engineering-intro)
*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)