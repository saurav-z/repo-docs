# Context Engineering: Supercharge AI Coding Assistants for Superior Results

**Stop relying on prompt engineering; embrace Context Engineering to empower your AI coding assistant to build complete, complex features with accuracy and efficiency.** ([Original Repo](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **Comprehensive Context:** Provide AI assistants with everything they need - documentation, examples, and rules - for end-to-end feature implementation.
*   **Reduced AI Failures:** Minimize errors by ensuring AI understands your project's specific context.
*   **Consistency & Standardization:** Enforce coding conventions and project patterns across all implementations.
*   **Complex Feature Development:** Enable AI to handle multi-step tasks with validation and self-correction.
*   **Productivity Boost:** Generate comprehensive Product Requirements Prompts (PRPs) to streamline development.

## Getting Started

This template provides a framework for Context Engineering, allowing you to significantly improve the performance of your AI coding assistant.

### 1. Clone the Template

```bash
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro
```

### 2. Customize Project Rules (Optional)

Modify `CLAUDE.md` to define your project-specific guidelines, coding standards, and best practices.

### 3. Add Code Examples (Highly Recommended)

Place relevant code examples in the `examples/` folder to guide the AI's implementation. This is critical for success!

### 4. Create a Feature Request

Edit `INITIAL.md` to clearly articulate your feature requirements. See `INITIAL_EXAMPLE.md` for guidance.

### 5. Generate a Product Requirements Prompt (PRP)

Leverage the `/generate-prp` command (within Claude Code) to automatically create a comprehensive blueprint for your feature:

```bash
/generate-prp INITIAL.md
```

### 6. Execute the PRP

Use the `/execute-prp` command (within Claude Code) to have the AI coding assistant implement your feature based on the PRP:

```bash
/execute-prp PRPs/your-feature-name.md
```

## Template Structure

```
context-engineering-intro/
├── .claude/
│   ├── commands/        # Custom commands for Claude Code
│   │   ├── generate-prp.md   # Generates PRPs
│   │   └── execute-prp.md    # Executes PRPs
│   └── settings.local.json    # Claude Code permissions
├── PRPs/                # Generated Product Requirements Prompts
│   ├── templates/       # PRP templates
│   │   └── prp_base.md
│   └── EXAMPLE_multi_agent_prp.md  # Example PRP
├── examples/            # Code examples
├── CLAUDE.md            # Global project rules
├── INITIAL.md           # Feature request template
├── INITIAL_EXAMPLE.md   # Example feature request
└── README.md            # This file
```

## Core Concepts

### What is Context Engineering?

Context Engineering is a superior alternative to prompt engineering, providing AI assistants with comprehensive context to ensure successful implementations:

*   **Prompt Engineering:** Relies on clever wording and phrasing, limited in scope.
*   **Context Engineering:** A complete system including documentation, examples, rules, patterns, and validation, for a complete implementation.

### Why Context Engineering?

1.  **Reduce AI Failures:** Most failures stem from lack of context, not model limitations.
2.  **Ensure Consistency:** Standardize code patterns and conventions.
3.  **Enable Complex Features:** Facilitate multi-step implementations.
4.  **Self-Correcting:** Use validation loops to address and resolve errors.

## Step-by-Step Guide Explained

### 1. Setting Project-Wide Rules

*   **`CLAUDE.md`:** Defines project-wide standards for code structure, testing, and documentation.
*   Customize this template to fit your needs and project requirements.

### 2. Crafting Feature Requests

*   **`INITIAL.md`:** Describe your desired feature, including:
    *   **FEATURE:** Specific functionality and requirements.
    *   **EXAMPLES:** References to code patterns in the `examples/` folder.
    *   **DOCUMENTATION:** Links to relevant resources.
    *   **OTHER CONSIDERATIONS:** Important details, such as authentication, rate limits, and performance goals.

### 3. Generating PRPs

*   The `/generate-prp` command automates the creation of comprehensive implementation blueprints called Product Requirements Prompts (PRPs).
*   PRPs contain all necessary information: context, implementation steps with validation, error handling, and testing requirements.
*   The command automatically reads your feature request, researches the codebase for patterns, searches for documentation, and generates a PRP.

### 4. Executing PRPs

*   The `/execute-prp` command leverages the PRP to guide the AI coding assistant to:
    1.  Load context.
    2.  Plan the implementation using TodoWrite.
    3.  Execute each step, incorporating validation.
    4.  Run tests and fix any issues.
    5.  Ensure all requirements are met.

## Effective INITIAL.md Files

*   **FEATURE:** Be specific and comprehensive. Provide clear requirements.
*   **EXAMPLES:** Crucial for the AI. Link to relevant code examples in the `examples/` directory.
*   **DOCUMENTATION:** Include essential API documentation, library guides, and server resources.
*   **OTHER CONSIDERATIONS:** Capture details like authentication, rate limits, and potential pitfalls.

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research:** Analyzes the codebase and identifies patterns and conventions.
2.  **Documentation Gathering:** Fetches relevant API documentation and incorporates it into the PRP.
3.  **Blueprint Creation:** Generates a step-by-step implementation plan with validation gates and test requirements.
4.  **Quality Check:** Evaluates the PRP's confidence level and ensures all necessary context is included.

### How `/execute-prp` Works

1.  Loads the entire PRP.
2.  Creates a task list using TodoWrite.
3.  Implements each component.
4.  Validates the code using tests and linting.
5.  Iterates and resolves any issues.
6.  Completes when all requirements are satisfied.

## The Importance of Code Examples

The `examples/` folder is a vital component for Context Engineering success, as it provides the AI assistant with patterns to follow:

### Examples Should Include:

1.  **Code Structure Patterns:** Module organization, import conventions, class/function patterns.
2.  **Testing Patterns:** Test file structure, mocking approaches, assertion styles.
3.  **Integration Patterns:** API client implementations, database connections, authentication flows.
4.  **CLI Patterns:** Argument parsing, output formatting, error handling.

### Example Directory Structure

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

## Best Practices for Context Engineering

1.  **Specificity in INITIAL.md:** Provide clear requirements and constraints, and liberally reference code examples.
2.  **Comprehensive Examples:** The more examples, the better. Show both what *to do* and what *not to do*, and include error handling examples.
3.  **Validation Gates:** Use PRPs, which include test commands that must pass. The AI will iterate until all validations are successful.
4.  **Leverage Documentation:** Include official API documentation, server resources, and reference specific documentation sections.
5.  **Customize CLAUDE.md:** Define your coding standards and include project-specific rules within `CLAUDE.md`.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)