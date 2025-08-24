# Context Engineering: Revolutionize Your AI Coding with This Template

**Tired of generic prompt engineering? Dive into Context Engineering, a comprehensive system for providing AI assistants with the complete context they need for powerful, accurate, and consistent code generation.** [Explore the original repository here](https://github.com/coleam00/context-engineering-intro).

## Key Features:

*   **Context-Driven AI**: Move beyond prompts and empower AI with comprehensive context, including documentation, examples, and project-specific rules.
*   **Reduced AI Failures**: Minimize agent errors by providing all the necessary information upfront.
*   **Consistent Code**: Enforce project patterns, conventions, and documentation standards for reliable results.
*   **Complex Feature Enablement**:  Tackle multi-step implementations with ease, thanks to a robust context-aware system.
*   **Self-Correcting Capabilities**: Leverage validation loops to ensure the AI fixes its own mistakes and delivers high-quality code.

## Quick Start:

1.  **Clone the Template**:
    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```
2.  **Set Up Project Rules (Optional)**: Customize `CLAUDE.md` with your project's specific guidelines.
3.  **Add Code Examples (Highly Recommended)**: Place relevant code examples within the `examples/` folder.
4.  **Create Your Feature Request**:  Describe your desired features in `INITIAL.md`.
5.  **Generate a Product Requirements Prompt (PRP)**:
    ```bash
    /generate-prp INITIAL.md
    ```
6.  **Execute the PRP to Build Your Feature**:
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

Context Engineering is a paradigm shift from traditional prompt engineering, offering a far more robust and reliable approach to AI coding.

### Prompt Engineering vs. Context Engineering:

**Prompt Engineering:**

*   Focuses on precise wording and phrasing.
*   Limited by the way you frame the task.
*   Like giving someone a sticky note.

**Context Engineering:**

*   Provides a complete system for context.
*   Includes documentation, examples, rules, patterns, and validation.
*   Like writing a full screenplay with all the details.

### Why Context Engineering Matters:

1.  **Reduces AI Failures:** Addresses context failure, which are the root cause of most agent failures.
2.  **Ensures Consistency:** Guarantees the AI follows your project patterns and conventions.
3.  **Enables Complex Features:** Allows the AI to handle multi-step implementations with proper guidance.
4.  **Self-Correcting:** Utilizes validation loops to enable AI to fix mistakes.

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

1.  **Set Up Global Rules (CLAUDE.md)**:
    *   `CLAUDE.md` defines project-wide rules for the AI assistant.
    *   Customize it with project-specific settings like code structure, testing requirements, and documentation standards.
2.  **Create Your Initial Feature Request (INITIAL.md)**:
    *   Describe your desired feature in `INITIAL.md` using the template below:
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
    *   See `INITIAL_EXAMPLE.md` for guidance.
3.  **Generate the Product Requirements Prompt (PRP)**:
    *   PRPs are comprehensive implementation blueprints that include context, documentation, implementation steps, error handling, and testing requirements.
    *   Run the following command in Claude Code:
    ```bash
    /generate-prp INITIAL.md
    ```
    *   The command will analyze your code, gather documentation, and create a PRP.
4.  **Execute the PRP**:
    *   Implement your feature by running the PRP with this command:
    ```bash
    /execute-prp PRPs/your-feature-name.md
    ```
    *   The AI will read context, implement, validate, and iterate until all requirements are met.

## Writing Effective INITIAL.md Files:

*   **FEATURE**: Be specific and detail all functional and implementation requirements.
*   **EXAMPLES**: Reference example files from the `examples/` folder.
*   **DOCUMENTATION**: Include relevant documentation URLs.
*   **OTHER CONSIDERATIONS**:  Mention any authentication, rate limits, or other specific requirements.

## The PRP Workflow:

### How `/generate-prp` Works:

1.  **Research**: Analyze your codebase, search for patterns, and identify existing conventions.
2.  **Documentation Gathering**: Fetch relevant API documentation and include any relevant quirks.
3.  **Blueprint Creation**: Generate a step-by-step implementation plan with validation gates.
4.  **Quality Check**: Assign a confidence level and ensure comprehensive context inclusion.

### How `/execute-prp` Works:

1.  **Load Context**: Read the complete PRP document.
2.  **Plan**: Create a detailed task list.
3.  **Execute**: Implement each component.
4.  **Validate**: Run tests.
5.  **Iterate**: Fix issues.
6.  **Complete**: Ensure all requirements are met.

## Using Examples Effectively:

The `examples/` folder is crucial for successful AI code generation.

### What to Include:

1.  Code Structure Patterns.
2.  Testing Patterns.
3.  Integration Patterns.
4.  CLI Patterns.

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

1.  **Be Explicit in INITIAL.md**:  Include all requirements and reference examples.
2.  **Provide Comprehensive Examples**:  Show what to do and what to avoid.  Include error handling.
3.  **Use Validation Gates**: PRPs include test commands that must pass.
4.  **Leverage Documentation**: Include official API docs.
5.  **Customize CLAUDE.md**: Add your own conventions and project-specific rules.

## Resources:

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)