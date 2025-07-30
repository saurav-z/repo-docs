# Context Engineering Template: Revolutionize AI Coding with Comprehensive Context

**Tired of generic prompts? This template empowers you with Context Engineering, providing your AI coding assistants the complete context they need for robust, end-to-end feature implementation.**

[View the original repo on GitHub](https://github.com/coleam00/context-engineering-intro)

## Key Features:

*   **Comprehensive Context:** Provide your AI with documentation, examples, rules, and validation to significantly reduce failures and ensure consistent, high-quality code.
*   **Automated PRP Generation:**  Generate detailed Product Requirements Prompts (PRPs) with a simple command, streamlining the feature implementation process.
*   **Example-Driven Development:**  Leverage the `examples/` folder to provide code patterns and best practices, guiding your AI towards desired implementations.
*   **Validation and Iteration:**  PRPs include built-in validation gates, ensuring that your AI fixes issues and delivers working code.
*   **Customizable Rules:**  Define project-specific rules in `CLAUDE.md` to enforce coding standards, style conventions, and more.

## Getting Started

This template provides a structured workflow for context engineering. Follow these steps to get started:

### 1. Clone the Template

```bash
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro
```

### 2. Define Project Rules (Optional but Recommended)

Edit `CLAUDE.md` to establish your project's global rules, including:

*   Code structure guidelines
*   Testing requirements
*   Style conventions
*   Documentation standards

### 3. Provide Code Examples (Essential!)

Place relevant code examples in the `examples/` folder.  This is *critical* for guiding your AI.  A well-structured `examples/` directory helps your AI understand your coding style, testing practices, and integration patterns.

### 4. Create Feature Requests

Describe your desired features in `INITIAL.md`.  Be specific about functionality, requirements, and any relevant documentation.

### 5. Generate Product Requirements Prompt (PRP)

Use the following command within Claude Code:

```bash
/generate-prp INITIAL.md
```

This command will automatically generate a comprehensive PRP based on your feature request, codebase patterns, and relevant documentation.

### 6. Execute the PRP

Execute the generated PRP within Claude Code to implement the feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

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

## Deep Dive: Context Engineering Concepts

### What is Context Engineering?

Context Engineering is a superior approach to prompt engineering, which provides AI coding assistants with comprehensive context:

*   **Context Engineering**:  A complete system including documentation, examples, rules, patterns, and validation. This is like writing a full screenplay.
*   **Prompt Engineering**:  Focuses on clever wording and phrasing. This is like giving someone a sticky note.

#### Why Context Engineering Matters:

*   **Reduced AI Failures:**  Context failures, not model failures, are the leading cause of agent issues.
*   **Consistency:** AI adheres to your project's patterns and standards.
*   **Complex Features:**  AI can handle multi-step implementations effectively.
*   **Self-Correcting:**  Validation loops allow AI to fix its own mistakes.

### Effective Feature Requests (INITIAL.md)

*   **FEATURE:**  Be specific.  Instead of "Build a web scraper," try "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL."
*   **EXAMPLES:**  Reference and explain how examples in the `examples/` folder should be used.
*   **DOCUMENTATION:**  Include URLs to API documentation, library guides, and other resources.
*   **OTHER CONSIDERATIONS:**  Note authentication, rate limits, potential pitfalls, and performance requirements.

### The PRP Workflow

*   `/generate-prp`:
    1.  Analyzes codebase for patterns.
    2.  Gathers relevant documentation.
    3.  Creates a step-by-step implementation plan.
    4.  Adds validation checks.
*   `/execute-prp`:
    1.  Loads the PRP.
    2.  Creates a task list.
    3.  Implements components.
    4.  Validates code with tests and linting.
    5.  Iterates and fixes issues.
    6.  Ensures all requirements are met.

### Leveraging Examples (examples/)

The `examples/` directory is crucial for success! It should include code demonstrating:

*   Code structure patterns.
*   Testing patterns.
*   Integration patterns.
*   CLI patterns.

Example structure:

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

### Best Practices

1.  Be explicit in `INITIAL.md`.
2.  Provide comprehensive examples.
3.  Use validation gates in PRPs.
4.  Leverage official documentation.
5.  Customize `CLAUDE.md`.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)