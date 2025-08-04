# Context Engineering Template: Revolutionize AI Coding with Comprehensive Context

**Tired of generic prompt engineering?  This template provides a complete system for Context Engineering, giving your AI coding assistants the information they need to excel, resulting in 10x better results.** 

[View the original repository on GitHub](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Context-Driven AI:** Provides the necessary context for AI coding assistants to get the job done effectively.
*   **Structured Workflow:** Streamlined approach for feature requests, product requirement generation, and execution.
*   **Comprehensive Templates:** Ready-to-use templates for project rules (CLAUDE.md), feature requests (INITIAL.md), and product requirement prompts (PRPs).
*   **Example-Driven Development:** Leverage code examples to guide AI assistants and ensure consistent coding patterns.
*   **Validation Gates:**  PRPs include built-in validation, ensuring the AI assistant produces working code.

## Getting Started

Follow these steps to quickly start with Context Engineering:

1.  **Clone the Template:**
    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```
2.  **Define Project Rules (Optional):**
    Edit `CLAUDE.md` to establish project-specific guidelines. A template is already provided.
3.  **Provide Code Examples (Highly Recommended):**
    Place relevant code examples in the `examples/` folder to demonstrate patterns.
4.  **Create a Feature Request:**
    Edit `INITIAL.md` with your specific feature requirements.
5.  **Generate a Product Requirements Prompt (PRP):**
    Run in Claude Code:
    ```bash
    /generate-prp INITIAL.md
    ```
6.  **Execute the PRP:**
    Run in Claude Code:
    ```bash
    /execute-prp PRPs/your-feature-name.md
    ```

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

## Core Concepts

### What is Context Engineering?

Context Engineering is a superior alternative to traditional prompt engineering, focused on providing comprehensive context to AI coding assistants. This approach reduces failures, ensures consistency, and enables complex feature implementation by providing detailed project guidelines, examples, and documentation.

### Context Engineering vs. Prompt Engineering

*   **Prompt Engineering:** Relies on clever phrasing and limited task descriptions.
*   **Context Engineering:** Provides a complete system with documentation, examples, rules, and validation, enabling more complex and reliable AI coding assistance.

### Why Context Engineering Matters

*   **Reduces AI Failures:** Addressing context failures rather than model failures.
*   **Ensures Consistency:** Enforces project patterns and conventions.
*   **Enables Complex Features:** Facilitates multi-step implementations.
*   **Self-Correcting:** Validation loops enable AI to fix its own mistakes.

## Detailed Workflow

### 1. Setting Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file is central to project-wide context. It allows you to define:

*   Project awareness
*   Code structure guidelines
*   Testing requirements
*   Style conventions
*   Documentation standards

### 2. Crafting Your Initial Feature Request (INITIAL.md)

The `INITIAL.md` file describes your feature requirements, including:

*   **FEATURE:** Clearly describe what you want to build (functionality and requirements).
*   **EXAMPLES:** Reference code examples in the `/examples` folder.
*   **DOCUMENTATION:** Include relevant API documentation, libraries, and resources.
*   **OTHER CONSIDERATIONS:** Address authentication, rate limits, and potential issues.

See `INITIAL_EXAMPLE.md` for an example.

### 3. Generating a Product Requirements Prompt (PRP)

PRPs are comprehensive blueprints similar to PRDs, used to guide the AI coding assistant. To generate a PRP:

1.  Run `/generate-prp INITIAL.md` in Claude Code.
2.  This command reads your feature request, researches your codebase, and creates a PRP in `PRPs/your-feature-name.md`.

### 4. Executing the PRP

Implement your feature by running:

1.  Run `/execute-prp PRPs/your-feature-name.md` in Claude Code.
2.  The AI assistant then reads the PRP, creates a plan, executes the steps, validates the implementation, runs tests, and iterates until all requirements are met.

## Effective Use of Examples

The `examples/` folder is vital for success, providing pattern guidance to AI assistants.

### Example Content

*   Code Structure Patterns
*   Testing Patterns
*   Integration Patterns
*   CLI Patterns

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

## Best Practices for Context Engineering

### 1. Be Explicit in INITIAL.md
*   Provide clear requirements and constraints.
*   Use specific references to examples.

### 2. Provide Comprehensive Examples
*   Include sufficient examples to show patterns.
*   Demonstrate both good and bad practices.

### 3. Use Validation Gates
*   PRPs incorporate tests to ensure working code.
*   The AI iterates until all validations are successful.

### 4. Leverage Documentation
*   Incorporate official API documentation.
*   Refer to specific documentation sections.

### 5. Customize CLAUDE.md
*   Add project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)