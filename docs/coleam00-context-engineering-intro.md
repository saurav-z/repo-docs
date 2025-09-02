# Context Engineering: Supercharge Your AI Coding with Comprehensive Context

**Tired of generic prompt engineering? Context Engineering provides a complete system for giving AI coding assistants the information they need to build complex features with unprecedented accuracy.** ([See the original repo](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **Boost AI Accuracy**: Significantly reduces AI failures by providing comprehensive context.
*   **Enforce Consistency**: Ensures your AI follows your project's coding patterns and conventions.
*   **Enable Complex Features**: Allows AI to handle multi-step implementations with precision.
*   **Self-Correcting Code**: Validation loops allow AI to find and fix its own mistakes.
*   **Comprehensive Workflow**: Generate and execute Product Requirement Prompts (PRPs) for streamlined feature development.

## Getting Started

### 1. Clone the Template

```bash
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro
```

### 2. Customize Project Rules (Optional)

*   Edit `CLAUDE.md` to add your project-specific guidelines for code structure, testing, style, and documentation.

### 3. Add Code Examples (Essential)

*   Place relevant code examples in the `examples/` folder.  Examples are *critical* for the AI to learn from.

### 4. Create Feature Requests

*   Edit `INITIAL.md` to define what you want to build, including specifications, requirements, examples, documentation, and any special considerations.

### 5. Generate a Product Requirements Prompt (PRP)

*   Run in Claude Code: `/generate-prp INITIAL.md`
*   This command leverages your `INITIAL.md` file to generate a comprehensive PRP in `PRPs/your-feature-name.md`.

### 6. Execute the PRP

*   Run in Claude Code: `/execute-prp PRPs/your-feature-name.md`
*   The AI assistant will then implement your feature based on the PRP's instructions.

## Core Concepts

### What is Context Engineering?

Context Engineering moves beyond simple prompt engineering to provide a complete system for instructing AI coding assistants. Unlike prompt engineering, which is like writing a sticky note, Context Engineering is like writing a complete screenplay. It encompasses:

*   **Comprehensive Context:** Documentation, examples, rules, patterns, and validation.
*   **Enhanced Reliability:** Addresses failures caused by lack of context.
*   **Improved Consistency:** Enforces project-specific patterns.
*   **Greater Capability:** Enables the creation of complex features.

### Template Structure

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

## Detailed Workflow

### Creating Feature Requests (INITIAL.md)

Use `INITIAL.md` to communicate your feature requests:

*   **FEATURE**: Specify *exactly* what you want to build.
*   **EXAMPLES**: Refer to relevant example files in the `examples/` directory, explaining how the AI should use them.
*   **DOCUMENTATION**: Include links to APIs, libraries, and documentation.
*   **OTHER CONSIDERATIONS**: Note any requirements, limitations, or common pitfalls.

### The PRP Workflow

1.  `/generate-prp` reads `INITIAL.md`, analyzes your codebase, and searches for relevant documentation.
2.  It then creates a detailed PRP in `PRPs/your-feature-name.md`.
3.  `/execute-prp` loads the PRP, generates a task list, implements each step, validates the implementation, runs tests, and iterates to fix issues.
4.  The AI coding assistant will implement each component, run tests, and ensure all requirements are met.

### Leveraging Code Examples

The `examples/` folder is a cornerstone of this approach:

*   **Code Structure Patterns:** Module organization, import conventions, class/function patterns.
*   **Testing Patterns:** Test file structure, mocking techniques, assertion styles.
*   **Integration Patterns:** API client implementations, database connections, authentication flows.
*   **CLI Patterns:** Argument parsing, output formatting, error handling.

## Best Practices

*   **Be Explicit in `INITIAL.md`**: Don't assume the AI knows. Be specific and provide requirements.
*   **Provide Comprehensive Examples**: More examples lead to better implementations. Include both what to do and what to avoid.
*   **Use Validation Gates**: PRPs include test commands, ensuring working code.
*   **Leverage Documentation**: Include official API docs and server resources.
*   **Customize `CLAUDE.md`**: Add project-specific rules and coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)