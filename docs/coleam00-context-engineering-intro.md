# Context Engineering Template: Revolutionize AI Coding with Comprehensive Context

**Tired of underwhelming AI coding results? This template empowers you to build robust AI-driven features by providing comprehensive context, making AI coding assistants 10x more effective than prompt engineering.**

[Original Repository](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Comprehensive Context:**  Provide all necessary information, including documentation, examples, rules, and validation, for superior AI coding results.
*   **PRP (Product Requirements Prompt) Workflow:** Automate the creation of detailed implementation blueprints for your AI coding assistant.
*   **Example-Driven Development:**  Leverage code examples to guide the AI, ensuring consistency and adherence to your project's style.
*   **Validation and Self-Correction:**  Incorporate tests and validation gates within your PRPs, enabling the AI to identify and resolve issues automatically.
*   **Customizable Rules:**  Define project-specific rules and coding standards to ensure consistent and high-quality code generation.

## Getting Started

This template streamlines the process of context engineering.

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Customize project rules (optional)
# Edit CLAUDE.md to set your project-specific guidelines

# 3. Add code examples (highly recommended)
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

Context Engineering represents a paradigm shift from traditional prompt engineering, offering a complete system to provide comprehensive context:

*   **Prompt Engineering:** Limited phrasing, like a sticky note.
*   **Context Engineering:** Comprehensive context including documentation, examples, and validation, like a full screenplay.

#### Why Context Engineering Matters:

1.  **Reduced AI Failures**: Addresses context-related failures.
2.  **Ensures Consistency**: AI adheres to your project's conventions.
3.  **Enables Complex Features**: AI handles multi-step implementations with context.
4.  **Self-Correcting**: Validation loops allow AI to fix its own mistakes.

### Step-by-Step Guide

1.  **Set Up Global Rules (`CLAUDE.md`):** Define project-wide rules for your AI assistant. The template includes:
    *   Project awareness, code structure, testing, style, and documentation standards.

2.  **Create Your Feature Request (`INITIAL.md`):** Describe what you want to build. Key sections: `FEATURE`, `EXAMPLES`, `DOCUMENTATION`, and `OTHER CONSIDERATIONS`.

3.  **Generate the PRP:**

    *   The `/generate-prp` command creates comprehensive implementation blueprints (PRPs).
    *   It reads your feature request, researches the codebase, searches documentation, and generates a PRP in the `PRPs/` directory.

    ```bash
    /generate-prp INITIAL.md
    ```

4.  **Execute the PRP:**

    *   The `/execute-prp` command executes the generated PRP.
    *   The AI reads context, creates an implementation plan, executes, validates, and iterates until all requirements are met.

    ```bash
    /execute-prp PRPs/your-feature-name.md
    ```

### Writing Effective INITIAL.md Files

*   **FEATURE:** Be specific and detailed about desired functionality.
*   **EXAMPLES:** Reference code examples in the `examples/` folder.
*   **DOCUMENTATION:** Include all relevant documentation and resources.
*   **OTHER CONSIDERATIONS:** Capture important details like authentication, limits, and performance needs.

### The PRP Workflow

*   **Research Phase:** Analyzes your codebase for patterns, identifies conventions.
*   **Documentation Gathering:** Fetches relevant documentation.
*   **Blueprint Creation:** Creates a step-by-step implementation plan, includes validation, and adds test requirements.
*   **Quality Check:** Scores confidence level and ensures all context is included.

### Using Examples Effectively

The `examples/` folder is **critical** for success.

*   **What to Include:**
    *   Code Structure Patterns
    *   Testing Patterns
    *   Integration Patterns
    *   CLI Patterns

*   **Example Structure:**
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

1.  **Be Explicit in `INITIAL.md`**:  Include specific requirements and reference examples.
2.  **Provide Comprehensive Examples**: Show both what to do and what not to do.
3.  **Use Validation Gates**: Ensure working code with test-driven development.
4.  **Leverage Documentation**: Include official API docs and server resources.
5.  **Customize `CLAUDE.md`**:  Add project conventions and coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)