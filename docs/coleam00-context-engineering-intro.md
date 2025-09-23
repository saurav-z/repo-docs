# Context Engineering: Supercharge Your AI Coding with Comprehensive Context

**Stop wrestling with prompts and embrace Context Engineering – the revolutionary approach to AI coding that provides your AI assistant with all the information it needs to deliver high-quality, end-to-end solutions.** [Explore the original repo here](https://github.com/coleam00/context-engineering-intro).

## Key Features:

*   **Context-Driven AI:** Move beyond basic prompts and equip your AI with project-specific rules, documentation, and code examples.
*   **Simplified Workflow:** Streamlined process to generate and execute detailed Product Requirements Prompts (PRPs) directly within your AI coding assistant.
*   **Reduced Errors & Improved Consistency:** Minimize AI failures and ensure your projects adhere to consistent coding standards and patterns.
*   **Enhanced Complex Feature Implementation:** Empower your AI to handle multi-step implementations with thorough context and validation.
*   **Self-Correcting Capabilities:** Leverage built-in validation loops that enable the AI to identify and resolve its own mistakes.

## Getting Started

### 1. Clone the Template

```bash
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro
```

### 2. Set Up Project Rules (Optional)

*   Edit the `CLAUDE.md` file to customize your project's guidelines, including code style, testing, and documentation standards. This acts as a global instruction set for the AI assistant.

### 3. Add Examples (Recommended)

*   Place relevant code patterns in the `examples/` directory to guide the AI's implementation. This is *critical* for achieving the desired results.

### 4. Create Feature Requests

*   Describe your desired features in `INITIAL.md` to clearly communicate your project requirements and desired functionality.

### 5. Generate a PRP

*   Run the command below in Claude Code to generate a comprehensive Product Requirements Prompt:

```bash
/generate-prp INITIAL.md
```

### 6. Execute the PRP

*   Implement your feature by executing the generated PRP:

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

## What is Context Engineering?

Context Engineering is a superior approach to prompt engineering, providing an AI with all the necessary information for successful coding.

### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering:** Relies on clever phrasing and task-specific prompts, akin to a sticky note.
*   **Context Engineering:** Provides a complete system with documentation, examples, rules, patterns, and validation, like a detailed screenplay.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses context-related failures, the primary cause of AI agent issues.
2.  **Ensures Consistency:** Aligns AI output with project patterns and conventions.
3.  **Enables Complex Features:** Facilitates multi-step implementations.
4.  **Self-Correcting:** Utilizes validation loops to enable AI to rectify its own errors.

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

## Step-by-Step Guide

### 1. Define Global Rules (CLAUDE.md)

*   Customize the `CLAUDE.md` file with project-specific rules, including code style, testing requirements, and documentation practices.

### 2. Create Your Feature Request (INITIAL.md)

*   Describe your desired feature. Use the provided template as a guide:

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

### 3. Generate the PRP

*   Use the `/generate-prp` command to automatically create a comprehensive PRP that includes:
    *   Context and documentation
    *   Implementation steps with validation
    *   Error handling strategies
    *   Test requirements

### 4. Execute the PRP

*   Run the `/execute-prp` command to let the AI implement your feature using the generated PRP.

## Writing Effective INITIAL.md Files

### Key Sections Explained

*   **FEATURE:** Be specific and comprehensive, providing a clear description of the desired functionality and project requirements.
*   **EXAMPLES:** Reference relevant code patterns from the `examples/` folder to guide the AI's implementation.
*   **DOCUMENTATION:** Include links to all relevant resources, such as API documentation, library guides, and any other documentation.
*   **OTHER CONSIDERATIONS:** Capture details like authentication, rate limits, common pitfalls, and performance requirements.

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research Phase:** Analyzes your codebase to identify and understand coding patterns.
2.  **Documentation Gathering:** Integrates external resources, like API documentation.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan, with validation and test requirements.
4.  **Quality Check:** Includes a confidence level score to ensure all essential context is incorporated.

### How `/execute-prp` Works

1.  **Load Context:** Reads and parses the entire PRP.
2.  **Plan:** Decomposes the task into a detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Executes tests and applies linting.
5.  **Iterate:** Addresses any identified issues.
6.  **Complete:** Confirms that all requirements are met.

## Using Examples Effectively

The `examples/` directory is *essential* for maximizing the effectiveness of your AI coding assistant.

### What to Include in Examples

1.  **Code Structure Patterns**
2.  **Testing Patterns**
3.  **Integration Patterns**
4.  **CLI Patterns**

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
### 2. Provide Comprehensive Examples
### 3. Use Validation Gates
### 4. Leverage Documentation
### 5. Customize CLAUDE.md

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)