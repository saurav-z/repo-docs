# Context Engineering: Supercharge Your AI Coding Assistant 🚀

**Unlock 10x the coding efficiency and accuracy with Context Engineering, a revolutionary approach that provides your AI assistant with the complete context needed for successful, end-to-end code generation.** ([View the original repository](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **Comprehensive Context:** Provide your AI with all the information it needs to get the job done.
*   **Reduced AI Failures:** Minimize errors by addressing context failures, not just model limitations.
*   **Consistent Coding:** Enforce project patterns, conventions, and standards across all code.
*   **Complex Feature Enablement:** Empower your AI to build sophisticated, multi-step features with confidence.
*   **Self-Correcting Code:** Leverage validation loops for AI to autonomously fix its own mistakes.

## Getting Started

```bash
# 1. Clone the Template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set Up Your Project Rules (optional)
# Edit CLAUDE.md to define project-specific guidelines, conventions, and standards.

# 3. Add Code Examples (highly recommended)
# Place relevant code examples in the examples/ folder to guide the AI.

# 4. Create Your Initial Feature Request
# Edit INITIAL.md to define your requirements in detail.

# 5. Generate a Comprehensive PRP (Product Requirements Prompt)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP to Implement Your Feature
# In Claude Code, run:
/execute-prp PRPs/your-feature-name.md
```

## Core Concepts

### What is Context Engineering?

Context Engineering surpasses prompt engineering by providing AI assistants with a complete understanding of the project. It's like providing a full screenplay versus a sticky note.

### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering:** Relies on clever wording and specific phrasing; limited in scope.
*   **Context Engineering:** Provides comprehensive context including documentation, examples, rules, patterns, and validation; a complete system for success.

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

## Step-by-Step Guide

### 1. Define Global Rules (CLAUDE.md)

Customize `CLAUDE.md` to establish project-wide rules and conventions, covering:

*   Project awareness
*   Code structure
*   Testing requirements
*   Style conventions
*   Documentation standards

### 2. Create Your Feature Request (INITIAL.md)

Describe your desired feature in detail within `INITIAL.md`, including:

*   **FEATURE:** A specific and detailed description of what you want to build.
*   **EXAMPLES:** Reference code examples in the `examples/` folder to guide the AI.
*   **DOCUMENTATION:** Include links to relevant documentation and resources.
*   **OTHER CONSIDERATIONS:** Note any specific requirements, limitations, or gotchas.

**Example: See `INITIAL_EXAMPLE.md` for a complete example.**

### 3. Generate the PRP (Product Requirements Prompt)

Generate a comprehensive PRP using the command:

```bash
/generate-prp INITIAL.md
```

This command will:

1.  Analyze your codebase for patterns.
2.  Gather relevant documentation.
3.  Create a detailed PRP with a step-by-step implementation plan.

### 4. Execute the PRP

Implement your feature by running:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read all context from the PRP.
2.  Create a detailed implementation plan.
3.  Execute each step with validation.
4.  Run tests and fix any issues.
5.  Ensure all success criteria are met.

## Best Practices

### Writing Effective INITIAL.md

*   **Be Specific:** Detail functionality and requirements clearly.
*   **Leverage Examples:** Reference code examples to guide the AI's implementation.
*   **Include Documentation:** Link to all relevant API documentation and resources.
*   **Address Considerations:** Note authentication, rate limits, and potential pitfalls.

### The PRP Workflow

*   `/generate-prp`: Researches the codebase, gathers documentation, and creates a detailed implementation plan.
*   `/execute-prp`: Loads the PRP, creates a task list, implements each component, validates, iterates, and completes the feature.

### Maximizing the Power of Examples

*   **Code Structure Patterns:** Module organization, import conventions, etc.
*   **Testing Patterns:** Test file structure, mocking, and assertion styles.
*   **Integration Patterns:** API client implementations, database connections, etc.
*   **CLI Patterns:** Argument parsing, output formatting, and error handling.

### Additional Tips

*   **Be Explicit:** Define your preferences clearly in `INITIAL.md`.
*   **Provide Comprehensive Examples:** More examples lead to better implementations.
*   **Use Validation Gates:** PRPs include tests to ensure working code.
*   **Leverage Documentation:** Include API docs and relevant resources.
*   **Customize CLAUDE.md:** Tailor project-specific rules and coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)