# Context Engineering Template: Revolutionizing AI Code Assistants

**Unlock the power of AI coding assistants with Context Engineering, a superior approach to prompt engineering, enabling them to build complex features end-to-end.** ([See original repo](https://github.com/coleam00/context-engineering-intro))

## Key Features:

*   **Comprehensive Context:** Provides AI with the information it needs to get the job done, including documentation, examples, rules, and validation.
*   **Reduced AI Failures:** Addresses context failures, the primary cause of AI agent errors.
*   **Consistency & Standardization:** Ensures your project's patterns and conventions are followed.
*   **Enables Complex Feature Development:** Facilitates multi-step implementations with the right context.
*   **Self-Correcting Capabilities:** Leverages validation loops for AI to fix its mistakes.

## Getting Started: Quick Guide

1.  **Clone the Template:** `git clone https://github.com/coleam00/Context-Engineering-Intro.git`
2.  **Navigate:** `cd Context-Engineering-Intro`
3.  **Set up Project Rules (Optional):** Customize `CLAUDE.md` for your project guidelines.
4.  **Add Examples (Crucial):** Place relevant code examples in the `examples/` folder.
5.  **Create Feature Request:**  Edit `INITIAL.md` with your feature requirements.
6.  **Generate PRP:** Run `/generate-prp INITIAL.md` in Claude Code.
7.  **Execute PRP:** Run `/execute-prp PRPs/your-feature-name.md` in Claude Code.

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering is a revolutionary approach to building AI-powered software that prioritizes context.

### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering:** Relies on specific wording, limiting how you can phrase a task.
*   **Context Engineering:** Provides a complete system with documentation, examples, rules, patterns, and validation.

### Why Context Engineering Matters

1.  **Reduces Failures:** Addresses context failures.
2.  **Ensures Consistency:** Maintains project patterns and conventions.
3.  **Enables Complexity:** Facilitates multi-step implementations.
4.  **Self-Correcting:** Employs validation loops.

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

### 1. Set Up Global Rules (CLAUDE.md)

`CLAUDE.md` defines project-wide rules:

*   Project Awareness
*   Code Structure
*   Testing Requirements
*   Style Conventions
*   Documentation Standards

Customize it for your project.

### 2. Create Your Feature Request (INITIAL.md)

Describe your feature:

```markdown
## FEATURE:
[Detailed description of functionality]

## EXAMPLES:
[List example files and explain usage]

## DOCUMENTATION:
[Links to relevant resources]

## OTHER CONSIDERATIONS:
[Specific requirements, gotchas]
```

See `INITIAL_EXAMPLE.md` for a complete example.

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints:

*   Complete context
*   Implementation steps with validation
*   Error handling
*   Test requirements

Run in Claude Code:
```bash
/generate-prp INITIAL.md
```

This command will:
1.  Read your feature request
2.  Research codebase
3.  Search for relevant documentation
4.  Create a PRP in `PRPs/your-feature-name.md`

### 4. Execute the PRP

Implement your feature by running:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI coding assistant will:
1.  Read the PRP
2.  Create a plan
3.  Execute each step with validation
4.  Run tests and fix issues
5.  Ensure success criteria are met

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific and comprehensive.

**EXAMPLES**: Utilize the `examples/` folder. Reference files and patterns.

**DOCUMENTATION**: Include all relevant resources.

**OTHER CONSIDERATIONS**: Capture important details.

## The PRP Workflow

### How `/generate-prp` Works

1.  **Research:** Analyze codebase, search for implementations.
2.  **Documentation:** Fetch relevant API docs.
3.  **Blueprint:** Create step-by-step implementation plan with validation and tests.
4.  **Quality Check:**  Assess confidence level.

### How `/execute-prp` Works

1.  Load context
2.  Plan
3.  Execute
4.  Validate
5.  Iterate
6.  Complete

See `PRPs/EXAMPLE_multi_agent_prp.md` for a PRP example.

## Using Examples Effectively

The `examples/` folder is **essential** for success.

### What to Include in Examples

1.  Code Structure
2.  Testing Patterns
3.  Integration Patterns
4.  CLI Patterns

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