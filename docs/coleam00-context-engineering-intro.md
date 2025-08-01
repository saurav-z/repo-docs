# Supercharge Your AI Coding with Context Engineering

**Context Engineering empowers AI assistants to produce high-quality code by providing comprehensive context, leading to more reliable and efficient development.** ([Original Repository](https://github.com/coleam00/context-engineering-intro))

## Key Features:

*   **Comprehensive Context:** Provide your AI assistant with all the information needed to get the job done, including documentation, examples, rules, and validation.
*   **Reduced AI Failures:** Minimize agent errors by addressing context gaps.
*   **Consistency & Standardization:** Ensure your AI assistant adheres to project patterns and conventions.
*   **Enable Complex Features:** Handle multi-step implementations with ease.
*   **Self-Correcting:** Leverage validation loops to fix and refine code automatically.

## Getting Started: A Step-by-Step Guide

### 1. Clone the Template
```bash
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro
```

### 2. Set Up Global Rules (CLAUDE.md) (Optional)

Customize `CLAUDE.md` to define project-specific guidelines:

*   **Project Awareness:** Reads planning docs and checks tasks.
*   **Code Structure:** Enforces file size limits and module organization.
*   **Testing Requirements:** Implements unit test patterns and coverage expectations.
*   **Style Conventions:** Defines language preferences and formatting rules.
*   **Documentation Standards:** Sets docstring formats and commenting practices.

### 3. Add Code Examples (examples/) (Highly Recommended)

Place code examples in the `/examples` directory to showcase code structure, testing, and integration patterns.

### 4. Create Your Feature Request (INITIAL.md)

Edit `INITIAL.md` to detail your feature requirements. See `INITIAL_EXAMPLE.md` for inspiration. Include sections for:

*   **FEATURE**: Describe the desired functionality precisely.
*   **EXAMPLES**: Reference relevant code examples.
*   **DOCUMENTATION**: Link to essential documentation.
*   **OTHER CONSIDERATIONS**: Specify any special requirements.

### 5. Generate a Product Requirements Prompt (PRP)
Run in Claude Code:
```bash
/generate-prp INITIAL.md
```

This command generates a comprehensive PRP in `PRPs/your-feature-name.md`, using your feature request and codebase information.

### 6. Execute the PRP

Run in Claude Code:
```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will execute the PRP, implementing the feature, running tests, and ensuring all success criteria are met.

## Template Structure

```
context-engineering-intro/
├── .claude/
│   ├── commands/          # Custom commands
│   │   ├── generate-prp.md # Generates PRPs
│   │   └── execute-prp.md  # Executes PRPs
│   └── settings.local.json # Claude Code permissions
├── PRPs/                # Generated Product Requirement Prompts
│   ├── templates/
│   │   └── prp_base.md    # PRP template
│   └── EXAMPLE_multi_agent_prp.md  # Example PRP
├── examples/              # Code examples (critical!)
├── CLAUDE.md             # Global rules for AI assistant
├── INITIAL.md           # Feature request template
├── INITIAL_EXAMPLE.md   # Example feature request
└── README.md            # This file
```

## Deep Dive: Key Concepts

### What is Context Engineering?

Context Engineering is a superior approach to prompt engineering, providing a complete system for the AI assistant.

### The Power of Examples (examples/)

The `/examples` folder is crucial for guiding the AI, including:

*   Code Structure Patterns
*   Testing Patterns
*   Integration Patterns
*   CLI Patterns

### Writing Effective INITIAL.md Files

Provide comprehensive information in your feature requests.

*   **FEATURE:** Be specific and detail the desired functionality.
*   **EXAMPLES:** Refer to examples in the `examples/` directory.
*   **DOCUMENTATION:** Include relevant documentation links.
*   **OTHER CONSIDERATIONS:** Capture any specific requirements or constraints.

### The PRP Workflow

`/generate-prp` command analyzes your codebase, gathers documentation, and creates a comprehensive implementation blueprint.

`/execute-prp` reads the PRP, plans, implements, validates, iterates, and completes the feature.

### Best Practices

1.  Be explicit in `INITIAL.md`.
2.  Provide comprehensive code examples.
3.  Utilize validation gates.
4.  Leverage documentation.
5.  Customize `CLAUDE.md`.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)