# Context Engineering: The Future of AI Coding

**Unlock the power of AI coding assistants by providing comprehensive context, leading to more accurate, consistent, and complex feature implementations.** ([View the original repo](https://github.com/coleam00/context-engineering-intro))

## Key Features:

*   **Context-Driven AI:** Move beyond basic prompt engineering and give your AI the full context it needs.
*   **Comprehensive PRPs (Product Requirements Prompts):** Generate detailed implementation blueprints for your features.
*   **Example-Driven Development:** Provide code examples to guide the AI assistant and ensure consistency.
*   **Validation & Self-Correction:** PRPs include testing, ensuring quality and allowing the AI to fix its own mistakes.
*   **Project-Specific Rules:** Customize a global rules file (`CLAUDE.md`) to align the AI with your team's conventions.

## Getting Started

Follow these simple steps to begin using the Context Engineering Template:

### 1. Clone and Set Up

```bash
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro
```

### 2. Define Project Guidelines (Optional)

Customize the `CLAUDE.md` file with your project's coding standards, testing requirements, and documentation preferences.

### 3. Provide Code Examples (Highly Recommended)

Place relevant code examples in the `examples/` directory to show the AI how to code in your preferred style.

### 4. Create a Feature Request

Describe the desired feature in `INITIAL.md`, including functionality, dependencies, and constraints.

### 5. Generate a PRP

Use the `/generate-prp` command to create a detailed Product Requirements Prompt based on your feature request.

```bash
/generate-prp INITIAL.md
```

### 6. Execute the PRP

Run the `/execute-prp` command to instruct the AI assistant to implement the feature.

```bash
/execute-prp PRPs/your-feature-name.md
```

## Comprehensive Guide

### What is Context Engineering?

Context Engineering provides AI coding assistants with comprehensive context, transforming how they execute your instructions.

#### Prompt Engineering vs. Context Engineering

*   **Prompt Engineering:** Focused on crafting the best wording and specific phrasing, which can be limiting.
*   **Context Engineering:** A full system of documentation, examples, rules, patterns, and validation, making it significantly more powerful.

#### Why Context Engineering Matters:

1.  **Reduced AI Failures:** Addresses AI failures by giving it the necessary context.
2.  **Ensured Consistency:** Aligns AI output with your project's patterns and conventions.
3.  **Enables Complex Features:** Allows AI to handle multi-step implementations.
4.  **Self-Correcting:** Validation loops allow AI to fix mistakes and create stable, working code.

### Template Structure

The template contains critical files:

```
context-engineering-intro/
├── .claude/
│   └── commands/
│   └── settings.local.json
├── PRPs/
│   ├── templates/
│   └── EXAMPLE_multi_agent_prp.md
├── examples/
├── CLAUDE.md
├── INITIAL.md
├── INITIAL_EXAMPLE.md
└── README.md
```

*   `.claude/`: Contains custom commands for PRP generation and execution.
*   `PRPs/`: Stores PRPs (Product Requirements Prompts) which contains implementation plans.
*   `examples/`: Contains code examples to guide the AI (critical!).
*   `CLAUDE.md`:  Defines global rules for the AI.
*   `INITIAL.md`:  Template for feature requests.

### Writing Effective INITIAL.md Files

Use `INITIAL.md` to clearly define your project's needs:

*   **FEATURE:** Specifically describe the desired functionality and requirements.
*   **EXAMPLES:** Refer to examples in the `examples/` folder.
*   **DOCUMENTATION:** Provide links to relevant documentation and resources.
*   **OTHER CONSIDERATIONS:** Include relevant details like authentication, rate limits, or common pitfalls.

### The PRP Workflow

1.  **Research Phase:** The `/generate-prp` command analyzes your code for patterns, searches for documentation, and identifies best practices.
2.  **Documentation Gathering:** Fetches relevant API docs, includes library documentation, and includes gotchas and quirks.
3.  **Blueprint Creation:** Develops a step-by-step implementation plan, incorporates validation gates, and adds test requirements.
4.  **Quality Check:** Scores the confidence level and ensures all necessary context is included.
5.  **Execution:** The `/execute-prp` command reads the PRP, creates an implementation plan, executes each step, validates the results, and iterates to correct any issues.

### Using Examples Effectively

The `examples/` folder is essential for successful context engineering.

*   **Include:** Code structure, testing, integration, and CLI patterns.
*   **Structure:**  A well-organized directory to group examples.

### Best Practices

1.  **Be Explicit in `INITIAL.md`:** Provide specific requirements, constraints, and include detailed context.
2.  **Provide Comprehensive Examples:** Offer more examples to improve implementations and include error handling.
3.  **Use Validation Gates:** PRPs include test commands that will iterate until all validations succeed.
4.  **Leverage Documentation:** Include all necessary API documents and reference specific sections.
5.  **Customize `CLAUDE.md`:**  Add project-specific rules and conventions to customize your project.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)