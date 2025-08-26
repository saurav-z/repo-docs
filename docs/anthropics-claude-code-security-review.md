# Enhance Your Code Security with AI-Powered Reviews Using Claude Code

**Automatically identify and remediate security vulnerabilities in your code with the Anthropic Claude Code Security Reviewer, a GitHub Action that leverages AI to provide in-depth code analysis.**  [See the original repository](https://github.com/anthropics/claude-code-security-review).

## Key Features

*   **AI-Powered Analysis:** Utilizes Claude's advanced reasoning to identify security vulnerabilities with deep semantic understanding.
*   **Diff-Aware Scanning:** Focuses analysis on changed files within pull requests, optimizing efficiency.
*   **Automated PR Comments:**  Provides inline comments directly within pull requests, highlighting security findings.
*   **Contextual Understanding:** Goes beyond basic pattern matching to understand code semantics and potential security implications.
*   **Language Agnostic:** Works with a wide range of programming languages.
*   **Advanced False Positive Filtering:** Reduces noise and focuses on critical vulnerabilities.

## How it Works

1.  **PR Analysis:** When a pull request is opened, Claude analyzes the diff.
2.  **Contextual Review:** Claude examines code changes, understanding their purpose and potential security risks.
3.  **Finding Generation:** Security issues are identified with detailed explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Advanced filtering removes less impactful or false-positive prone findings.
5.  **PR Comments:** Findings are posted as review comments on specific lines of code.

## Quick Start: Integrating into Your GitHub Workflow

Integrate the Claude Code Security Reviewer into your `.github/workflows/security.yml` file:

```yaml
name: Security Review

permissions:
  pull-requests: write  # Needed for leaving PR comments
  contents: read

on:
  pull_request:

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
          fetch-depth: 2
      
      - uses: anthropics/claude-code-security-review@main
        with:
          comment-pr: true
          claude-api-key: ${{ secrets.CLAUDE_API_KEY }}
```

## Configuration Options

Customize the behavior of the security scanner using these options:

*   `claude-api-key`: Your Anthropic Claude API key. (Required)
*   `comment-pr`: Whether to comment on pull requests with findings (default: `true`).
*   `upload-results`: Whether to upload results as artifacts (default: `true`).
*   `exclude-directories`: A comma-separated list of directories to exclude from scanning.
*   `claude-model`: Specifies the Claude model to use (default: `claude-opus-4-1-20250805`).
*   `claudecode-timeout`: Sets the timeout for ClaudeCode analysis in minutes (default: `20`).
*   `run-every-commit`: Runs ClaudeCode on every commit (may increase false positives).
*   `false-positive-filtering-instructions`: Path to a custom false-positive filtering instructions file.
*   `custom-security-scan-instructions`: Path to a custom security scan instructions file to append to audit prompt.

### Action Outputs

*   `findings-count`:  The total number of security findings.
*   `results-file`: The path to the results JSON file.

## Security Analysis Capabilities

The tool detects a wide range of vulnerabilities:

*   **Injection Attacks:** SQL, command, LDAP, XPath, NoSQL, XXE
*   **Authentication & Authorization:** Broken authentication, privilege escalation, etc.
*   **Data Exposure:** Hardcoded secrets, sensitive data logging, information disclosure.
*   **Cryptographic Issues:** Weak algorithms, improper key management.
*   **Input Validation:** Missing validation, improper sanitization, buffer overflows.
*   **Business Logic Flaws:** Race conditions, TOCTOU issues.
*   **Configuration Security:** Insecure defaults, missing security headers.
*   **Supply Chain:** Vulnerable dependencies, typosquatting risks.
*   **Code Execution:** RCE via deserialization, pickle injection, eval injection.
*   **Cross-Site Scripting (XSS):** Reflected, stored, and DOM-based XSS

### False Positive Filtering

The tool automatically filters out common, lower-impact findings such as:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

Customize filtering to align with your project's security goals.

### Benefits Over Traditional SAST

*   **Contextual Understanding:** Analyzes code semantics and intent, not just patterns.
*   **Lower False Positives:**  AI-powered analysis reduces noise.
*   **Detailed Explanations:** Provides clear explanations and remediation guidance.
*   **Adaptive Learning:**  Customizable for your organization's specific needs.

## Claude Code Integration: /security-review Command 

Use the `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) within Claude Code to initiate security analysis.  Customize the command by copying and editing the `security-review.md` file located in `.claude/commands/` in your project.

## Custom Scanning Configuration

For more details on custom scanning and false positive filtering instructions, see the [`docs/`](docs/) folder.

## Testing

Ensure functionality by running the test suite:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.