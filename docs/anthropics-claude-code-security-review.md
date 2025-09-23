# Enhance Your Code Security with AI-Powered Reviews Using Claude Code

**Automatically detect security vulnerabilities in your code with the Anthropic Claude Code Security Reviewer, a GitHub Action that leverages advanced AI for comprehensive analysis.** ([Original Repo](https://github.com/anthropics/claude-code-security-review))

## Key Features

*   **AI-Powered Analysis:** Employs Claude's sophisticated reasoning to identify vulnerabilities with deep semantic understanding.
*   **Diff-Aware Scanning:** Efficiently analyzes only the changed files within pull requests.
*   **Automated PR Comments:** Automatically posts security findings as comments directly on pull requests.
*   **Contextual Understanding:** Goes beyond pattern matching to grasp the meaning and implications of your code.
*   **Language Agnostic:** Works seamlessly with a wide range of programming languages.
*   **Advanced False Positive Filtering:** Minimizes noise, focusing on the most critical vulnerabilities.

## Getting Started: Integrate with GitHub Actions

Easily integrate the Claude Code Security Reviewer into your workflow by adding the following snippet to your `.github/workflows/security.yml` file:

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

Customize the behavior of the security review action using these inputs:

### Action Inputs

| Input                        | Description                                                                                                                               | Default                   | Required |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | -------- |
| `claude-api-key`             | Your Anthropic Claude API key. *Note*: This key requires both Claude API and Claude Code usage enabled.                                | None                      | Yes      |
| `comment-pr`                 | Whether to comment on PRs with findings.                                                                                                | `true`                    | No       |
| `upload-results`             | Whether to upload results as artifacts.                                                                                                  | `true`                    | No       |
| `exclude-directories`      | A comma-separated list of directories to exclude from the security scan.                                                            | None                      | No       |
| `claude-model`               | The Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use.                                 | `claude-opus-4-1-20250805` | No       |
| `claudecode-timeout`         | Timeout for the ClaudeCode analysis in minutes.                                                                                         | `20`                      | No       |
| `run-every-commit`           | Run ClaudeCode on every commit (skips cache check). *Warning*: May increase false positives on PRs with many commits.                         | `false`                   | No       |
| `false-positive-filtering-instructions` | Path to a custom file containing filtering instructions.                                                        | None                      | No       |
| `custom-security-scan-instructions` | Path to a custom security scan instructions text file to append to audit prompt.                                                        | None                      | No       |


### Action Outputs

| Output         | Description                                  |
| -------------- | -------------------------------------------- |
| `findings-count` | The total number of security findings.      |
| `results-file`   | The path to the generated results JSON file. |

## How It Works: A Deep Dive

1.  **PR Analysis:** The action begins by analyzing the pull request to understand what has been changed.
2.  **Contextual Review:** Claude examines the code changes within their context, considering their purpose and potential security implications.
3.  **Finding Generation:** Identified security issues are highlighted with detailed explanations, severity ratings, and remediation advice.
4.  **False Positive Filtering:** Advanced filtering mechanisms reduce noise by removing low-impact or false-positive-prone findings.
5.  **PR Comments:** Findings are presented as review comments, highlighting the specific lines of code in the pull request.

## Security Analysis Capabilities

### Vulnerabilities Detected

*   **Injection Attacks:** SQL, command, LDAP, XPath, NoSQL, XXE
*   **Authentication & Authorization:** Broken authentication, privilege escalation, insecure direct object references, session flaws
*   **Data Exposure:** Hardcoded secrets, sensitive data logging, information disclosure, PII violations
*   **Cryptographic Issues:** Weak algorithms, improper key management, insecure random number generation
*   **Input Validation:** Missing or improper sanitization, buffer overflows
*   **Business Logic Flaws:** Race conditions, TOCTOU issues
*   **Configuration Security:** Insecure defaults, missing security headers, permissive CORS
*   **Supply Chain:** Vulnerable dependencies, typosquatting risks
*   **Code Execution:** RCE via deserialization, pickle injection, eval injection
*   **Cross-Site Scripting (XSS):** Reflected, stored, and DOM-based XSS

### False Positive Filtering

The tool automatically excludes:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

Customization of the filtering is also available.

### Advantages Over Traditional SAST

*   **Contextual Understanding:** Goes beyond pattern matching to understand the *meaning* of the code.
*   **Reduced False Positives:** AI-powered analysis helps filter out irrelevant findings.
*   **Detailed Explanations:** Provides clear reasons for vulnerabilities and how to fix them.
*   **Customizable:** Adaptable to your organization's specific security needs.

## Advanced Usage

### Claude Code Integration: /security-review Command

Use the `/security-review` slash command in Claude Code to perform a comprehensive security review of all pending changes. Customize the command by editing the `.claude/commands/security-review.md` file in your project.

### Custom Scanning Configuration

Create custom scanning and false positive filtering instructions. See the [`docs/`](docs/) folder for detailed information.

## Testing

Verify the functionality by running the test suite:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support

For any issues or questions, you can:

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

This project is licensed under the [MIT License](LICENSE).