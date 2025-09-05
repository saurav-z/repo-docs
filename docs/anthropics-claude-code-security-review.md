# Enhance Your Code Security with AI-Powered Reviews Using Claude Code

**Automatically identify and address security vulnerabilities in your code with the Claude Code Security Reviewer, a powerful GitHub Action powered by Anthropic's Claude.** ([Original Repository](https://github.com/anthropics/claude-code-security-review))

## Key Features:

*   **AI-Powered Security Analysis:** Leverages Anthropic's Claude Code for in-depth, semantic-aware vulnerability detection.
*   **Differential Scanning:** Analyzes only the changed files within a pull request, streamlining the review process.
*   **Automated PR Comments:** Posts findings directly in pull requests, pinpointing specific lines of code with detailed explanations and remediation guidance.
*   **Contextual Understanding:** Goes beyond superficial pattern matching, grasping the code's intent and implications.
*   **Language Agnostic:** Works seamlessly with any programming language.
*   **Reduced Noise:** Advanced filtering minimizes false positives, focusing on critical vulnerabilities.

## How It Works: A Step-by-Step Breakdown

1.  **PR Trigger:** Initiated when a pull request is created or updated.
2.  **Code Analysis:** The action examines the diff of the pull request to identify the altered code.
3.  **Contextual Review:** Claude Code analyzes the changes in context, understanding the code's purpose and potential security issues.
4.  **Vulnerability Identification:** Security vulnerabilities are identified, complete with detailed explanations, severity assessments, and remediation suggestions.
5.  **False Positive Filtering:** Advanced filtering logic removes low-impact or false positives to minimize noise.
6.  **PR Commenting:** Identified vulnerabilities are reported as comments on the specific lines of code.

## Installation and Configuration

Easily integrate the Claude Code Security Reviewer into your workflow using the following steps:

1.  **Add to your repository's `.github/workflows/security.yml`:**

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

2.  **Configure Action Inputs (see table below).**

## Configuration Options:

### Action Inputs:

| Input                     | Description                                                                                                                               | Default                        | Required |
| :------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------- | :------- |
| `claude-api-key`          | Your Anthropic Claude API key for security analysis. *Note*: Ensure this key is enabled for both the Claude API and Claude Code usage. | None                           | Yes      |
| `comment-pr`              | Whether to comment on pull requests with findings.                                                                                        | `true`                         | No       |
| `upload-results`          | Whether to upload results as artifacts.                                                                                                   | `true`                         | No       |
| `exclude-directories`     | Comma-separated list of directories to exclude from scanning.                                                                             | None                           | No       |
| `claude-model`            | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use.                                     | `claude-opus-4-1-20250805` | No       |
| `claudecode-timeout`      | Timeout for ClaudeCode analysis in minutes.                                                                                                | `20`                           | No       |
| `run-every-commit`        | Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits.                        | `false`                        | No       |
| `false-positive-filtering-instructions`   | Path to custom false positive filtering instructions text file.   | None                           | No       |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt. | None | No |

### Action Outputs:

| Output           | Description                          |
| :--------------- | :----------------------------------- |
| `findings-count` | Total number of security findings.   |
| `results-file`   | Path to the results JSON file.     |

## Comprehensive Security Analysis:

The Claude Code Security Reviewer identifies a broad spectrum of vulnerabilities, including:

*   Injection Attacks (SQL, command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization Flaws (broken auth, privilege escalation, insecure direct object references, bypass logic, session flaws)
*   Data Exposure (hardcoded secrets, sensitive data logging, information disclosure, PII handling violations)
*   Cryptographic Issues (weak algorithms, improper key management, insecure random number generation)
*   Input Validation Vulnerabilities (missing validation, improper sanitization, buffer overflows)
*   Business Logic Flaws (race conditions, TOCTOU issues)
*   Configuration Security (insecure defaults, missing security headers, permissive CORS)
*   Supply Chain Risks (vulnerable dependencies, typosquatting)
*   Code Execution Vulnerabilities (RCE via deserialization, pickle injection, eval injection)
*   Cross-Site Scripting (XSS) (reflected, stored, and DOM-based)

### Advanced False Positive Filtering:

The tool automatically excludes findings for the following vulnerabilities to focus on high-impact vulnerabilities:
*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

This filtering can be customized to suit your project's security needs.

### Customize with the `/security-review` Command:

Claude Code includes a `/security-review` command for direct security analysis in your development environment.  Customize the analysis by copying `.claude/commands/security-review.md` from the repository and editing the false positive filtering instructions.

### Custom Scanning Configuration:

Configure custom scanning and false positive filtering instructions; find details in the `/docs/` folder.

## Testing:

Validate the functionality by running the test suite:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support:

For any issues or questions, please:

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License:

MIT License - see the [LICENSE](LICENSE) file for details.