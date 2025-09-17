# AI-Powered Code Security Review with Claude 

**Protect your code and streamline your security workflow with the Claude Code Security Review GitHub Action, leveraging the power of AI to identify vulnerabilities in your pull requests.** ([Original Repository](https://github.com/anthropics/claude-code-security-review))

## Key Features

*   **Intelligent Analysis:** Uses Anthropic's Claude AI for deep semantic code analysis.
*   **Diff-Aware Scanning:** Focuses security reviews on changed code within pull requests.
*   **Automated PR Comments:**  Posts findings directly to your pull requests, facilitating quick review.
*   **Contextual Understanding:** Analyzes code's meaning and potential security risks.
*   **Language Agnostic:** Works seamlessly with various programming languages.
*   **Reduced Noise:** Employs advanced filtering to minimize false positives.

## Get Started: Quick Installation

Integrate the security review into your workflow by adding the following to your repository's `.github/workflows/security.yml`:

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

**Important:** Ensure you have a valid Anthropic Claude API key.  The key must be enabled for both the Claude API and Claude Code usage.

## Configuration Options

Customize the action's behavior using these inputs:

| Input                       | Description                                                                                                                                                                            | Default                        | Required |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------- | :------- |
| `claude-api-key`            | Your Anthropic Claude API key. _(Must be enabled for Claude API and Claude Code)_                                                                                                        | None                           | Yes      |
| `comment-pr`                | Whether to comment on PRs with findings.                                                                                                                                                 | `true`                         | No       |
| `upload-results`            | Whether to upload results as artifacts.                                                                                                                                                 | `true`                         | No       |
| `exclude-directories`       | Comma-separated list of directories to exclude from scanning.                                                                                                                            | None                           | No       |
| `claude-model`              | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use.                                                                              | `claude-opus-4-1-20250805` | No       |
| `claudecode-timeout`        | Timeout for ClaudeCode analysis in minutes.                                                                                                                                            | `20`                           | No       |
| `run-every-commit`          | Run ClaudeCode on every commit (skips cache check). *Warning: May increase false positives on PRs with many commits.*                                                                | `false`                        | No       |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file                                                                                                                        | None                           | No       |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt                                                                                                            | None                           | No       |

### Action Outputs

The following outputs are available after the action completes:

| Output          | Description                           |
| :-------------- | :------------------------------------ |
| `findings-count` | Total number of security findings. |
| `results-file`   | Path to the results JSON file.      |

## How It Works

The security review process consists of these steps:

1.  **PR Analysis:** On pull request creation, the system analyzes the diff to understand code changes.
2.  **Contextual Review:** Claude examines the changes, considering their purpose and security implications.
3.  **Finding Generation:** Identified security issues are reported with detailed explanations and severity levels.
4.  **False Positive Filtering:** Advanced filtering reduces noise and focuses on real vulnerabilities.
5.  **PR Comments:** Findings are posted directly as comments within the pull request.

## Security Analysis Capabilities

This tool can identify a wide range of security vulnerabilities, including:

### Types of Vulnerabilities Detected

*   Injection Attacks (SQL, command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization (Broken auth, privilege escalation, insecure direct object references, bypass logic, session flaws)
*   Data Exposure (Hardcoded secrets, sensitive data logging, information disclosure, PII handling violations)
*   Cryptographic Issues (Weak algorithms, improper key management, insecure random number generation)
*   Input Validation (Missing validation, improper sanitization, buffer overflows)
*   Business Logic Flaws (Race conditions, time-of-check-time-of-use (TOCTOU) issues)
*   Configuration Security (Insecure defaults, missing security headers, permissive CORS)
*   Supply Chain (Vulnerable dependencies, typosquatting risks)
*   Code Execution (RCE via deserialization, pickle injection, eval injection)
*   Cross-Site Scripting (XSS) (Reflected, stored, and DOM-based XSS)

### False Positive Filtering

The system automatically excludes a variety of low-impact and false-positive prone findings to increase accuracy.

## Benefits Over Traditional SAST

*   **Semantic Understanding:** Understands code meaning and intent, not just patterns.
*   **Reduced False Positives:** AI-powered analysis minimizes noise.
*   **Detailed Explanations:** Provides clear explanations of vulnerabilities and remediation steps.
*   **Customization:** Can be tailored to meet your project's security needs.

## Advanced Features

### Claude Code Integration: `/security-review` Command 

The action integrates with the `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) directly within the Claude Code development environment. This allows you to perform a comprehensive security review of all pending changes within the Claude Code environment. You can customize the command to add additional organization-specific directions to the false positive filtering instructions.

### Custom Scanning Configuration

For customized security scanning and false-positive filtering instructions, see the [documentation](docs/) folder.

## Testing

To run the test suite, navigate to the project directory and run:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   [Open an issue](https://github.com/anthropics/claude-code-security-review/issues) in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.