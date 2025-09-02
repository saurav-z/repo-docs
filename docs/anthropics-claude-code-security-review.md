# Enhance Your Code Security with the Claude Code Security Reviewer

**Automatically identify and fix security vulnerabilities in your code with the power of AI.** This GitHub Action leverages Anthropic's Claude Code to provide intelligent, context-aware security analysis directly within your pull requests.  Get started today and secure your software! ([Original Repository](https://github.com/anthropics/claude-code-security-review))

## Key Features

*   **AI-Powered Analysis:** Leverages Claude's advanced reasoning for deep semantic understanding of your code.
*   **Diff-Aware Scanning:** Focuses on changed files in pull requests for faster, more relevant analysis.
*   **Automated PR Comments:** Posts security findings directly within your pull requests for easy review.
*   **Contextual Understanding:** Goes beyond pattern matching to understand code's purpose and potential vulnerabilities.
*   **Language Agnostic:** Works seamlessly with a wide variety of programming languages.
*   **Reduced Noise:** Advanced false positive filtering to focus on critical security issues.

## Getting Started

Integrate the Claude Code Security Reviewer into your GitHub Actions workflow by adding the following snippet to your `.github/workflows/security.yml` file:

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

**Important:** Ensure you have an Anthropic Claude API key and that it's enabled for both Claude API and Claude Code usage.

## Configuration Options

Customize the action's behavior using the following inputs:

### Action Inputs

| Input | Description | Default | Required |
|-------|-------------|---------|----------|
| `claude-api-key` | Anthropic Claude API key for security analysis. *Note*: This API key needs to be enabled for both the Claude API and Claude Code usage. | None | Yes |
| `comment-pr` | Whether to comment on PRs with findings | `true` | No |
| `upload-results` | Whether to upload results as artifacts | `true` | No |
| `exclude-directories` | Comma-separated list of directories to exclude from scanning | None | No |
| `claude-model` | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use. Defaults to Opus 4.1. | `claude-opus-4-1-20250805` | No |
| `claudecode-timeout` | Timeout for ClaudeCode analysis in minutes | `20` | No |
| `run-every-commit` | Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits. | `false` | No |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file | None | No |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt | None | No |

### Action Outputs

| Output | Description |
|--------|-------------|
| `findings-count` | Total number of security findings |
| `results-file` | Path to the results JSON file |

## How It Works

1.  **PR Analysis:** Analyzes the diff to understand code changes.
2.  **Contextual Review:** Examines changes, considering their purpose and security implications.
3.  **Finding Generation:** Identifies security issues with detailed explanations and remediation guidance.
4.  **False Positive Filtering:** Reduces noise to focus on critical vulnerabilities.
5.  **PR Comments:** Posts findings as comments on the relevant lines of code.

## Security Analysis Capabilities

This action detects a wide range of vulnerabilities, including:

*   Injection Attacks (SQL, Command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization Issues (Broken Auth, Privilege Escalation, Insecure Direct Object References, Session Flaws)
*   Data Exposure (Hardcoded Secrets, Sensitive Data Logging, Information Disclosure, PII Handling)
*   Cryptographic Issues (Weak Algorithms, Key Management, RNG)
*   Input Validation Problems (Missing Validation, Sanitization, Buffer Overflows)
*   Business Logic Flaws (Race Conditions, TOCTOU)
*   Configuration Security Issues (Insecure Defaults, Missing Headers, Permissive CORS)
*   Supply Chain Risks (Vulnerable Dependencies, Typosquatting)
*   Code Execution (RCE via Deserialization, Pickle Injection, Eval Injection)
*   Cross-Site Scripting (XSS)

### False Positive Filtering

The tool automatically filters out a number of low-impact findings to focus on the most important vulnerabilities:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

Customize the false positive filtering to match your project's security goals.

### Benefits Over Traditional SAST

*   **Contextual Understanding:** Analyzes code semantics and intent.
*   **Reduced False Positives:** AI-powered analysis minimizes noise.
*   **Detailed Explanations:** Provides clear explanations and remediation steps.
*   **Customizable:** Adaptable to your organization's security requirements.

## Integration with Claude Code: `/security-review` Command

The integrated `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) provides the same security review capabilities as the GitHub Action, but directly within your Claude Code environment. To use it, simply type `/security-review` to analyze all pending changes.

Customize the command by copying the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file to your project's `.claude/commands/` folder and edit it according to your needs.

## Custom Scanning Configuration

Explore custom scanning and false positive filtering instructions in the [`docs/`](docs/) folder for advanced customization.

## Testing

Validate the functionality by running the test suite:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support

For any issues or questions:

*   Open an issue in this repository.
*   Consult the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.