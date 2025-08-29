# Enhance Your Code Security with AI-Powered Reviews | Claude Code Security Reviewer

**Stop vulnerabilities before they reach production!** The Claude Code Security Reviewer uses Anthropic's Claude AI to intelligently analyze your code changes and pinpoint potential security flaws. [Learn More at the Original Repo](https://github.com/anthropics/claude-code-security-review).

## Key Features:

*   **AI-Driven Analysis:** Leverages Claude's advanced reasoning for deep semantic understanding of code and vulnerability detection.
*   **Diff-Aware Scanning:** Efficiently analyzes only the changed files in pull requests, saving time and resources.
*   **Direct PR Comments:** Provides clear and concise security findings directly within your pull requests for easy review.
*   **Contextual Understanding:** Goes beyond simple pattern matching, comprehending code semantics and intent.
*   **Language Agnostic:** Compatible with a wide range of programming languages.
*   **Reduced Noise:** Advanced false positive filtering to focus on real security risks.

## Getting Started: Secure Your Code in Minutes

Integrate the Claude Code Security Reviewer into your GitHub workflow with these simple steps:

1.  **Add the GitHub Action:** Include the following snippet in your repository's `.github/workflows/security.yml` file:

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

2.  **Configure Your API Key:**  Make sure to set your `CLAUDE_API_KEY` in your GitHub Secrets settings. This key must be enabled for both the Claude API and Claude Code usage.

## Configuration Options

Customize the behavior of the security review with these configuration options:

### Action Inputs

| Input | Description | Default | Required |
|---|---|---|---|
| `claude-api-key` | Your Anthropic Claude API key for security analysis. Ensure it's enabled for both the Claude API and Claude Code. | None | Yes |
| `comment-pr` | Enable or disable commenting on pull requests with findings. | `true` | No |
| `upload-results` | Enable or disable uploading results as artifacts. | `true` | No |
| `exclude-directories` |  Comma-separated list of directories to exclude from scanning. | None | No |
| `claude-model` |  Specify the Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use.  | `claude-opus-4-1-20250805` | No |
| `claudecode-timeout` | Set the timeout for ClaudeCode analysis in minutes. | `20` | No |
| `run-every-commit` | Enable to run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits. | `false` | No |
| `false-positive-filtering-instructions` | Path to a custom file containing false positive filtering instructions. | None | No |
| `custom-security-scan-instructions` | Path to a custom security scan instructions text file. | None | No |

### Action Outputs

| Output | Description |
|---|---|
| `findings-count` | The total number of security findings. |
| `results-file` |  Path to the JSON file containing the results. |

## How It Works

The Claude Code Security Reviewer performs the following steps:

1.  **PR Analysis:** Analyzes the pull request diff to identify code changes.
2.  **Contextual Review:**  Examines the modified code with deep semantic understanding.
3.  **Finding Generation:**  Identifies security issues with detailed explanations, severity ratings, and remediation advice.
4.  **False Positive Filtering:**  Removes low-impact or likely false positive findings.
5.  **PR Comments:** Posts findings as review comments directly within the pull request.

## Security Analysis Capabilities

### Vulnerabilities Detected:

*   **Injection Attacks:** SQL, command, LDAP, XPath, NoSQL, XXE
*   **Authentication & Authorization:** Broken authentication, privilege escalation, insecure direct object references, bypass logic, session flaws
*   **Data Exposure:** Hardcoded secrets, sensitive data logging, information disclosure, PII handling violations
*   **Cryptographic Issues:** Weak algorithms, improper key management, insecure random number generation
*   **Input Validation:** Missing validation, improper sanitization, buffer overflows
*   **Business Logic Flaws:** Race conditions, TOCTOU issues
*   **Configuration Security:** Insecure defaults, missing security headers, permissive CORS
*   **Supply Chain:** Vulnerable dependencies, typosquatting risks
*   **Code Execution:** RCE via deserialization, pickle injection, eval injection
*   **Cross-Site Scripting (XSS):** Reflected, stored, and DOM-based XSS

### False Positive Filtering:

The tool automatically filters out:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

The false positive filtering can also be customized to fit the security needs of your specific project.

### Benefits over Traditional SAST:

*   **Semantic Understanding:** Analyzes code meaning, not just patterns.
*   **Reduced False Positives:** AI-powered analysis decreases noise.
*   **Detailed Explanations:** Provides clear explanations and remediation guidance.
*   **Customizable:** Adapts to your organization's security policies.

## Advanced Features

### Claude Code Integration: `/security-review` Command

Use the `/security-review` slash command within Claude Code to initiate a comprehensive security review of your code. Customize the command by editing the `.claude/commands/security-review.md` file.

### Custom Scanning Configuration

Tailor the scanner's behavior by setting custom scanning and false positive filtering instructions. Refer to the [`docs/`](docs/) folder for more details.

## Testing

Ensure the functionality by running the test suite:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

For assistance or to report issues:

*   Open an issue in this repository.
*   Review the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.