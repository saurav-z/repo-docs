# Enhance Your Code Security with AI: Claude Code Security Reviewer

**Automatically identify and address security vulnerabilities in your code with the power of AI.** This GitHub Action leverages Anthropic's Claude Code to provide intelligent, context-aware security analysis directly within your pull requests. Learn more about this tool on the original repo:  [https://github.com/anthropics/claude-code-security-review](https://github.com/anthropics/claude-code-security-review).

## Key Features:

*   **AI-Powered Security Analysis:**  Uses Claude's advanced reasoning to understand code semantics and detect vulnerabilities.
*   **Diff-Aware Scanning:** Focuses on changed files in pull requests for efficient analysis.
*   **Automated PR Comments:**  Provides direct feedback within your pull requests, highlighting vulnerabilities.
*   **Contextual Understanding:** Goes beyond pattern matching to understand the code's intent.
*   **Language Agnostic:** Works with any programming language.
*   **Reduced False Positives:** Advanced filtering minimizes noise, focusing on real security risks.

## Getting Started

Integrate the Claude Code Security Reviewer into your GitHub workflow:

1.  Add the following to your repository's `.github/workflows/security.yml` file:

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

2.  Provide your Anthropic Claude API key as a GitHub secret named `CLAUDE_API_KEY`. **Important:**  Ensure your API key is enabled for both the Claude API and Claude Code usage.

## Configuration Options:

Customize the behavior of the security review with the following inputs:

| Input                       | Description                                                                                                                                                                                                   | Default                       | Required |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | -------- |
| `claude-api-key`            | Your Anthropic Claude API key.  **Note:** Must be enabled for both the Claude API and Claude Code usage.                                                                                                  | None                          | Yes      |
| `comment-pr`                | Whether to comment on pull requests with findings.                                                                                                                                                            | `true`                        | No       |
| `upload-results`            | Whether to upload results as artifacts.                                                                                                                                                                     | `true`                        | No       |
| `exclude-directories`       | Comma-separated list of directories to exclude from scanning.                                                                                                                                                 | None                          | No       |
| `claude-model`              | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use.                                                                                                   | `claude-opus-4-1-20250805`  | No       |
| `claudecode-timeout`        | Timeout for ClaudeCode analysis in minutes.                                                                                                                                                                   | `20`                          | No       |
| `run-every-commit`          | Run ClaudeCode on every commit (skips cache check). *Warning:* May increase false positives on PRs with many commits.                                                                                       | `false`                       | No       |
| `false-positive-filtering-instructions` | Path to a custom false positive filtering instructions text file.                                                                                                                              | None                          | No       |
| `custom-security-scan-instructions` | Path to a custom security scan instructions text file to append to the audit prompt.                                                                                                                        | None                          | No       |

### Action Outputs:

| Output          | Description                       |
| --------------- | --------------------------------- |
| `findings-count` | Total number of security findings |
| `results-file`    | Path to the results JSON file      |

## How It Works

1.  **Pull Request Analysis:** The action analyzes the diff of your pull request to understand the changes.
2.  **Contextual Review:** Claude examines the code changes in context, understanding the purpose and potential security impacts.
3.  **Finding Generation:** Security issues are identified with detailed explanations, severity levels, and remediation guidance.
4.  **False Positive Filtering:** Advanced filtering removes low-impact or false-positive findings.
5.  **PR Comments:** Findings are posted as review comments directly on the relevant lines of code.

## Security Analysis Capabilities

The tool detects a wide range of vulnerabilities, including:

*   Injection Attacks (SQL, Command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization issues (Broken authentication, privilege escalation, insecure direct object references, bypass logic, session flaws)
*   Data Exposure (Hardcoded secrets, sensitive data logging, information disclosure, PII handling violations)
*   Cryptographic Issues (Weak algorithms, improper key management, insecure random number generation)
*   Input Validation problems (Missing validation, improper sanitization, buffer overflows)
*   Business Logic Flaws (Race conditions, TOCTOU issues)
*   Configuration Security vulnerabilities (Insecure defaults, missing security headers, permissive CORS)
*   Supply Chain Risks (Vulnerable dependencies, typosquatting)
*   Code Execution risks (RCE via deserialization, pickle injection, eval injection)
*   Cross-Site Scripting (XSS) vulnerabilities (Reflected, stored, and DOM-based XSS)

### False Positive Filtering

The tool automatically filters out common low-impact or frequently reported false positives, such as:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation issues without proven impact
*   Open redirect vulnerabilities

You can also customize the filtering for your project's specific security needs.

### Benefits Over Traditional SAST

*   **Contextual Understanding:** Analyzes code semantics, not just patterns.
*   **Lower False Positives:** AI-powered analysis reduces noise.
*   **Detailed Explanations:** Provides clear explanations and remediation advice.
*   **Customization:** Adaptable to your organization's security requirements.

## Additional Features

### Claude Code Integration: /security-review Command

Claude Code offers a `/security-review` slash command that provides the same security analysis capabilities within your Claude Code development environment. Use `/security-review` to perform a comprehensive security review of your pending changes.

#### Customizing the Command

Customize the `/security-review` command:

1.  Copy the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to tailor the security analysis (e.g., add custom instructions).

## Advanced Configuration

For detailed guidance on custom scanning and false positive filtering, see the [`docs/`](docs/) folder in the repository.

## Testing

Run the test suite:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

MIT License - see [LICENSE](LICENSE) file for details.