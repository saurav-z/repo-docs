# Secure Your Code with AI: Claude Code Security Review

**Find and fix vulnerabilities automatically with the Claude Code Security Review GitHub Action, leveraging Anthropic's Claude to provide intelligent, context-aware security analysis.** Learn more at the [original repository](https://github.com/anthropics/claude-code-security-review).

## Key Features

*   ✅ **AI-Powered Security:** Uses Claude's advanced reasoning for deep semantic understanding of code and vulnerability detection.
*   🔍 **Diff-Aware Analysis:** Analyzes only the changed files in pull requests, saving time and resources.
*   💬 **Automated PR Comments:** Directly comments on pull requests with security findings, providing immediate feedback.
*   🧠 **Contextual Understanding:** Goes beyond pattern matching to understand code's purpose and potential security implications.
*   🌐 **Language Agnostic:** Works with any programming language, making it versatile for diverse projects.
*   🚫 **Advanced False Positive Filtering:** Reduces noise and focuses on real vulnerabilities, improving efficiency.

## Getting Started: Integrate with GitHub Actions

1.  **Add to your workflow:** Include the following snippet in your `.github/workflows/security.yml` file:

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

2.  **Configure:** Set your Anthropic Claude API key in your repository's secrets (`CLAUDE_API_KEY`).

## Configuration Options

Customize the behavior of the security review using the following inputs:

| Input                        | Description                                                                                                                                                                                              | Default                  | Required |
| :--------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------- | :------- |
| `claude-api-key`            | Your Anthropic Claude API key. **Note:** This API key needs to be enabled for both the Claude API and Claude Code usage.                                                                                 | None                     | Yes      |
| `comment-pr`                 | Whether to comment on pull requests with findings.                                                                                                                                                        | `true`                   | No       |
| `upload-results`             | Whether to upload results as artifacts.                                                                                                                                                                   | `true`                   | No       |
| `exclude-directories`        | Comma-separated list of directories to exclude from scanning.                                                                                                                                            | None                     | No       |
| `claude-model`               | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use. Defaults to Opus 4.1.                                                                          | `claude-opus-4-1-20250805` | No       |
| `claudecode-timeout`         | Timeout for ClaudeCode analysis in minutes.                                                                                                                                                              | `20`                     | No       |
| `run-every-commit`           | Run ClaudeCode on every commit (skips cache check). **Warning:** May increase false positives on PRs with many commits.                                                                                 | `false`                  | No       |
| `false-positive-filtering-instructions` | Path to a custom false positive filtering instructions text file.                                                                                                                                | None                     | No       |
| `custom-security-scan-instructions` | Path to a custom security scan instructions text file to append to the audit prompt.                                                                                                                                | None                     | No       |

### Action Outputs

| Output         | Description                                   |
| :------------- | :-------------------------------------------- |
| `findings-count` | Total number of security findings.        |
| `results-file`  | Path to the results JSON file.              |

## How It Works

1.  **PR Analysis:** The action analyzes the pull request diff.
2.  **Contextual Review:** Claude examines code changes, understanding their context.
3.  **Finding Generation:** Security issues are identified with explanations and remediation advice.
4.  **False Positive Filtering:** Unnecessary findings are filtered out.
5.  **PR Comments:** Findings are posted as comments within the pull request.

## Security Analysis Capabilities

### Types of Vulnerabilities Detected

*   Injection Attacks
*   Authentication & Authorization Flaws
*   Data Exposure Risks
*   Cryptographic Issues
*   Input Validation Problems
*   Business Logic Flaws
*   Configuration Security Issues
*   Supply Chain Vulnerabilities
*   Code Execution Risks
*   Cross-Site Scripting (XSS)

### False Positive Filtering

The tool automatically excludes many low-impact and false positive prone findings:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

## Claude Code Integration: `/security-review` Command

Use the `/security-review` slash command within your Claude Code development environment for quick security checks. Customize the command by copying and editing the `security-review.md` file in your project's `.claude/commands/` folder.

## Custom Scanning Configuration

Customize the security scan and false positive filtering using custom instructions - see the [`docs/`](docs/) folder for details.

## Testing

Run the test suite with:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support

*   Open an issue in this repository
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history)

## License

MIT License - see [LICENSE](LICENSE) for details.