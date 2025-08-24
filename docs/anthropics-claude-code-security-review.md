# Enhance Your Code Security with AI: Claude Code Security Reviewer

**Identify and remediate vulnerabilities in your code with the power of AI using the Claude Code Security Reviewer, a GitHub Action that leverages Anthropic's Claude Code for comprehensive security analysis.** ([Original Repository](https://github.com/anthropics/claude-code-security-review))

## Key Features:

*   **AI-Powered Security Analysis:** Utilizes Claude's advanced reasoning to detect vulnerabilities through deep semantic understanding.
*   **Diff-Aware Scanning:** Focuses analysis on changed files within pull requests.
*   **Automated PR Comments:** Automatically posts findings directly in your pull requests for quick review.
*   **Contextual Understanding:** Goes beyond pattern matching, comprehending code semantics and intent.
*   **Language Agnostic:** Works seamlessly with any programming language.
*   **Reduced Noise:** Includes advanced false positive filtering to focus on critical vulnerabilities.

## Getting Started: Quick Installation

Integrate the Claude Code Security Reviewer into your GitHub repository's workflow with these simple steps:

1.  Add the following configuration to your `.github/workflows/security.yml` file:

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
2.  Replace `YOUR_CLAUDE_API_KEY` with your actual Claude API key.
3.  Customize your action using the configuration options below.

## Configuration Options:

### Action Inputs:

| Input                       | Description                                                                                                                             | Default                           | Required |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | -------- |
| `claude-api-key`          | Anthropic Claude API key for security analysis                                                                                           | None                              | Yes      |
| `comment-pr`              | Whether to comment on PRs with findings                                                                                                  | `true`                            | No       |
| `upload-results`            | Whether to upload results as artifacts                                                                                                   | `true`                            | No       |
| `exclude-directories`       | Comma-separated list of directories to exclude from scanning                                                                         | None                              | No       |
| `claude-model`              | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use. Defaults to Opus 4.1.            | `claude-opus-4-1-20250805`        | No       |
| `claudecode-timeout`        | Timeout for ClaudeCode analysis in minutes                                                                                               | `20`                              | No       |
| `run-every-commit`          | Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits.                      | `false`                           | No       |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file                                                                | None                              | No       |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt                                                       | None                              | No       |

### Action Outputs:

| Output          | Description                               |
| --------------- | ----------------------------------------- |
| `findings-count` | Total number of security findings        |
| `results-file`  | Path to the results JSON file           |

## Technical Architecture:

*   **`github_action_audit.py`**: Main script for the GitHub Action.
*   **`prompts.py`**: Security audit prompt templates.
*   **`findings_filter.py`**: False positive filtering logic.
*   **`claude_api_client.py`**: Claude API client.
*   **`json_parser.py`**: Robust JSON parsing.
*   **`requirements.txt`**: Python dependencies.
*   **`test_*.py`**: Test suites.
*   **`evals/`**: Eval tooling.

## How It Works:

1.  **PR Analysis:** Analyzes the diff to understand what changed.
2.  **Contextual Review:** Examines code changes in context, understanding potential security implications.
3.  **Finding Generation:** Identifies security issues with detailed explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Reduces noise by removing low-impact or false positive findings.
5.  **PR Comments:** Findings are posted as review comments on the specific lines of code.

## Comprehensive Security Analysis Capabilities:

### Types of Vulnerabilities Detected:

*   Injection Attacks (SQL, command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization Issues
*   Data Exposure
*   Cryptographic Issues
*   Input Validation Failures
*   Business Logic Flaws
*   Configuration Security Issues
*   Supply Chain Risks
*   Code Execution Vulnerabilities
*   Cross-Site Scripting (XSS)

### Enhanced False Positive Filtering:

The tool automatically excludes a variety of low-impact and false-positive-prone findings:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

## Benefits Over Traditional SAST:

*   **Contextual Understanding:** Understands code semantics and intent, not just patterns.
*   **Lower False Positives:** AI-powered analysis reduces noise.
*   **Detailed Explanations:** Provides clear explanations and remediation guidance.
*   **Adaptive Learning:** Customizable with organization-specific security requirements.

## Integrate with Claude Code: /security-review Command

Use the `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) within your Claude Code development environment for security analysis.

### Customizing the Command

1.  Copy `security-review.md` to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to customize the security analysis.

## Custom Scanning Configuration

See the [`docs/`](docs/) folder for custom scanning and filtering instructions.

## Testing

Run the test suite to validate functionality:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging.

## License

MIT License - see [LICENSE](LICENSE) file for details.