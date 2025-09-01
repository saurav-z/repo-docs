# Enhance Your Code Security with AI-Powered Analysis: Claude Code Security Reviewer

**Protect your code with intelligent security analysis.** The Claude Code Security Reviewer, a GitHub Action, uses Anthropic's Claude Code to identify and explain security vulnerabilities in your code, helping you ship more secure software. [Learn more at the original repo.](https://github.com/anthropics/claude-code-security-review)

## Key Features

*   ✅ **AI-Driven Security**: Employs Claude's advanced reasoning to deeply analyze code for vulnerabilities.
*   ✅ **Diff-Aware Scanning**: Focuses on changed files in pull requests, saving time and resources.
*   ✅ **Automated PR Comments**: Directly comments on pull requests, highlighting findings for easy review.
*   ✅ **Contextual Understanding**: Analyzes code semantics, moving beyond simple pattern matching.
*   ✅ **Language Agnostic**: Works seamlessly with any programming language.
*   ✅ **Reduced Noise**: Advanced filtering minimizes false positives, emphasizing real vulnerabilities.

## Getting Started: Quick Installation

Integrate the Claude Code Security Reviewer into your project's workflow by adding the following to your `.github/workflows/security.yml` file:

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

Customize the action with the following inputs:

### Action Inputs

| Input                     | Description                                                                                                  | Default                     | Required |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------- | -------- |
| `claude-api-key`          | Your Anthropic Claude API key. *Requires both Claude API and Claude Code usage enabled.*                     | None                        | Yes      |
| `comment-pr`              | Whether to comment on PRs with findings.                                                                     | `true`                      | No       |
| `upload-results`          | Whether to upload results as artifacts.                                                                      | `true`                      | No       |
| `exclude-directories`     | Comma-separated list of directories to exclude from scanning.                                               | None                        | No       |
| `claude-model`            | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use.    | `claude-opus-4-1-20250805` | No       |
| `claudecode-timeout`      | Timeout for ClaudeCode analysis in minutes.                                                                  | `20`                        | No       |
| `run-every-commit`        | Run ClaudeCode on every commit (skips cache check). *Warning: May increase false positives on PRs with many commits.* | `false`                     | No       |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file                     | None | No       |
| `custom-security-scan-instructions`      | Path to custom security scan instructions text file to append to audit prompt.               | None                        | No       |

### Action Outputs

| Output          | Description                        |
| --------------- | ---------------------------------- |
| `findings-count` | Total number of security findings. |
| `results-file`  | Path to the results JSON file.     |

## How It Works

1.  **PR Analysis**: The action analyzes pull requests to identify changes.
2.  **Contextual Review**: Claude examines the code changes, understanding their purpose and security implications.
3.  **Finding Generation**: Security issues are identified, with detailed explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering**: Advanced filtering reduces noise to focus on real vulnerabilities.
5.  **PR Comments**: Findings are posted as review comments on the specific lines of code.

## Security Analysis Capabilities

### Types of Vulnerabilities Detected

*   Injection Attacks (SQL, command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization issues
*   Data Exposure risks
*   Cryptographic vulnerabilities
*   Input Validation problems
*   Business Logic Flaws
*   Configuration Security issues
*   Supply Chain risks
*   Code Execution vulnerabilities
*   Cross-Site Scripting (XSS)

### False Positive Filtering

The tool automatically filters out common false positives like:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

You can also customize false positive filtering to fit your project's needs.

### Benefits Over Traditional SAST

*   **Superior Contextual Understanding**: Analyzes code semantics, not just patterns.
*   **Lower False Positives**: AI-powered analysis reduces noise.
*   **Detailed Explanations**: Provides clear explanations and remediation guidance.
*   **Customizable**: Adaptable to your organization's security requirements.

## Integration

*   **GitHub Actions**: Follow the quick start guide above.
*   **Local Development**: See the [evaluation framework documentation](claudecode/evals/README.md).

<a id="security-review-slash-command"></a>
## Enhanced Security Review: The `/security-review` Command

Within Claude Code, you can use the `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) to perform a comprehensive security review of your code directly within your development environment.

### Customization

1.  Copy the `security-review.md` file from this repository to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to customize the security analysis.

## Custom Configuration

You can customize scanning and false positive filtering instructions; see the `/docs` folder for more details.

## Testing

To validate the functionality, run the test suite:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

MIT License - See the [LICENSE](LICENSE) file for details.