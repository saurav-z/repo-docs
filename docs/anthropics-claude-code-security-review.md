# AI-Powered Code Security Review with Claude Code

**Enhance your code security with intelligent AI analysis by integrating the Claude Code Security Reviewer, your AI-powered security analysis solution, directly into your GitHub workflows.** [Learn more about the original repo](https://github.com/anthropics/claude-code-security-review).

## Key Features

*   ✅ **AI-Powered Analysis:** Leverage Claude's advanced reasoning for deep semantic understanding of code.
*   ✅ **Diff-Aware Scanning:** Only analyzes changed files in pull requests for efficient reviews.
*   ✅ **Automated PR Comments:** Receives automatic comments on pull requests highlighting security findings.
*   ✅ **Contextual Understanding:** Goes beyond pattern matching to grasp the code's intent and purpose.
*   ✅ **Language Agnostic:** Compatible with any programming language.
*   ✅ **False Positive Filtering:** Reduces noise through advanced filtering, focusing on real vulnerabilities.

## How It Works: A Streamlined Security Workflow

1.  **Pull Request Trigger:** The workflow starts when a pull request is opened.
2.  **Change Analysis:** Claude analyzes the code changes within the pull request's diff.
3.  **Contextual Review:** The AI examines the altered code in context, understanding its purpose and potential security implications.
4.  **Vulnerability Detection:** Security issues are identified, accompanied by detailed explanations, severity ratings, and remediation guidance.
5.  **False Positive Filtering:** Advanced filtering helps minimize false positives to enhance the clarity of findings.
6.  **PR Commenting:** Findings are posted as comments directly on the relevant lines of code.

## Installation & Setup

Integrate this security review into your GitHub workflow is streamlined with the Quick Start guide.  

**Quick Start:**

Add this to your repository's `.github/workflows/security.yml`:

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

Customize the action's behavior with several configuration options.

### Action Inputs

| Input | Description | Default | Required |
|-------|-------------|---------|----------|
| `claude-api-key` | Anthropic Claude API key for security analysis. <br>*Note*: This API key needs to be enabled for both the Claude API and Claude Code usage. | None | Yes |
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

## Security Analysis Capabilities

This tool is designed to detect a broad range of security vulnerabilities and it offers robust false positive filtering.

### Types of Vulnerabilities Detected

*   Injection Attacks (SQL, command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization Issues
*   Data Exposure Risks
*   Cryptographic Weaknesses
*   Input Validation Problems
*   Business Logic Flaws
*   Configuration Security Issues
*   Supply Chain Vulnerabilities
*   Code Execution Risks
*   Cross-Site Scripting (XSS)

### False Positive Filtering

The tool filters out low-impact and false-positive-prone findings, including:
*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

The false positive filtering can also be customized to fit the project's specific security needs.

### Claude Code Integration: /security-review Command 

The Claude Code Security Reviewer offers a built-in `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) that seamlessly integrates the security analysis into your Claude Code development environment. Execute `/security-review` for a comprehensive security review of all ongoing modifications.

Customize this command by copying `security-review.md` from the repository to the `.claude/commands/` directory and modifying the security analysis instructions.

## Testing

To test the functionality, run the test suite:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

For assistance:
*   Open an issue on GitHub.
*   Examine the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging insights.

## License

Distributed under the MIT License - see the [LICENSE](LICENSE) file.