# OWASP Nettacker: Your All-in-One Security Tool for Automated Penetration Testing

OWASP Nettacker is a powerful, open-source framework designed for automated penetration testing and information gathering, empowering security professionals and ethical hackers to efficiently identify and address vulnerabilities. ([View on GitHub](https://github.com/OWASP/Nettacker))

## Key Features:

*   **Modular Architecture:** Execute individual modules for specific tasks, like port scanning, directory discovery, and vulnerability checks.
*   **Multi-Protocol & Multithreaded Scanning:** Supports HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, XML-RPC, and more, with parallel scanning for speed.
*   **Comprehensive Reporting:** Generate reports in HTML, JSON, CSV, and plain text formats for easy analysis and sharing.
*   **Built-in Database & Drift Detection:** Store scan results for historical comparisons, enabling the detection of new hosts, open ports, or vulnerabilities.
*   **Flexible Interface:** Utilize a command-line interface (CLI), REST API, and a user-friendly web UI for diverse integration options.
*   **Evasion Techniques:** Employ configurable delays, proxy support, and randomized user-agents to bypass detection.
*   **Versatile Target Input:** Target single IPs, IP ranges, CIDR blocks, domain names, and URLs, or load targets from a file.

## Use Cases:

*   **Penetration Testing:** Automate reconnaissance, misconfiguration checks, and vulnerability assessments.
*   **Reconnaissance & Vulnerability Assessment:** Map hosts, identify open ports and services, and perform credential brute-forcing.
*   **Attack Surface Mapping:** Quickly discover exposed hosts, ports, subdomains, and services.
*   **Bug Bounty Recon:** Speed up reconnaissance tasks like subdomain enumeration and default credential checks.
*   **Network Vulnerability Scanning:** Efficiently scan IPs, IP ranges, or entire CIDR blocks.
*   **Shadow IT & Asset Discovery:** Identify unmanaged or forgotten hosts and services.
*   **CI/CD & Compliance Monitoring:** Track infrastructure changes and detect vulnerabilities.

## Quick Start (Docker):

### CLI

```bash
# Basic port scan
docker run owasp/nettacker -i 192.168.0.1 -m port_scan
# Scan a network for port 22
docker run owasp/nettacker -i 192.168.0.0/24 -m port_scan -g 22
# Scan subdomains for http/https
docker run owasp/nettacker -i owasp.org -d -s -m http_status_scan
# Get help
docker run owasp/nettacker --help
```

### Web UI

```bash
$ docker-compose up
```

*   Access the Web GUI at `https://localhost:5000` and use the API Key in the CLI logs to log in.
*   The local database is `.nettacker/data/nettacker.db` (sqlite).
*   Results are stored in `.nettacker/data/results`.
*   `docker-compose` shares your Nettacker folder, preserving your data.

## Community & Resources:

*   **Project Home:** [OWASP Nettacker](https://owasp.org/nettacker)
*   **Documentation:** [Nettacker Documentation](https://nettacker.readthedocs.io)
*   **GitHub Repository:** [OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Docker Image:** [OWASP/Nettacker on Docker Hub](https://hub.docker.com/r/owasp/nettacker)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on https://owasp.slack.com
*   **Donate:** [OWASP Donation](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)

## Thanks to Our Contributors!

OWASP Nettacker thrives on community contributions. A huge thank you to all contributors!

[![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)](https://github.com/OWASP/Nettacker/graphs/contributors)

## Adopters

We're grateful to the organizations, community projects, and individuals who adopt and rely on OWASP Nettacker for their security workflows. If you're using OWASP Nettacker, add your details to the [ADOPTERS.md](ADOPTERS.md) file!

## Google Summer of Code (GSoC)

OWASP Nettacker participates in the Google Summer of Code initiative. Thanks to Google and all the students who have contributed!

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers Over Time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)