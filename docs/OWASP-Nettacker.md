# OWASP Nettacker: Your Automated Penetration Testing and Information Gathering Framework

**OWASP Nettacker is a powerful, open-source framework designed to help cybersecurity professionals and ethical hackers efficiently assess network security.**

[Go to the original repository](https://github.com/OWASP/Nettacker)

## Key Features

*   **Modular Architecture:** Execute specific tasks (port scanning, vulnerability checks, etc.) with modular components for control.
*   **Multi-Protocol & Multithreaded Scanning:** Supports HTTP/HTTPS, FTP, SSH, SMB, SMTP, and more; scans in parallel for speed.
*   **Comprehensive Reporting:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Built-in Database & Drift Detection:** Store scan results for comparison and identification of changes over time.
*   **CLI, REST API & Web UI:** Offers both programmatic integration and a user-friendly web interface for defining scans and viewing results.
*   **Evasion Techniques:** Utilize configurable delays, proxy support, and randomized user-agents to avoid detection.
*   **Flexible Target Specifications:** Accepts single IPs, IP ranges, CIDR blocks, domains, and URLs, with support for loading targets from files.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance and vulnerability assessments.
*   **Reconnaissance & Vulnerability Assessment:** Discover hosts, open ports, services, and vulnerabilities.
*   **Attack Surface Mapping:** Identify exposed assets, ports, and services rapidly.
*   **Bug Bounty Reconnaissance:** Automate subdomain enumeration and vulnerability checks.
*   **Network Vulnerability Scanning:** Perform efficient assessments across networks.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged assets and changes over time.
*   **CI/CD & Compliance Monitoring:** Integrate for infrastructure change tracking and vulnerability detection.

## Quick Start (Docker)

### CLI

```bash
# Basic port scan on a single IP address:
$ docker run owasp/nettacker -i 192.168.0.1 -m port_scan
# Scan the entire Class C network for any devices with port 22 open:
$ docker run owasp/nettacker -i 192.168.0.0/24 -m port_scan -g 22
# Scan all subdomains of 'owasp.org' for http/https services and return HTTP status code
$ docker run owasp/nettacker -i owasp.org -d -s -m http_status_scan
# Display Help
$ docker run owasp/nettacker --help
```

### Web UI

```bash
$ docker-compose up
```

*   Use the API Key displayed in the CLI to login to the Web GUI.
*   Web GUI is accessible from your (https://localhost:5000) or https://nettacker-api.z3r0d4y.com:5000/ (pointed to your localhost)
*   The local database is `.nettacker/data/nettacker.db` (sqlite).
*   Default results path is `.nettacker/data/results`
*   `docker-compose` will share your nettacker folder, so you will not lose any data after `docker-compose down`
*   To see the API key in you can also run `docker logs nettacker_nettacker`.
*   More details and install without docker https://nettacker.readthedocs.io/en/latest/Installation

## Important Notes

*   **Disclaimer:** Use this software responsibly and ethically. Do not target systems without permission.
*   See [ADOPTERS.md](ADOPTERS.md) to see who is using OWASP Nettacker.
*   OWASP Nettacker is a [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com) Project.

## Contribution & Support

OWASP Nettacker is an open-source project, built on the principles of collaboration and shared knowledge. Thank you to all of our contributors! 

[![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)](https://github.com/OWASP/Nettacker/graphs/contributors)

*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on https://owasp.slack.com
*   **GitHub repo:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Donate:** [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)