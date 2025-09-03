# OWASP Nettacker: Your Automated Penetration Testing and Information Gathering Toolkit

**OWASP Nettacker** empowers security professionals and ethical hackers to efficiently perform reconnaissance, vulnerability assessments, and network security audits. ([View on GitHub](https://github.com/OWASP/Nettacker))

## Key Features

*   **Modular Architecture:** Customize scans with individual modules for tasks like port scanning, directory discovery, and vulnerability checks.
*   **Multi-Protocol Support:** Scan various protocols including HTTP/HTTPS, FTP, SSH, SMB, and more.
*   **Multithreaded Scanning:** Leverage parallel scanning for faster results.
*   **Comprehensive Output:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Built-in Database:** Store and compare scan results to track changes and detect new vulnerabilities.
*   **CLI, REST API & Web UI:** Choose between command-line, API, or web interface control.
*   **Evasion Techniques:** Implement configurable delays, proxy support, and user-agent randomization to evade detection.
*   **Flexible Targeting:** Scan single IPs, IP ranges, CIDR blocks, domains, and URLs.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance and vulnerability assessments.
*   **Reconnaissance and Vulnerability Assessment:** Identify live hosts, open ports, services, and potential vulnerabilities.
*   **Attack Surface Mapping:** Quickly discover exposed assets.
*   **Bug Bounty Recon:** Automate reconnaissance tasks.
*   **Network Vulnerability Scanning:** Conduct efficient large-scale network assessments.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged assets.
*   **CI/CD & Compliance Monitoring:** Integrate into pipelines to detect vulnerabilities.

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

*   Access the Web GUI at `https://localhost:5000`.
*   Use the API Key displayed in the CLI to login.
*   The local database is located at `.nettacker/data/nettacker.db` (sqlite).
*   The default results path is `.nettacker/data/results`.
*   `docker-compose` shares your nettacker folder, preserving data.
*   View the API key by running `docker logs nettacker_nettacker`.
*   For more details and installation without Docker, see the [documentation](https://nettacker.readthedocs.io/en/latest/Installation).

## Contributors

OWASP Nettacker is an open-source project, and the OWASP community actively contributes to its development.  See the [ADOPTERS.md](ADOPTERS.md) for a list of adopters.

[![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)](https://github.com/OWASP/Nettacker/graphs/contributors)

## Google Summer of Code

OWASP Nettacker participates in the Google Summer of Code initiative.

## Star History

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)