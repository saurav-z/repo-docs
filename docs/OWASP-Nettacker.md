# OWASP Nettacker: Automate Your Penetration Testing & Reconnaissance

**OWASP Nettacker is a powerful, open-source framework that helps security professionals and ethical hackers efficiently perform penetration testing, vulnerability assessments, and network security audits.** ([Original Repo](https://github.com/OWASP/Nettacker))

## Key Features:

*   **Modular Architecture:** Choose specific modules for tasks like port scanning, vulnerability detection, and credential brute-forcing.
*   **Multi-Protocol & Multithreaded Scanning:** Supports HTTP/HTTPS, FTP, SSH, and more, with parallel scanning for speed.
*   **Comprehensive Reporting:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Database & Drift Detection:** Track scan results over time to identify new hosts, open ports, and vulnerabilities.
*   **CLI, REST API & Web UI:** Offers flexible integration through command-line, REST API and a user-friendly web interface.
*   **Evasion Techniques:** Use delays, proxies, and randomized user-agents to bypass detection.
*   **Flexible Target Input:** Scan single IPs, IP ranges, CIDR blocks, domain names, or URLs.

## Use Cases:

*   Penetration Testing
*   Reconnaissance & Vulnerability Assessment
*   Attack Surface Mapping
*   Bug Bounty Reconnaissance
*   Network Vulnerability Scanning
*   Shadow IT & Asset Discovery
*   CI/CD & Compliance Monitoring

## Quick Start (Docker):

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

*   Access the Web GUI via `https://localhost:5000` (or your Docker host address) after obtaining the API key from the CLI output.
*   Local database: `.nettacker/data/nettacker.db` (SQLite)
*   Default results path: `.nettacker/data/results`
*   The `docker-compose` setup shares your Nettacker data directory, ensuring data persistence.

## Links:

*   **OWASP Nettacker Project Home Page:** [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **GitHub Repository:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Docker Image:** [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)

##  Acknowledgements

*   **Contributors:** OWASP Nettacker is built on the contributions of the vibrant OWASP community.
*   **Adopters:**  Organizations and individuals are encouraged to add themselves to the [ADOPTERS.md](ADOPTERS.md) file.
*   **Google Summer of Code:** Thank you to Google and the participating students.

## Stargazers over time
[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)