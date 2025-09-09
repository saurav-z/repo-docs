# OWASP Nettacker: Your Automated Security Testing and Information Gathering Toolkit

**OWASP Nettacker is an open-source penetration testing framework that empowers security professionals and ethical hackers to discover vulnerabilities and assess network security posture efficiently.**

[Go to the original repo](https://github.com/OWASP/Nettacker)

## Key Features

*   **Modular Architecture:** Customize your scans with modular components for tasks like port scanning, vulnerability detection, and credential brute-forcing.
*   **Multi-Protocol & Multithreaded Scanning:**  Supports various protocols (HTTP/HTTPS, FTP, SSH, etc.) and utilizes multithreading for faster scans.
*   **Comprehensive Reporting:** Generate reports in various formats (HTML, JSON, CSV, text) for easy analysis.
*   **Database & Drift Detection:**  Track scan results over time to detect changes and identify new vulnerabilities or assets.
*   **CLI, REST API & Web UI:** Provides multiple interfaces for flexible use and integration.
*   **Evasion Techniques:** Employ configurable delays, proxy support, and user-agent randomization to bypass detection.
*   **Flexible Target Input:**  Accepts single IPs, IP ranges, CIDR blocks, domain names, and URLs.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, service discovery, and vulnerability scanning.
*   **Reconnaissance & Vulnerability Assessment:** Map hosts, identify open ports and services, and perform credential brute-forcing.
*   **Attack Surface Mapping:**  Quickly discover exposed hosts, subdomains, and services.
*   **Bug Bounty Recon:**  Automate common reconnaissance tasks.
*   **Network Vulnerability Scanning:**  Efficiently scan networks using a modular, multithreaded approach.
*   **Shadow IT & Asset Discovery:** Uncover forgotten or unmanaged assets.
*   **CI/CD & Compliance Monitoring:** Integrate Nettacker into pipelines to track changes and vulnerabilities.

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
* Use the API Key displayed in the CLI to login to the Web GUI
* Web GUI is accessible from your (https://localhost:5000) or https://nettacker-api.z3r0d4y.com:5000/ (pointed to your localhost)
* The local database is `.nettacker/data/nettacker.db` (sqlite).
* Default results path is `.nettacker/data/results`
* `docker-compose` will share your nettacker folder, so you will not lose any data after `docker-compose down`
* To see the API key in you can also run `docker logs nettacker_nettacker`.
* More details and install without docker https://nettacker.readthedocs.io/en/latest/Installation

## Resources

*   **Project Home Page:** [OWASP Nettacker](https://owasp.org/nettacker)
*   **Documentation:** [Nettacker Documentation](https://nettacker.readthedocs.io)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ)
*   **Installation:** [Installation Guide](https://nettacker.readthedocs.io/en/latest/Installation)
*   **Usage:** [Usage Guide](https://nettacker.readthedocs.io/en/latest/Usage)
*   **Docker Image:** [Docker Hub](https://hub.docker.com/r/owasp/nettacker)

## Contribute & Support

OWASP Nettacker is an open-source project driven by community contributions.  Join us in making Nettacker even better!

*   **Contributors:** Thanks to all our awesome contributors!
*   **Adopters:** See [ADOPTERS.md](ADOPTERS.md) to find organizations who use Nettacker!  Please submit a PR to add your organization.
*   **Donate:** [OWASP Donate](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)

## Google Summer of Code (GSoC) Project

OWASP Nettacker is participating in the Google Summer of Code Initiative. Thanks to the Google Summer of Code Initiative and all the students who contributed to this project during their summer breaks.
<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers over time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)