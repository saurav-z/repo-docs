# OWASP Nettacker: Automate Penetration Testing and Information Gathering

**OWASP Nettacker is an open-source, Python-based framework designed to streamline your cybersecurity tasks, from reconnaissance to vulnerability assessment.** ([Original Repository](https://github.com/OWASP/Nettacker))

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"><img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**Disclaimer:** This software is intended for ethical and responsible use only. Always obtain explicit permission before scanning systems or applications.

## Key Features

*   **Modular Architecture:** Execute individual modules for specific tasks like port scanning, subdomain enumeration, or vulnerability checks, giving you fine-grained control.
*   **Multi-Protocol & Multithreaded Scanning:** Supports various protocols (HTTP/HTTPS, FTP, SSH, etc.) and parallel scanning for enhanced speed.
*   **Comprehensive Reporting:** Generate detailed reports in HTML, JSON, CSV, and plain text formats.
*   **Built-in Database & Drift Detection:**  Store scan results for easy comparison and detection of changes in your infrastructure, ideal for CI/CD.
*   **CLI, REST API & Web UI:** Offers flexibility with command-line interface, REST API for automation, and a user-friendly web interface.
*   **Evasion Techniques:** Employ configurable delays, proxy support, and user-agent randomization to avoid detection.
*   **Flexible Target Input:** Accepts single IPs, IP ranges, CIDR blocks, domain names, and URLs, all mixed in a single scan.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance and vulnerability assessments.
*   **Recon & Vulnerability Assessment:** Identify live hosts, open ports, services, and potential vulnerabilities.
*   **Attack Surface Mapping:** Quickly discover exposed assets and potential entry points.
*   **Bug Bounty Recon:**  Accelerate reconnaissance for bug bounty programs.
*   **Network Vulnerability Scanning:** Perform efficient and scalable network assessments.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged assets and track changes over time.
*   **CI/CD & Compliance Monitoring:** Integrate Nettacker into your CI/CD pipelines to monitor infrastructure changes and detect new vulnerabilities.

## Quick Start (Docker)

### CLI

```bash
# Basic port scan on a single IP address:
docker run owasp/nettacker -i 192.168.0.1 -m port_scan

# Scan the entire Class C network for devices with port 22 open:
docker run owasp/nettacker -i 192.168.0.0/24 -m port_scan -g 22

# Scan all subdomains of 'owasp.org' for http/https services and return HTTP status code
docker run owasp/nettacker -i owasp.org -d -s -m http_status_scan

# Display Help
docker run owasp/nettacker --help
```

### Web UI

```bash
docker-compose up
```
*   Use the API Key displayed in the CLI to login to the Web GUI
*   Web GUI is accessible from your (https://localhost:5000) or https://nettacker-api.z3r0d4y.com:5000/ (pointed to your localhost)
*   The local database is `.nettacker/data/nettacker.db` (sqlite).
*   Default results path is `.nettacker/data/results`
*   `docker-compose` will share your nettacker folder, so you will not lose any data after `docker-compose down`
*   To see the API key in you can also run `docker logs nettacker_nettacker`.
*   More details and install without docker https://nettacker.readthedocs.io/en/latest/Installation

##  Links

*   **OWASP Nettacker Project Home Page:** [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **GitHub Repository:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Docker Image:** [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   **Donate:** [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)

## Contributors

OWASP Nettacker thrives on community contributions! We appreciate all contributions to the project.

![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)

## Adopters

We appreciate all organizations using Nettacker! Add your organization to the [ADOPTERS.md](ADOPTERS.md) file.

## Google Summer of Code (GSoC)

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

Thanks to the Google Summer of Code Initiative and all the students who contributed to this project!

## Stargazers over time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)