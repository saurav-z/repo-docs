# OWASP Nettacker: Automated Penetration Testing & Information Gathering

**OWASP Nettacker is an open-source, powerful, and modular framework designed to automate penetration testing and reconnaissance, enabling you to identify vulnerabilities and assess network security effectively.**

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![Repo Size](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)
<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"><img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**Disclaimer:** This software is designed for ethical security assessments. Use responsibly and with proper authorization. The contributors are not liable for any misuse.

## Key Features

*   **Modular Architecture:** Utilize independent modules for tasks like port scanning, service detection, and vulnerability checks, giving you control over your scans.
*   **Multi-Protocol & Multithreaded Scanning:** Scan HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, XML-RPC, and run scans concurrently for increased efficiency.
*   **Comprehensive Output:** Generate reports in HTML, JSON, CSV, and plain text formats for easy analysis and sharing.
*   **Built-in Database & Drift Detection:** Store scan results for historical comparison, helping you identify new hosts, ports, or vulnerabilities.
*   **CLI, REST API & Web UI:** Integrate Nettacker into your workflow programmatically or through a user-friendly web interface.
*   **Evasion Techniques:** Configure delays, proxy support, and randomized user-agents to minimize detection by firewalls and intrusion detection systems.
*   **Flexible Target Specification:** Scan single IPs, IP ranges, CIDR blocks, domain names, and URLs, with options to mix target types or load them from a file.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, misconfiguration checks, service discovery, and vulnerability scanning for efficient testing.
*   **Recon & Vulnerability Assessment:** Identify live hosts, open ports, services, default credentials, directories, and perform brute-force or fuzzing attacks.
*   **Attack Surface Mapping:** Quickly discover exposed hosts, ports, subdomains, and services across both internal and external assets.
*   **Bug Bounty Recon:** Automate reconnaissance to find targets, subdomain enumeration, directory brute-forcing, and default credential checks for bug bounty hunting.
*   **Network Vulnerability Scanning:** Perform efficient parallel scans of IPs, IP ranges, CIDR blocks, or subdomains for network assessments.
*   **Shadow IT & Asset Discovery:** Use historical data and drift detection to uncover unmanaged or forgotten assets.
*   **CI/CD & Compliance Monitoring:** Track infrastructure changes and detect new vulnerabilities using stored scan history and comparison features.

## Quick Setup & Run

### CLI (Docker)

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

### Web UI (Docker)

```bash
$ docker-compose up 
```

*   Use the API Key displayed in the CLI to login to the Web GUI
*   Web GUI is accessible from your (https://localhost:5000) or https://nettacker-api.z3r0d4y.com:5000/ (pointed to your localhost)
*   The local database is `.nettacker/data/nettacker.db` (sqlite).
*   Default results path is `.nettacker/data/results`
*   `docker-compose` will share your nettacker folder, so you will not lose any data after `docker-compose down`
*   To see the API key in you can also run `docker logs nettacker_nettacker`.
*   More details and install without docker https://nettacker.readthedocs.io/en/latest/Installation

## Get Started

For detailed installation and usage instructions, please refer to the [official documentation](https://nettacker.readthedocs.io/en/latest/).

## Contributing

OWASP Nettacker is an open-source project supported by the OWASP community. Your contributions are welcome! Please see the [Contributing Guidelines](CONTRIBUTING.md) for more information.

## Adopters

We are grateful to the organizations, community projects, and individuals who use OWASP Nettacker. If you're using Nettacker, consider adding your information to the [ADOPTERS.md](ADOPTERS.md) file!

## Google Summer of Code (GSoC) Project

We are proud to have participated in the Google Summer of Code Initiative. Thanks to Google and all the students for their contributions!
<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers over time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)

## Further Resources

*   [OWASP Nettacker Project Home Page](https://owasp.org/nettacker)
*   [Documentation](https://nettacker.readthedocs.io)
*   [Slack](https://owasp.slack.com/archives/CQZGG24FQ)
*   [GitHub Repository](https://github.com/OWASP/Nettacker)