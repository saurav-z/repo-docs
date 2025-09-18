# OWASP Nettacker: Automate Your Penetration Testing and Information Gathering

**OWASP Nettacker is a powerful, open-source framework designed to streamline penetration testing, vulnerability assessments, and network security audits, making it a go-to tool for ethical hackers and security professionals.** ([Original Repo](https://github.com/OWASP/Nettacker))

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"><img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**DISCLAIMER:  Use responsibly and ethically. Obtain proper authorization before using this software. Contributors are not responsible for any illegal use.**

![2018-01-19_0-45-07](https://user-images.githubusercontent.com/7676267/35123376-283d5a3e-fcb7-11e7-9b1c-92b78ed4fecc.gif)

## Key Features

*   **Modular Architecture:** Perform targeted assessments with individual modules for port scanning, subdomain enumeration, vulnerability checks, and credential brute-forcing.
*   **Multi-Protocol & Multithreaded Scanning:** Supports HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, XML-RPC, and parallel scanning for speed.
*   **Comprehensive Reporting:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Database & Drift Detection:** Store scan data for historical analysis and identify changes (new hosts, ports, vulnerabilities) for CI/CD pipelines.
*   **CLI, REST API & Web UI:** Choose your preferred interface – command-line, REST API, or a user-friendly web UI.
*   **Evasion Techniques:** Implement delays, proxy support, and randomized user-agents to avoid detection.
*   **Flexible Target Input:** Accepts single IPs, ranges, CIDR blocks, domain names, and URLs.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, misconfiguration checks, and vulnerability scanning.
*   **Reconnaissance & Vulnerability Assessment:** Discover live hosts, open ports, services, and potential weaknesses.
*   **Attack Surface Mapping:** Quickly identify exposed assets and services.
*   **Bug Bounty Recon:** Automate and scale common reconnaissance tasks.
*   **Network Vulnerability Scanning:** Efficiently scan networks and assess vulnerabilities.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged or forgotten assets.
*   **CI/CD & Compliance Monitoring:** Track infrastructure changes and detect new vulnerabilities.

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

*   Use the API Key displayed in the CLI to login to the Web GUI
*   Web GUI is accessible from your (https://localhost:5000) or https://nettacker-api.z3r0d4y.com:5000/ (pointed to your localhost)
*   The local database is `.nettacker/data/nettacker.db` (sqlite).
*   Default results path is `.nettacker/data/results`
*   `docker-compose` will share your nettacker folder, so you will not lose any data after `docker-compose down`
*   To see the API key in you can also run `docker logs nettacker_nettacker`.
*   More details and install without docker https://nettacker.readthedocs.io/en/latest/Installation

##  Contributing & Community

OWASP Nettacker thrives on community contributions.  Join us in building a robust and evolving security tool!

[![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)](https://github.com/OWASP/Nettacker/graphs/contributors)

###  Adopters

We appreciate the organizations, projects, and individuals using Nettacker. Add your info to the [ADOPTERS.md](ADOPTERS.md) file!

### Google Summer of Code (GSoC)

☀️ OWASP Nettacker has participated in the Google Summer of Code initiative.

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers over time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)

## Resources

*   **OWASP Nettacker Project Home Page:** [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on https://owasp.slack.com
*   **Installation:** [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)
*   **Usage:** [https://nettacker.readthedocs.io/en/latest/Usage](https://nettacker.readthedocs.io/en/latest/Usage)
*   **GitHub Repo:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Docker Image:** [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   **Dockerfile Usage:** [https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker](https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker)
*   **OpenHub:** [https://www.openhub.net/p/OWASP-Nettacker](https://www.openhub.net/p/OWASP-Nettacker)
*   **Donate:** [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More:** [https://www.secologist.com/open-source-projects](https://www.secologist.com/open-source-projects)