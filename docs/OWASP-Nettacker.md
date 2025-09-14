# OWASP Nettacker: Automated Penetration Testing and Information Gathering

**OWASP Nettacker is a powerful open-source framework designed to streamline penetration testing, reconnaissance, and vulnerability assessment.**  

[View the original repository on GitHub](https://github.com/OWASP/Nettacker)

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"><img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**Disclaimer:** *This software is intended for authorized penetration testing and ethical security assessments. Use it responsibly and with explicit permission.*

![2018-01-19_0-45-07](https://user-images.githubusercontent.com/7676267/35123376-283d5a3e-fcb7-11e7-9b1c-92b78ed4fecc.gif)

## Key Features

*   **Modular Architecture:** Easily customize your scans with a modular design that lets you choose which tests to run, including port scanning, service detection, and credential brute-forcing.
*   **Multi-Protocol & Multithreaded Scanning:** Supports a wide range of protocols like HTTP/HTTPS, FTP, SSH, SMB, SMTP, and more, with parallel scanning for speed.
*   **Comprehensive Output:** Generate detailed reports in multiple formats (HTML, JSON, CSV, TXT) to share findings effectively.
*   **Built-in Database & Drift Detection:** Track scan history and identify changes in your infrastructure over time for improved security posture.
*   **CLI, REST API & Web UI:** Choose your preferred interface (command-line, API, or web-based) for maximum flexibility.
*   **Evasion Techniques:** Utilize proxy support, user-agent randomization, and delays to bypass detection mechanisms.
*   **Flexible Target Specification:** Scan individual IPs, ranges, CIDR blocks, domain names, and URLs, or import targets from a list.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, vulnerability assessments, and service discovery to streamline your penetration testing workflow.
*   **Recon & Vulnerability Assessment:** Identify live hosts, open ports, potential vulnerabilities, and default credentials.
*   **Attack Surface Mapping:** Quickly discover exposed assets, open ports, and subdomains for both internal and external assets.
*   **Bug Bounty Recon:** Automate common reconnaissance tasks to find bugs faster.
*   **Network Vulnerability Scanning:** Perform efficient network assessments across IPs, ranges, or entire CIDR blocks.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged or forgotten hosts and services.
*   **CI/CD & Compliance Monitoring:** Integrate with your CI/CD pipeline to detect new vulnerabilities and track changes.

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

*   Access the Web GUI at `https://localhost:5000` (or the address provided by your docker setup).
*   Use the API Key displayed in the CLI output for login.
*   The local database is located at `.nettacker/data/nettacker.db` (sqlite).
*   Default results are saved in `.nettacker/data/results`.
*   Your nettacker folder is shared so your data won't be lost when using `docker-compose down`
*   View the API key by running `docker logs nettacker_nettacker`.
*   More details and install without docker: [Installation](https://nettacker.readthedocs.io/en/latest/Installation)

## Resources

*   **Project Home Page:** [OWASP Nettacker](https://owasp.org/nettacker)
*   **Documentation:** [Nettacker Documentation](https://nettacker.readthedocs.io)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on https://owasp.slack.com
*   **Installation:** [Installation](https://nettacker.readthedocs.io/en/latest/Installation)
*   **Usage:** [Usage](https://nettacker.readthedocs.io/en/latest/Usage)
*   **GitHub Repository:** [OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Docker Image:** [OWASP Nettacker on Docker Hub](https://hub.docker.com/r/owasp/nettacker)
*   **Dockerfile Instructions:** [Using the Dockerfile](https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker)
*   **OpenHub:** [OWASP Nettacker on OpenHub](https://www.openhub.net/p/OWASP-Nettacker)
*   **Donate:** [Donate to OWASP Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More:** [Secologist Article](https://www.secologist.com/open-source-projects)

## Community & Contributions

OWASP Nettacker is a community-driven project, and we appreciate contributions from everyone.

![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)

### Adopters

We're grateful to the organizations, community projects, and individuals who adopt and rely on OWASP Nettacker for their security workflows.

If you’re using OWASP Nettacker in your organization or project, we’d love to hear from you! Feel free to add your details to the [ADOPTERS.md](ADOPTERS.md) file by submitting a pull request or reach out to us via GitHub issues. Let’s showcase how Nettacker is making a difference in the security community!

 See [ADOPTERS.md](ADOPTERS.md) for details.

## Google Summer of Code (GSoC)

*   ☀️ OWASP Nettacker participated in the Google Summer of Code Initiative.
*   🙏 Thanks to Google Summer of Code and all the students who contributed to this project:

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers Over Time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)