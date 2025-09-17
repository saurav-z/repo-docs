# OWASP Nettacker: Automated Penetration Testing & Information Gathering

**OWASP Nettacker is an open-source framework designed to automate penetration testing tasks, helping security professionals and ethical hackers efficiently identify network vulnerabilities.**

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"><img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**[View the original repo on GitHub](https://github.com/OWASP/Nettacker)**

***

**DISCLAIMER:** This software is intended for ethical and authorized use only.  Do not use Nettacker to target systems or applications without explicit permission.  The contributors are not responsible for any illegal use.

***

## Key Features

*   **Modular Architecture:** Easily customize scans with a wide range of modules, each dedicated to a specific task.
*   **Multi-Protocol & Multithreaded Scanning:** Supports HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, XML-RPC, and more, with parallel scanning for speed.
*   **Comprehensive Reporting:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Built-in Database & Drift Detection:** Track scan results over time with the database to detect changes and new vulnerabilities.
*   **CLI, REST API & Web UI:** Provides command-line, API, and web interface options for flexible use.
*   **Evasion Techniques:** Configure delays, proxy support, and randomized user-agents to bypass security measures.
*   **Flexible Target Input:** Supports single IPs, IP ranges, CIDR blocks, domain names, and URLs.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, service discovery, and vulnerability assessments.
*   **Recon & Vulnerability Assessment:** Discover hosts, ports, services, and potential vulnerabilities.
*   **Attack Surface Mapping:** Quickly identify exposed assets and services.
*   **Bug Bounty Recon:** Streamline reconnaissance tasks for bug bounty hunting.
*   **Network Vulnerability Scanning:** Perform efficient, parallel scans across networks.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged assets through historical scan data.
*   **CI/CD & Compliance Monitoring:** Track infrastructure changes and vulnerabilities in CI/CD pipelines.

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

## Contributing

OWASP Nettacker is an open-source project built on collaboration. Join the OWASP community and contribute to Nettacker's development!

## Adopters

We’re grateful to the organizations, community projects, and individuals who adopt and rely on OWASP Nettacker for their security workflows.

If you’re using OWASP Nettacker in your organization or project, we’d love to hear from you! Feel free to add your details to the [ADOPTERS.md](ADOPTERS.md) file by submitting a pull request or reach out to us via GitHub issues. Let’s showcase how Nettacker is making a difference in the security community!

 See [ADOPTERS.md](ADOPTERS.md) for details.

## Google Summer of Code (GSoC)

OWASP Nettacker participated in Google Summer of Code and is thankful for the contributions of the students!
<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Resources

*   **OWASP Nettacker Project Home Page:** [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on https://owasp.slack.com
*   **Installation:** [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)
*   **Usage:** [https://nettacker.readthedocs.io/en/latest/Usage](https://nettacker.readthedocs.io/en/latest/Usage)
*   **GitHub Repo:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Docker Image:** [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   **How to use the Dockerfile:** [https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker](https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker)
*   **OpenHub:** [https://www.openhub.net/p/OWASP-Nettacker](https://www.openhub.net/p/OWASP-Nettacker)
*   **Donate:** [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More:** [https://www.secologist.com/open-source-projects](https://www.secologist.com/open-source-projects)

## Stargazers over Time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)