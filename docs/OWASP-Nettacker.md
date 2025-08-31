# OWASP Nettacker: Automated Penetration Testing and Information Gathering Framework

**OWASP Nettacker is your all-in-one solution for automated security assessments, helping you identify vulnerabilities and strengthen your network defenses.** ([Original Repository](https://github.com/OWASP/Nettacker))

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200">
<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**DISCLAIMER:** *This software is designed for ethical use and penetration testing.  Always obtain proper authorization before using Nettacker to assess any systems.*

## Key Features

*   **Modular Architecture:** Utilize a modular design for individual tasks like port scanning, directory discovery, and vulnerability checks, allowing you to customize your scans.
*   **Multi-Protocol & Multithreaded Scanning:** Supports a wide range of protocols including HTTP/HTTPS, FTP, SSH, and more, while running scans in parallel for speed.
*   **Comprehensive Output:** Generate reports in HTML, JSON, CSV, and plain text formats for easy analysis and sharing.
*   **Built-in Database & Drift Detection:** Stores scan results to enable tracking changes over time and identifying new vulnerabilities, useful for CI/CD pipelines.
*   **CLI, REST API & Web UI:** Offers flexible options for interaction, including a command-line interface, a REST API for programmatic use, and an intuitive web UI.
*   **Evasion Techniques:** Implement configurable delays, proxy support, and randomized user-agents to avoid detection by security systems.
*   **Flexible Target Input:** Accepts single IPs, IP ranges, CIDR blocks, domain names, and URLs, with options to load targets from a file.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance and vulnerability scanning for efficient penetration testing workflows.
*   **Recon & Vulnerability Assessment:** Discover live hosts, open ports, services, and potential misconfigurations.
*   **Attack Surface Mapping:** Quickly map exposed hosts, ports, subdomains, and services.
*   **Bug Bounty Recon:** Automate and scale common reconnaissance tasks.
*   **Network Vulnerability Scanning:** Perform efficient, parallel scanning of networks of any size.
*   **Shadow IT & Asset Discovery:** Detect unmanaged assets and evolving network changes using historical scan data.
*   **CI/CD & Compliance Monitoring:** Integrate Nettacker into your CI/CD pipelines to track infrastructure changes and new vulnerabilities.

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

*   Access the Web GUI via `https://localhost:5000` or `https://nettacker-api.z3r0d4y.com:5000/`
*   Use the API Key displayed in the CLI to log in.
*   Local database: `.nettacker/data/nettacker.db` (SQLite)
*   Results path: `.nettacker/data/results`
*   The `docker-compose` setup shares your nettacker folder.
*   To see the API key, run `docker logs nettacker_nettacker`.
*   More installation options can be found [here](https://nettacker.readthedocs.io/en/latest/Installation).

## Contributing

OWASP Nettacker is an open-source project built on collaboration.  We welcome contributions from the community!

<details>
<summary>Contributors</summary>
OWASP Nettacker is built by the community, for the community. Thank you to our awesome contributors!

![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)
</details>

## Adopters

We're grateful to the organizations, community projects, and individuals who adopt and rely on OWASP Nettacker for their security workflows.

If you’re using OWASP Nettacker in your organization or project, we’d love to hear from you! Feel free to add your details to the [ADOPTERS.md](ADOPTERS.md) file by submitting a pull request or reach out to us via GitHub issues. Let’s showcase how Nettacker is making a difference in the security community!

## Google Summer of Code (GSoC) Project

OWASP Nettacker has participated in the Google Summer of Code (GSoC) initiative. Thank you to Google and all the students who have contributed to this project.

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Resources

*   **OWASP Nettacker Project Home Page:** [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on [https://owasp.slack.com](https://owasp.slack.com)
*   **Installation:** [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)
*   **Usage:** [https://nettacker.readthedocs.io/en/latest/Usage](https://nettacker.readthedocs.io/en/latest/Usage)
*   **GitHub Repository:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Docker Image:** [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   **How to use the Dockerfile:** [https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker](https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker)
*   **OpenHub:** [https://www.openhub.net/p/OWASP-Nettacker](https://www.openhub.net/p/OWASP-Nettacker)
*   **Donate:** [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More:** [https://www.secologist.com/open-source-projects](https://www.secologist.com/open-source-projects)

## Stargazers over time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)