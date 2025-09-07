# OWASP Nettacker: Automated Penetration Testing and Information Gathering

**OWASP Nettacker is a powerful open-source framework for automating penetration testing and reconnaissance tasks, helping security professionals identify vulnerabilities and strengthen their defenses.**

[<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200">](https://github.com/OWASP/Nettacker) [<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">](https://github.com/OWASP/Nettacker)

**[View the original repository on GitHub](https://github.com/OWASP/Nettacker)**

**DISCLAIMER:** This software is intended for ethical and responsible use only.  Use Nettacker for authorized penetration testing and information gathering purposes.  The contributors are not responsible for any illegal activities conducted with this tool.

## Key Features

*   **Modular Architecture:** Execute specific tasks (port scanning, directory discovery, vulnerability checks, brute-forcing) with individual modules for granular control.
*   **Multi-Protocol & Multithreaded Scanning:** Supports HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, XML-RPC, and parallel scanning for speed.
*   **Comprehensive Reporting:** Generate reports in HTML, JSON, CSV, and plain text formats for easy analysis.
*   **Built-in Database & Drift Detection:**  Store scan data and compare results to identify new hosts, open ports, or vulnerabilities over time.
*   **CLI, REST API & Web UI:**  Offers flexibility with command-line, REST API, and user-friendly web interface access.
*   **Evasion Techniques:** Employ configurable delays, proxy support, and randomized user-agents to avoid detection.
*   **Flexible Target Input:** Supports single IPs, IP ranges, CIDR blocks, domain names, and URLs, with options for list-based input.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, service discovery, and vulnerability assessments to streamline testing workflows.
*   **Recon & Vulnerability Assessment:** Discover live hosts, open ports, services, and directories, and perform credential brute-forcing.
*   **Attack Surface Mapping:** Quickly identify exposed hosts, ports, subdomains, and services for comprehensive asset discovery.
*   **Bug Bounty Recon:** Automate and scale reconnaissance tasks for bug bounty hunting.
*   **Network Vulnerability Scanning:** Efficiently scan IPs, IP ranges, CIDR blocks, and subdomains in parallel for large-scale assessments.
*   **Shadow IT & Asset Discovery:** Uncover forgotten hosts, open ports, and subdomains using historical scan data.
*   **CI/CD & Compliance Monitoring:** Integrate Nettacker into CI/CD pipelines to track infrastructure changes and detect new vulnerabilities.

## Quick Start with Docker

### CLI

```bash
# Basic port scan on a single IP address:
docker run owasp/nettacker -i 192.168.0.1 -m port_scan
# Scan the entire Class C network for any devices with port 22 open:
docker run owasp/nettacker -i 192.168.0.0/24 -m port_scan -g 22
# Scan all subdomains of 'owasp.org' for http/https services and return HTTP status code
docker run owasp/nettacker -i owasp.org -d -s -m http_status_scan
# Display Help
docker run owasp/nettacker --help
```

### Web UI

1.  `docker-compose up`
2.  Use the API Key displayed in the CLI to login to the Web GUI.
3.  Access the Web GUI at `https://localhost:5000` or `https://nettacker-api.z3r0d4y.com:5000/` (pointed to your localhost).
4.  The local database is `.nettacker/data/nettacker.db` (sqlite).
5.  Default results path is `.nettacker/data/results`.
6.  `docker-compose` will share your nettacker folder, so you will not lose any data after `docker-compose down`
7.  To see the API key in you can also run `docker logs nettacker_nettacker`.
8.  More details and install without docker https://nettacker.readthedocs.io/en/latest/Installation

## Community and Resources

*   **OWASP Nettacker Project Home Page:** [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on https://owasp.slack.com
*   **Installation:** [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)
*   **Usage:** [https://nettacker.readthedocs.io/en/latest/Usage](https://nettacker.readthedocs.io/en/latest/Usage)
*   **GitHub Repository:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Docker Image:** [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   **Dockerfile Usage:** [https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker](https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker)
*   **OpenHub:** [https://www.openhub.net/p/OWASP-Nettacker](https://www.openhub.net/p/OWASP-Nettacker)
*   **Donate:** [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More:** [https://www.secologist.com/open-source-projects](https://www.secologist.com/open-source-projects)

## Contributors

OWASP Nettacker thrives on community contributions. A big thank you to all our contributors!

![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)

## Adopters

We appreciate all the organizations and individuals who use OWASP Nettacker!  Add your details to the [ADOPTERS.md](ADOPTERS.md) file to be listed.

## Google Summer of Code

The OWASP Nettacker project participates in the Google Summer of Code initiative.

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers Over Time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)