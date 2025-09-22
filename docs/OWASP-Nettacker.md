# OWASP Nettacker: Your All-in-One Automated Penetration Testing Framework

**OWASP Nettacker** is a powerful open-source framework that streamlines information gathering and vulnerability assessment, empowering security professionals and ethical hackers to identify and mitigate risks effectively.

[View the original repository on GitHub](https://github.com/OWASP/Nettacker)

## Key Features

*   **Modular Architecture:** Execute specific tasks with modular design for fine-grained control.
*   **Multi-Protocol & Multithreaded Scanning:** Supports HTTP/HTTPS, FTP, SSH, SMB, SMTP, and more, with parallel scanning for speed.
*   **Comprehensive Output:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Built-in Database & Drift Detection:** Track and compare scan results to identify changes and vulnerabilities over time.
*   **CLI, REST API & Web UI:** Integrate Nettacker into your workflows via a command-line interface, REST API, or user-friendly web interface.
*   **Evasion Techniques:** Utilize configurable delays, proxy support, and randomized user-agents to bypass security measures.
*   **Flexible Targeting:** Scan single IPs, IP ranges, CIDR blocks, domain names, and URLs.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, and vulnerability assessments.
*   **Recon & Vulnerability Assessment:** Map hosts, ports, services, and directories, and test credentials.
*   **Attack Surface Mapping:** Quickly discover exposed hosts, ports, subdomains, and services.
*   **Bug Bounty Recon:** Automate common reconnaissance tasks to speed up finding targets.
*   **Network Vulnerability Scanning:** Efficiently scan large networks using a modular approach.
*   **Shadow IT & Asset Discovery:** Uncover forgotten assets and services using historical scan data.
*   **CI/CD & Compliance Monitoring:** Integrate Nettacker into pipelines to track infrastructure changes and detect vulnerabilities.

## Quick Start (Docker)

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
*   More details and install without docker [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)

## Disclaimer

*   ***This software is designed for ethical use. Always obtain permission before testing systems.***

## Community and Support

*   **OWASP Nettacker Project Home Page:** [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on https://owasp.slack.com
*   **Installation:** [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)
*   **Usage:** [https://nettacker.readthedocs.io/en/latest/Usage](https://nettacker.readthedocs.io/en/latest/Usage)
*   **GitHub:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Docker Image:** [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   **How to use the Dockerfile:** [https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker](https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker)
*   **OpenHub:** [https://www.openhub.net/p/OWASP-Nettacker](https://www.openhub.net/p/OWASP-Nettacker)
*   **Donate:** [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More:** [https://www.secologist.com/open-source-projects](https://www.secologist.com/open-source-projects)

## Adopters

Join the community of organizations and individuals who rely on OWASP Nettacker!  See [ADOPTERS.md](ADOPTERS.md) for details.

## Google Summer of Code (GSoC) Project

OWASP Nettacker has proudly participated in the Google Summer of Code initiative. Thanks to Google and all the students who contributed.