# theHarvester: Your OSINT Reconnaissance Toolkit

**Uncover a domain's digital footprint and identify potential attack vectors with theHarvester, a powerful open-source intelligence (OSINT) gathering tool.**

[![theHarvester](https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp)](https://github.com/laramies/theHarvester)
[![TheHarvester CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg)](https://github.com/laramies/theHarvester/actions/workflows/python-app.yml)
[![TheHarvester Docker Image CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg)](https://github.com/laramies/theHarvester/actions/workflows/docker-image.yml)
[![Rawsec's CyberSecurity Inventory](https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg)](https://inventory.raw.pm/)

theHarvester is a versatile tool for penetration testers and red teamers, designed to automate the reconnaissance phase by collecting valuable information from various public sources. Use theHarvester to discover potential vulnerabilities and weaknesses in your target's online presence.

**Key Features:**

*   **Automated OSINT Gathering:** Collects emails, subdomains, IPs, URLs, and more from a wide range of public sources.
*   **Modular Design:** Supports numerous passive and active modules for comprehensive data collection.
*   **Easy to Use:** Simple command-line interface for quick and efficient reconnaissance.
*   **API Integration:** Integrates with various search engines and services, including Shodan, Censys, and more.
*   **Versatile Reporting:** Provides structured output for easy analysis and reporting.

**Getting Started**

1.  **Installation**
    *   Requires Python 3.12 or higher.
    *   Refer to the [Installation Guide](https://github.com/laramies/theHarvester/wiki/Installation) for detailed instructions.

    **Quick Installation using `uv` (recommended):**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    git clone https://github.com/laramies/theHarvester
    cd theHarvester
    uv sync
    uv run theHarvester
    ```

2.  **Development Setup**

    To install development dependencies:

    ```bash
    uv sync --extra dev
    ```

    Run tests:
    ```bash
    uv run pytest
    ```

    Run linting and formatting:
    ```bash
    uv run ruff check
    ```

    ```bash
    uv run ruff format
    ```

**Modules**

*   **Passive Modules:**

    *   Baidu, BeVigil, Brave, Bufferoverun, Builtwith, Censys, Certspotter, Criminalip, Crtsh, Dehashed, Dnsdumpster, Duckduckgo, Fullhunt, Github-code, Hackertarget, Haveibeenpwned, Hunter, Hunterhow, Intelx, Leaklookup, Netlas, Onytpe, Otx, Pentesttools, Projectdiscovery, Rapiddns, Rocketreach, Securityscorecard, SecurityTrails, Shodan, Subdomaincenter, Subdomainfinderc99, Threatminer, Tomba, Urlscan, Venacus, Virustotal, Whoisxml, Yahoo, Zoomeye

*   **Active Modules:**

    *   DNS brute force, Screenshots

**Modules Requiring API Keys**

See the [API Keys Documentation](https://github.com/laramies/theHarvester/wiki/Installation#api-keys) for setup.

**Contributing**

We welcome contributions! Feel free to submit bug reports, feature requests, or pull requests.

**Get in Touch**

*   [Twitter - Christian Martorella](https://twitter.com/laramies)
*   [Twitter - Matthew Brown](https://twitter.com/NotoriousRebel1)
*   [Twitter - Jay Townsend](https://twitter.com/jay_townsend1)

**License**
*   [MIT License](https://github.com/laramies/theHarvester/blob/master/LICENSE)

**[Back to the GitHub Repository](https://github.com/laramies/theHarvester)**