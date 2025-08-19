# theHarvester: Powerful OSINT Tool for Reconnaissance

**Uncover a domain's attack surface and gather valuable intelligence with theHarvester, a versatile open-source intelligence (OSINT) gathering tool.**  [Check out the original repository](https://github.com/laramies/theHarvester)

![theHarvester](https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp)

[![TheHarvester CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg)]
[![TheHarvester Docker Image CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg)]
[![Rawsec's CyberSecurity Inventory](https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg)](https://inventory.raw.pm/)

## Key Features

*   **Automated OSINT Gathering:** Quickly collect information from multiple public sources.
*   **Comprehensive Data Collection:**  Find names, emails, IPs, subdomains, and URLs.
*   **Modular Design:** Leverages numerous passive and active modules for diverse data collection.
*   **Easy to Use:** Simple command-line interface for efficient reconnaissance.
*   **Essential for Security Assessments:**  A vital tool for red team engagements and penetration testing.

## Installation

**Prerequisites:**

*   Python 3.12 or higher

**Installation Steps:**

1.  **Install `uv` (optional, recommended for dependency management):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/laramies/theHarvester
    cd theHarvester
    ```
3.  **Install dependencies and create a virtual environment:**
    ```bash
    uv sync
    ```

4.  **Run theHarvester:**
    ```bash
    uv run theHarvester
    ```

## Development

To install development dependencies:
```bash
uv sync --extra dev
```

To run tests:
```bash
uv run pytest
```

To run linting and formatting:
```bash
uv run ruff check
```
```bash
uv run ruff format
```
## Modules

### Passive Modules

theHarvester utilizes a wide array of passive modules to gather intelligence without direct interaction with the target.  These modules query various search engines, APIs, and data sources to uncover valuable information.

*   baidu
*   bevigil
*   brave
*   bufferoverun
*   builtwith
*   censys
*   certspotter
*   criminalip
*   crtsh
*   dehashed
*   dnsdumpster
*   duckduckgo
*   fullhunt
*   github-code
*   hackertarget
*   haveibeenpwned
*   hunter
*   hunterhow
*   intelx
*   leaklookup
*   netlas
*   onyphe
*   otx
*   pentesttools
*   projecdiscovery
*   rapiddns
*   rocketreach
*   securityscorecard
*   securityTrails
*   -s, --shodan
*   subdomaincenter
*   subdomainfinderc99
*   threatminer
*   tomba
*   urlscan
*   venacus
*   virustotal
*   whoisxml
*   yahoo
*   zoomeye

### Active Modules

theHarvester also includes active modules for probing and interacting with the target environment.

*   DNS brute force
*   Screenshots

### Modules Requiring API Keys

Some modules require API keys for access.  Refer to the [Installation Wiki](https://github.com/laramies/theHarvester/wiki/Installation#api-keys) for detailed setup instructions.

* bevigil
* brave
* bufferoverun
* builtwith
* censys
* criminalip
* dehashed
* dnsdumpster
* fullhunt
* github-code
* haveibeenpwned
* hunter
* hunterhow
* intelx
* leaklookup
* netlas
* onyphe
* pentesttools
* projecdiscovery
* rocketreach
* securityscorecard
* securityTrails
* shodan
* tomba
* venacus
* whoisxml
* zoomeye


## Get Involved

*   **Follow:** [@laramies](https://twitter.com/laramies) (Christian Martorella) and [@NotoriousRebel1](https://twitter.com/NotoriousRebel1)  and [@jay_townsend1](https://twitter.com/jay_townsend1) and [@discoverscripts](https://twitter.com/discoverscripts)
*   **Contact:** cmartorella@edge-security.com

## Main Contributors

*   [@NotoriousRebel1](https://twitter.com/NotoriousRebel1) Matthew Brown
*   [@jay_townsend1](https://twitter.com/jay_townsend1) Jay "L1ghtn1ng" Townsend
*   [@discoverscripts](https://twitter.com/discoverscripts) Lee Baird

## Thanks

*   John Matherly - Shodan project
*   Ahmed Aboul Ela - subdomain names dictionaries (big and small)