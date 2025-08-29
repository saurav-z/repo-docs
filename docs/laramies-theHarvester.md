[![theHarvester](https://github.com/laramies/theHarvester/blob/master/theHarvester-logo.webp)](https://github.com/laramies/theHarvester)

[![TheHarvester CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Python%20CI/badge.svg)]
[![TheHarvester Docker Image CI](https://github.com/laramies/theHarvester/workflows/TheHarvester%20Docker%20Image%20CI/badge.svg)]
[![Rawsec's CyberSecurity Inventory](https://inventory.raw.pm/img/badges/Rawsec-inventoried-FF5050_flat_without_logo.svg)](https://inventory.raw.pm/)

# theHarvester: OSINT Reconnaissance Tool

**theHarvester is a powerful open-source intelligence (OSINT) gathering tool designed for penetration testers and security researchers to uncover a target's attack surface.** ([See the original repository](https://github.com/laramies/theHarvester))

## Key Features

*   **Comprehensive Information Gathering:** Collects valuable information including:
    *   Subdomains
    *   Emails
    *   IP addresses
    *   URLs
    *   Employee names
*   **Multiple Data Sources:** Leverages various public resources for data collection:
    *   Search engines (Google, DuckDuckGo, etc.)
    *   Social media platforms
    *   API integrations
    *   And more!
*   **Passive & Active Modules:** Includes passive modules to gather information without direct interaction with the target and active modules for DNS brute-forcing and screenshot capturing.
*   **API Key Support:** Seamless integration with services requiring API keys for enhanced data gathering.

## Installation and Setup

### Prerequisites
* Python 3.12 or higher.

### Installation Steps:
1.  **Install `uv`:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/laramies/theHarvester
    cd theHarvester
    ```
3.  **Install Dependencies (using uv):**
    ```bash
    uv sync
    ```
4.  **Run theHarvester:**
    ```bash
    uv run theHarvester
    ```

## Development

To contribute to the project:

### Install Development Dependencies
```bash
uv sync --extra dev
```

### Run Tests:
```bash
uv run pytest
```

### Run Linting and Formatting:
```bash
uv run ruff check
uv run ruff format
```

## Modules
### Passive Modules:
*   [Baidu](https://www.baidu.com)
*   [Bevilgil](https://bevigil.com/osint-api)
*   [Brave](https://api-dashboard.search.brave.com)
*   [Bufferoverun](https://tls.bufferover.run)
*   [Builtwith](https://builtwith.com)
*   [Censys](https://censys.io)
*   [Certspotter](https://sslmate.com/certspotter)
*   [Criminalip](https://www.criminalip.io)
*   [Crtsh](https://crt.sh)
*   [Dehashed](https://dehashed.com)
*   [Dnsdumpster](https://dnsdumpster.com)
*   [Duckduckgo](https://duckduckgo.com)
*   [Fullhunt](https://fullhunt.io)
*   [Github-code](https://www.github.com)
*   [Hackertarget](https://hackertarget.com)
*   [Haveibeenpwned](https://haveibeenpwned.com)
*   [Hunter](https://hunter.io)
*   [Hunterhow](https://hunter.how)
*   [Intelx](https://intelx.io)
*   [Leaklookup](https://leak-lookup.com)
*   [Netlas](https://app.netlas.io)
*   [Onyhe](https://www.onyphe.io)
*   [Otx](https://otx.alienvault.com)
*   [Pentesttools](https://pentest-tools.com)
*   [Projectdiscovery](https://chaos.projectdiscovery.io)
*   [Rapiddns](https://rapiddns.io)
*   [Rocketreach](https://rocketreach.co)
*   [Securityscorecard](https://securityscorecard.com)
*   [Securitytrails](https://securitytrails.com)
*   [Shodan](https://shodan.io)
*   [Subdomaincenter](https://www.subdomain.center)
*   [Subdomainfinderc99](https://subdomainfinder.c99.nl)
*   [Threatminer](https://www.threatminer.org)
*   [Tomba](https://tomba.io)
*   [Urlscan](https://urlscan.io)
*   [Venacus](https://venacus.com)
*   [Virustotal](https://www.virustotal.com)
*   [Whoisxml](https://subdomains.whoisxmlapi.com/api/pricing)
*   [Yahoo](https://www.yahoo.com)
*   [Zoomeye](https://www.zoomeye.org)

### Active Modules
*   DNS Brute Force
*   Screenshots

### Modules that require an API Key:
*   API key setup documentation: [API Key Setup](https://github.com/laramies/theHarvester/wiki/Installation#api-keys)

## API Key Pricing:
*   **BeVigil:** 50 free queries/month, 1k queries/month $50
*   **Brave:** Free plan available, Pro plans for higher limits
*   **Bufferoverun:** 100 free queries/month, 10k/month $25
*   **Builtwith:** 50 free queries ever, $2950/yr
*   **Censys:** 500 credits $100
*   **Criminalip:** 100 free queries/month, 700k/month $59
*   **Dehashed:** 500 credts $15, 5k credits $150
*   **Dnsdumpster:** 50 free querries/day, $49
*   **Fullhunt:** 50 free queries, 200 queries $29/month, 500 queries $59/month
*   **Github-code**
*   **Haveibeenpwned:** 10 email searches/min $4.50, 50 email searches/min $22
*   **Hunter:** 50 credits/month free, 12k credits/yr $34
*   **Hunterhow:** 10k free API results per 30 days, 50k API results per 30 days $10
*   **Intelx**
*   **Leaklookup:** 20 credits $10, 50 credits $20, 140 credits $50, 300 credits $100
*   **Netlas:** 50 free requests/day, 1k requests $49, 10k requests $249
*   **Onyhe:** 10M results/month $587
*   **Pentesttools:** 5 assets netsec $95/month, 5 assets webnetsec $140/month
*   **Projectdiscovery:** requires work email. Free monthly discovery and vulnerability scans on sign-up email domain, enterprise $
*   **Rocketreach:** 100 email lookups/month $48, 250 email lookups/month $108
*   **Securityscorecard**
*   **SecurityTrails:** 50 free queries/month, 20k queries/month $500
*   **Shodan:** Freelancer $69 month, Small Business $359 month
*   **Tomba:** 25 searches/month free, 1k searches/month $39, 5k searches/month $89
*   **Venacus:** 1 search/day free, 10 searches/day $12, 30 searches/day $36
*   **Whoisxml:** 2k queries $50, 5k queries $105
*   **Zoomeye:** 5 results/day free, 30/results/day $190/yr

## Get Involved
*   **Report Issues and Suggest Improvements:**
    *   [Twitter](https://twitter.com/laramies): Christian Martorella @laramies
    *   [cmartorella@edge-security.com](cmartorella@edge-security.com)
    *   [Twitter](https://twitter.com/NotoriousRebel1): Matthew Brown @NotoriousRebel1
    *   [Twitter](https://twitter.com/jay_townsend1): Jay "L1ghtn1ng" Townsend @jay_townsend1
*   **Main Contributors:**
    *   [Twitter](https://twitter.com/NotoriousRebel1) Matthew Brown @NotoriousRebel1
    *   [Twitter](https://twitter.com/jay_townsend1) Jay "L1ghtn1ng" Townsend @jay_townsend1
    *   [Twitter](https://twitter.com/discoverscripts) Lee Baird @discoverscripts

## Acknowledgements
*   John Matherly - Shodan project
*   Ahmed Aboul Ela - subdomain names dictionaries (big and small)