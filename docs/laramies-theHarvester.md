# theHarvester: Open Source Intelligence (OSINT) Gathering Tool

**Uncover a domain's digital footprint and assess its attack surface with theHarvester, a powerful OSINT tool for penetration testing and red team engagements.**

[Visit the Original Repository](https://github.com/laramies/theHarvester)

## Key Features

*   **Comprehensive Data Collection:** Gather names, emails, IPs, subdomains, and URLs from multiple public sources.
*   **Modular Design:** Leverages numerous passive and active modules for diverse data sources.
*   **User-Friendly:** Simple to use, making it easy to integrate into reconnaissance workflows.
*   **Supports Numerous Search Engines & APIs:** Integrates with popular OSINT resources.

## Installation

### Prerequisites

*   Python 3.12 or higher

### Installation Steps

1.  **Install `uv`:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/laramies/theHarvester
    cd theHarvester
    ```

3.  **Install Dependencies:**
    ```bash
    uv sync
    ```

4.  **Run the Tool:**
    ```bash
    uv run theHarvester
    ```

## Development

To set up your development environment:

1.  **Install Development Dependencies:**
    ```bash
    uv sync --extra dev
    ```

2.  **Run Tests:**
    ```bash
    uv run pytest
    ```

3.  **Lint and Format Code:**
    ```bash
    uv run ruff check
    uv run ruff format
    ```

## Passive Modules

theHarvester integrates with a wide range of passive modules, including:

*   baidu: Baidu search engine
*   bevigil: CloudSEK BeVigil
*   brave: Brave search engine
*   bufferoverun: Fast domain name lookups
*   builtwith: Website technology analysis
*   censys: Certificate search engine
*   certspotter: Certificate Transparency logs
*   criminalip: Cyber Threat Intelligence (CTI) search engine
*   crtsh: Comodo Certificate search
*   dehashed: Data breach search engine
*   dnsdumpster: Domain research tool
*   duckduckgo: DuckDuckGo search engine
*   fullhunt: Attack surface security platform
*   github-code: GitHub code search engine
*   hackertarget: Online vulnerability scanners
*   haveibeenpwned: Data breach checker
*   hunter: Hunter search engine
*   hunterhow: Search engine for security researchers
*   intelx: Intelx search engine
*   leaklookup: Data breach search engine
*   netlas: A Shodan or Censys competitor
*   onyphe: Cyber defense search engine
*   otx: AlienVault open threat exchange
*   pentesttools: Offensive security toolkit
*   projecdiscovery: Internet-wide assets data
*   rapiddns: DNS query tool
*   rocketreach: Contact information search
*   securityscorecard: Vendor risk assessment
*   securityTrails: Historical DNS data
*   -s, --shodan: Shodan search engine
*   subdomaincenter: Subdomain finder
*   subdomainfinderc99: Subdomain finder
*   threatminer: Threat intelligence data mining
*   tomba: Tomba search engine
*   urlscan: URL and website scanner
*   venacus: Venacus search engine
*   virustotal: Domain search
*   whoisxml: Subdomain search
*   yahoo: Yahoo search engine
*   zoomeye: China's version of Shodan

## Active Modules

*   DNS brute force: Dictionary-based subdomain enumeration
*   Screenshots: Capture screenshots of discovered subdomains

## Modules Requiring API Keys

Many modules require API keys for full functionality. Documentation for setting up API keys can be found [here](https://github.com/laramies/theHarvester/wiki/Installation#api-keys).

*   **bevigil** - 50 free queries/month, 1k queries/month $50
*   **brave** - Free plan available, Pro plans for higher limits
*   **bufferoverun** - 100 free queries/month, 10k/month $25
*   **builtwith** - 50 free queries ever, $2950/yr
*   **censys** - 500 credits $100
*   **criminalip** - 100 free queries/month, 700k/month $59
*   **dehashed** - 500 credts $15, 5k credits $150
*   **dnsdumpster** - 50 free querries/day, $49
*   **fullhunt** - 50 free queries, 200 queries $29/month, 500 queries $59/month
*   **github-code**
*   **haveibeenpwned** - 10 email searches/min $4.50, 50 email searches/min $22
*   **hunter** - 50 credits/month free, 12k credits/yr $34
*   **hunterhow** - 10k free API results per 30 days, 50k API results per 30 days $10
*   **intelx**
*   **leaklookup** - 20 credits $10, 50 credits $20, 140 credits $50, 300 credits $100
*   **netlas** - 50 free requests/day, 1k requests $49, 10k requests $249
*   **onyphe** - 10M results/month $587
*   **pentesttools** - 5 assets netsec $95/month, 5 assets webnetsec $140/month
*   **projecdiscovery** - requires work email. Free monthly discovery and vulnerability scans on sign-up email domain, enterprise $
*   **rocketreach** - 100 email lookups/month $48, 250 email lookups/month $108
*   **securityscorecard**
*   **securityTrails** - 50 free queries/month, 20k queries/month $500
*   **shodan** - Freelancer $69 month, Small Business $359 month
*   **tomba** - 25 searches/month free, 1k searches/month $39, 5k searches/month $89
*   **venacus** - 1 search/day free, 10 searches/day $12, 30 searches/day $36
*   **whoisxml** - 2k queries $50, 5k queries $105
*   **zoomeye** - 5 results/day free, 30/results/day $190/yr

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Contact

*   Christian Martorella @laramies ([@laramies](https://twitter.com/laramies)) cmartorella@edge-security.com
*   Matthew Brown @NotoriousRebel1 ([@NotoriousRebel1](https://twitter.com/NotoriousRebel1))
*   Jay "L1ghtn1ng" Townsend @jay_townsend1 ([@jay_townsend1](https://twitter.com/jay_townsend1))

## Main Contributors

*   Matthew Brown @NotoriousRebel1 ([@NotoriousRebel1](https://twitter.com/NotoriousRebel1))
*   Jay "L1ghtn1ng" Townsend @jay_townsend1 ([@jay_townsend1](https://twitter.com/jay_townsend1))
*   Lee Baird @discoverscripts ([@discoverscripts](https://twitter.com/discoverscripts))

## Acknowledgements

*   John Matherly - Shodan project
*   Ahmed Aboul Ela - subdomain names dictionaries (big and small)