# Destroylist: Your Shield Against Phishing and Scam Domains

Tired of phishing scams? **Destroylist provides real-time, comprehensive blacklists to protect you and your systems from online threats.** ([View the original repo](https://github.com/phishdestroy/destroylist))

## Key Features

*   **Comprehensive Blacklists:** Access multiple data feeds, including primary curated lists, DNS-verified active threats, and community-sourced blocklists.
*   **Real-time Updates:** Stay ahead of threats with frequently updated domain lists.
*   **Data Formats:** Utilizes JSON for easy integration with firewalls, DNS resolvers, browser extensions, and threat platforms.
*   **Community Driven:** Benefit from community contributions and open collaboration.
*   **Historical Archive:** Access a vast archive of over 500,000+ domains for research and analysis.

## Data Feeds

| Data Feed                 | Description                                     | Link                                                                     |
| ------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------ |
| Primary Curated List     | Core phishing/scam domains, real-time updates   | [list.json](https://github.com/phishdestroy/destroylist/raw/main/list.json) |
| Active DNS-Verified      | DNS-checked live threats                        | [active\_domains.json](https://github.com/phishdestroy/destroylist/raw/main/dns/active\_domains.json) |
| Community General        | Broad aggregated blocklist, hourly updates        | [blocklist.json](https://github.com/phishdestroy/destroylist/raw/main/community/blocklist.json) |
| Community Live           | DNS-checked active community threats            | [live\_blocklist.json](https://github.com/phishdestroy/destroylist/raw/main/community/live\_blocklist.json) |

## How It Works

Destroylist continuously gathers, syncs, and cleans data to provide up-to-date blacklists.  Domains are scanned, and complaints are filed with registrars to combat malicious activity.

## Use Cases

*   **Firewall Integration:** Enhance your network security with real-time threat intelligence.
*   **DNS Filtering:** Block malicious domains at the DNS level.
*   **Browser Extensions:** Protect users with browser-based phishing protection.
*   **Threat Research:** Analyze historical data and trends in phishing attacks.

## Appeals Process

If you believe a domain has been incorrectly listed:

*   Submit an [Appeals Form](https://phishdestroy.io/appeals/)
*   Open a GitHub Issue with supporting evidence.

## Get Involved

Contribute to the project by:

*   Suggesting detection improvements.
*   Sharing integration tips.
*   Providing fresh threat intelligence.

Open an issue or submit a pull request!