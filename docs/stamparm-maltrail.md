![Maltrail](https://i.imgur.com/3xjInOD.png)

[![Python 2.6|2.7|3.x](https://img.shields.io/badge/python-2.6|2.7|3.x-yellow.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/stamparm/maltrail#license) [![Malware families](https://img.shields.io/badge/malware_families-1494-orange.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Malware sinkholes](https://img.shields.io/badge/malware_sinkholes-1354-green.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Twitter](https://img.shields.io/badge/twitter-@maltrail-blue.svg)](https://twitter.com/maltrail)

## Maltrail: Detect and Mitigate Malicious Network Traffic with Ease

Maltrail is a powerful, open-source malicious traffic detection system designed to identify and alert you to potential threats within your network.  [Check out the original repository](https://github.com/stamparm/maltrail) for the latest updates and features.

**Key Features:**

*   **Real-time Traffic Monitoring:**  Passively monitors network traffic for suspicious activity.
*   **Extensive Threat Intelligence:** Utilizes a wide range of public and static blacklists, as well as heuristic analysis.
*   **Comprehensive Trail Database:** Includes domains, URLs, IP addresses, and HTTP User-Agent headers associated with malware.
*   **Flexible Architecture:** Supports a sensor-server-client architecture for distributed deployment or standalone sensor mode.
*   **User-Friendly Reporting Interface:** Provides a web-based interface for easy analysis and threat investigation.
*   **Heuristic Analysis:** Detects unknown threats and emerging malware through advanced analysis.
*   **Customizable:** Allows users to define their own trails and configure various settings.
*   **Easy Integration:** Supports integration with various third-party tools and platforms.

**Table of Contents**

*   [Introduction](#introduction)
*   [Architecture](#architecture)
*   [Demo Pages](#demo-pages)
*   [Requirements](#requirements)
*   [Quick Start](#quick-start)
*   [Administrator's Guide](#administrators-guide)
    *   [Sensor](#sensor)
    *   [Server](#server)
*   [User's Guide](#users-guide)
    *   [Reporting Interface](#reporting-interface)
*   [Real-life Cases](#real-life-cases)
    *   [Mass Scans](#mass-scans)
    *   [Anonymous Attackers](#anonymous-attackers)
    *   [Service Attackers](#service-attackers)
    *   [Malware](#malware)
    *   [Suspicious Domain Lookups](#suspicious-domain-lookups)
    *   [Suspicious Ipinfo Requests](#suspicious-ipinfo-requests)
    *   [Suspicious Direct File Downloads](#suspicious-direct-file-downloads)
    *   [Suspicious HTTP Requests](#suspicious-http-requests)
    *   [Port Scanning](#port-scanning)
    *   [DNS Resource Exhaustion](#dns-resource-exhaustion)
    *   [Data Leakage](#data-leakage)
    *   [False Positives](#false-positives)
*   [Best Practice(s)](#best-practices)
*   [License](#license)
*   [Sponsors](#sponsors)
*   [Developers](#developers)
*   [Presentations](#presentations)
*   [Publications](#publications)
*   [Blacklist](#blacklist)
*   [Thank You](#thank-you)
*   [Third-Party Integrations](#third-party-integrations)

## Introduction

Maltrail is a malicious traffic detection system designed to identify and alert you to potential threats within your network. It uses publicly available blacklists containing malicious or generally suspicious trails and static trails compiled from AV reports. A trail can be anything from a domain name (e.g., `zvpprsensinaix.com`) to a URL (e.g., `hXXp://109.162.38.120/harsh02.exe`), IP address (e.g., `185.130.5.231`), or HTTP User-Agent header value (e.g., `sqlmap`).

**Key Features of Maltrail Include:**

*   **Utilizes Blacklists:** Leverages public blacklists for detection.
*   **Static Trails:** Incorporates trails compiled from various AV reports.
*   **Heuristic Mechanisms:** Employs heuristic mechanisms for the discovery of unknown threats.

The following lists are utilized:
```
360bigviktor, 360chinad, 360conficker, 360cryptolocker, 360gameover, 
360locky, 360necurs, 360suppobox, 360tofsee, 360virut, abuseipdb, alienvault, 
atmos, badips, bitcoinnodes, blackbook, blocklist, botscout, 
bruteforceblocker, ciarmy, cobaltstrike, cruzit, cybercrimetracker, 
dataplane, dshieldip, emergingthreatsbot, emergingthreatscip, 
emergingthreatsdns, feodotrackerip, gpfcomics, greensnow, ipnoise,
kriskinteldns, kriskintelip, malc0de, malwaredomainlistdns, malwaredomains,
maxmind, minerchk, myip, openphish, palevotracker, policeman, pony,
proxylists, proxyrss, proxyspy, ransomwaretrackerdns, ransomwaretrackerip, 
ransomwaretrackerurl, riproxies, rutgers, sblam, socksproxy, sslbl, 
sslproxies, talosintelligence, torproject, trickbot, turris, urlhaus, 
viriback, vxvault, zeustrackermonitor, zeustrackerurl, etc.
```
Static entries manually included from reports:
```
1ms0rry, 404, 9002, aboc, absent, ab, acbackdoor, acridrain, activeagent, 
adrozek, advisorbot, adwind, adylkuzz, adzok, afrodita, agaadex, agenttesla, 
aldibot, alina, allakore, almalocker, almashreq, alpha, alureon, amadey, 
amavaldo, amend_miner, ammyyrat, android_acecard, android_actionspy, 
android_adrd, android_ahmythrat, android_alienspy, android_andichap, 
android_androrat, android_anubis, android_arspam, android_asacub, 
android_backflash, android_bankbot, android_bankun, android_basbanke, 
android_basebridge, android_besyria, android_blackrock, android_boxer, 
android_buhsam, android_busygasper, android_calibar, android_callerspy, 
android_camscanner, android_cerberus, android_chuli, android_circle, 
android_claco, android_clickfraud, android_cometbot, android_cookiethief, 
android_coolreaper, android_copycat, android_counterclank, android_cyberwurx, 
android_darkshades, android_dendoroid, android_dougalek, android_droidjack, 
android_droidkungfu, android_enesoluty, android_eventbot, android_ewalls, 
android_ewind, android_exodus, android_exprespam, android_fakeapp, 
android_fakebanco, android_fakedown, android_fakeinst, android_fakelog, 
android_fakemart, android_fakemrat, android_fakeneflic, android_fakesecsuit, 
android_fanta, android_feabme, android_flexispy, android_fobus, 
android_fraudbot, android_friend, android_frogonal, android_funkybot, 
android_gabas, android_geinimi, android_generic, android_geost, 
android_ghostpush, android_ginmaster, android_ginp, android_gmaster, 
android_gnews, android_godwon, android_golddream, android_goldencup, 
android_golfspy, android_gonesixty, android_goontact, android_gplayed, 
android_gustuff, android_gypte, android_henbox, android_hiddad, 
android_hydra, android_ibanking, android_joker, android_jsmshider, 
android_kbuster, android_kemoge, android_ligarat, android_lockdroid, 
android_lotoor, android_lovetrap, android_malbus, android_mandrake, 
android_maxit, android_mobok, android_mobstspy, android_monokle, 
android_notcompatible, android_oneclickfraud, android_opfake, 
android_ozotshielder, android_parcel, android_phonespy, android_pikspam, 
android_pjapps, android_qdplugin, android_raddex, android_ransomware, 
android_redalert, android_regon, android_remotecode, android_repane, 
android_riltok, android_roamingmantis, android_roidsec, android_rotexy, 
android_samsapo, android_sandrorat, android_selfmite, android_shadowvoice, 
android_shopper, android_simbad, android_simplocker, android_skullkey, 
android_sndapps, android_spynote, android_spytekcell, android_stels, 
android_svpeng, android_swanalitics, android_teelog, android_telerat, 
android_tetus, android_thiefbot, android_tonclank, android_torec, 
android_triada, android_uracto, android_usbcleaver, android_viceleaker, 
android_vmvol, android_walkinwat, android_windseeker, android_wirex, 
android_wolfrat, android_xavirad, android_xbot007, android_xerxes, 
android_xhelper, android_xploitspy, android_z3core, android_zertsecurity, 
android_ztorg, andromeda, antefrigus, antibot, anubis, anuna, apocalypse, 
apt_12, apt_17, apt_18, apt_23, apt_27, apt_30, apt_33, apt_37, apt_38, 
apt_aridviper, apt_babar, apt_bahamut, etc.
```
![Reporting tool](https://i.imgur.com/Sd9eqoa.png)

## Architecture

Maltrail utilizes a sensor-server-client architecture:

*   **Sensor:** The sensor is a standalone component running on the monitoring node (e.g., Linux platform connected passively to the SPAN/mirroring port or transparently inline on a Linux bridge) or at the standalone machine (e.g. Honeypot) where it "monitors" the passing **Traffic** for blacklisted items/trails (i.e. domain names, URLs and/or IPs). It monitors the traffic for blacklisted items like domain names, URLs, and/or IPs. When a match is found, the event details are sent to the server.
*   **Server:** Stores event details and provides a back-end for the reporting web application. By default, the server and sensor run on the same machine. The server stores the events and provides a reporting web application.
*   **Client:** A web-based reporting interface for viewing and analyzing detected threats. The front-end reporting part is based on the ["Fat client"](https://en.wikipedia.org/wiki/Fat_client) architecture (i.e. all data post-processing is being done inside the client's web browser instance).

Note: **Server** component can be skipped altogether, and just use the standalone **Sensor**. In such case, all events would be stored in the local logging directory, while the log entries could be examined either manually or by some CSV reading application.

![Architecture diagram](https://i.imgur.com/2IP9Mh2.png)

## Demo pages

You can explore fully functional demo pages with real-life threat examples [here](https://maltraildemo.github.io/).

## Requirements

*   Python 2.6, 2.7, or 3.x on \*nix/BSD systems
*   pcapy-ng package
    *   **NOTE:** Use of ```pcapy``` lib instead of ```pcapy-ng``` can lead to incorrect work of Maltrail, especially on **Python 3.x** environments. [Examples](https://github.com/stamparm/maltrail/issues?q=label%3Apcapy-ng-related+is%3Aclosed).
*   **Sensor:** At least 1GB of RAM is recommended to run in single-process mode. More RAM is needed if multiprocessing mode is enabled, depending on the value of the `CAPTURE_BUFFER` option. Sensor component (in the general case) requires administrative/root privileges.
*   **Server:** No specific requirements

## Quick Start

These commands will get your Maltrail **Sensor** up and running with default settings:

*   **Ubuntu/Debian:**
```bash
sudo apt-get install git python3 python3-dev python3-pip python-is-python3 libpcap-dev build-essential procps schedtool
sudo pip3 install pcapy-ng
git clone --depth 1 https://github.com/stamparm/maltrail.git
cd maltrail
sudo python3 sensor.py
```

*   **SUSE/openSUSE:**
```bash
sudo zypper install gcc gcc-c++ git libpcap-devel python3-devel python3-pip procps schedtool
sudo pip3 install pcapy-ng
git clone --depth 1 https://github.com/stamparm/maltrail.git
cd maltrail
sudo python3 sensor.py
```

*   Ensure interfaces are in promiscuous mode:
```bash
for dev in $(ifconfig | grep mtu | grep -Eo '^\w+'); do ifconfig $dev promisc; done
```

![Sensor](https://i.imgur.com/E9tt2ek.png)

*   **Optional: Start the Server:** Open a new terminal and run:
```bash
[[ -d maltrail ]] || git clone --depth 1 https://github.com/stamparm/maltrail.git
cd maltrail
python server.py
```
![Server](https://i.imgur.com/loGW6GA.png)

*   **Docker:**
    *   See instructions to build image, start the server, and test from original README.

*   **Test:**
```bash
ping -c 1 136.161.101.53
cat /var/log/maltrail/$(date +"%Y-%m-%d").log
```
![Test](https://i.imgur.com/NYJg6Kl.png)

*   **Test DNS Capturing:**
```bash
nslookup morphed.ru
cat /var/log/maltrail/$(date +"%Y-%m-%d").log
```
![Test2](https://i.imgur.com/62oafEe.png)

*   **Stop Sensor/Server:**
```bash
sudo pkill -f sensor.py
pkill -f server.py
```

*   **Access Reporting Interface:** Open your web browser and go to http://127.0.0.1:8338 (default credentials: `admin:changeme!`).
![Reporting interface](https://i.imgur.com/VAsq8cs.png)

## Administrator's Guide

### Sensor
Sensor configuration is found in `maltrail.conf` under the `[Sensor]` section.  
*   **USE_MULTIPROCESSING:** set to `true` will use all CPU cores.
*   **USE_FEED_UPDATES:** Turns off trail updates from feeds.
*   **UPDATE_PERIOD:** Sets the number of seconds between each automatic trails update.
*   **CUSTOM_TRAILS_DIR:** Allows users to provide a custom trails directory.
*   **USE_HEURISTICS:** Turns on heuristic mechanisms (e.g., long domain names, etc.).
*   **CAPTURE_BUFFER:** Memory used in multiprocessing mode.
*   **MONITOR_INTERFACE:** The interface to capture packets from. `any` captures from all interfaces.
*   **CAPTURE_FILTER:** The `tcpdump` filter.
*   **SENSOR_NAME:** Name in the `sensor_name` event value.
*   **LOG_SERVER:** If set, events are sent remotely to the server.
*   **UPDATE_SERVER:** Trails are pulled from the given location.
*   **SYSLOG_SERVER/LOGSTASH_SERVER:**  Send sensor events to other servers.

Example event data sent over UDP:

*   **SYSLOG_SERVER:**
```Dec 24 15:05:55 beast CEF:0|Maltrail|sensor|0.27.68|2020-12-24|andromeda (malware)|2|src=192.168.5.137 spt=60453 dst=8.8.8.8 dpt=53 trail=morphed.ru ref=(static)```

*   **LOGSTASH_SERVER:**
```json
{"timestamp": 1608818692, "sensor": "beast", "severity": "high", "src_ip": "192.168.5.137", "src_port": 48949, "dst_ip": "8.8.8.8", "dst_port": 53, "proto": "UDP", "type": "DNS", "trail": "morphed.ru", "info": "andromeda (malware)", "reference": "(static)"}
```

When sensor is running for the first time and/or after a longer period of non-running, it will automatically update the trails from trail definitions.
Detected events are stored inside the **Server**'s logging directory (i.e. option `LOG_DIR` inside the `maltrail.conf` file's section `[All]`) in easy-to-read CSV format (Note: whitespace ' ' is used as a delimiter) as single line entries consisting of: `time` `sensor` `src_ip` `src_port` `dst_ip` `dst_port` `proto` `trail_type` `trail` `trail_info` `reference` (e.g. `"2015-10-19 15:48:41.152513" beast 192.168.5.33 32985 8.8.8.8 53 UDP DNS 0000mps.webpreview.dsl.net malicious siteinspector.comodo.com`):

![Sample log](https://i.imgur.com/RycgVru.png)

### Server

Server configuration is found in `maltrail.conf` under the `[Server]` section.  
*   **HTTP_ADDRESS/HTTP_PORT:** Web server address and port.  Use `0.0.0.0` to listen on all interfaces.
*   **USE_SSL:** Enables `SSL/TLS`.
*   **SSL_PEM:** Path to the server's private/cert PEM file.
*   **USERS:** User settings including username, password (SHA256), UID, and filter netmask(s).
*   **UDP_ADDRESS/UDP_PORT:** Server's log collection listening address and port.
*   **FAIL2BAN_REGEX:** Regular expression used for `/fail2ban` web calls, for extraction of today's attacker source IPs.
*   **BLACKLIST:** Allows to build regular expressions to apply on one field. For each rule, the syntax is : `<field> <control> <regexp>` where :
    *   `field` indicates the field to compare, it can be: `src_ip`,`src_port`,`dst_ip`,`dst_port`,`protocol`,`type`,`trail` or `filter`.
    *   `control` can be either `~` for *matches* or `!~` for *doesn't match*
    *   `regexp` is the regular expression to apply to the field.
    Chain another rule with the `and` keyword (the `or` keyword is not supported, just add a line for this).
    You can use the keyword `BLACKLIST` alone or add a name : `BLACKLIST_NAME`. In the latter case, the url will be : `/blacklist/name`

Example
```
BLACKLIST_OUT
    src_ip !~ ^192.168. and dst_port ~ ^22$
    src_ip !~ ^192.168. and filter ~ scan
    src_ip !~ ^192.168. and filter ~ known attacker

BLACKLIST_IN
    src_ip ~ ^192.168. and filter ~ malware
```

## User's Guide

### Reporting Interface

The reporting interface provides a web-based view of the detected threats. After authentication (default is `admin:changeme!`), you'll see the reporting interface.
*   **Timeline:** Allows you to select log events from the past.
*   **Summary:** Presents a summary of events through charts and graphs.
*   **Detailed Table:** A paginated table displaying threat details such as source/destination IPs, ports, protocol, trail, and more.

![Reporting interface](https://i.imgur.com/PZY8JEC.png)

*   **Tooltip:** Hovering over `src_ip` and `dst_ip` shows reverse DNS and WHOIS information.

![On mouse over IP](https://i.imgur.com/BgKchAX.png)

*   **Bubble Icon:** Ellipsis icons represent details when event details differ.

![On mouse over bubble](https://i.imgur.com/BfYT2u7.png)

*   **Trail Search:** Hovering the mouse over the threat's trail triggers a search against searX.

![On mouse over trail](https://i.imgur.com/ZxnHn1N.png)

*   **Tags:**  Add tags to threats for better organization.

![Tags](https://i.imgur.com/u5Z4752.png)

### Real-life Cases

These are some real-world scenarios the tool can identify:

*   **Mass Scans:** Detects widespread network scanning.
*   **Anonymous Attackers:** Identifies attackers using Tor.
*   **Service Attackers:** Detects attacks against specific services.
*   **Malware:** Detects connections to C&C servers and downloads.
*   **Suspicious Domain Lookups:** Highlights domains frequently involved in malicious activity.
*   **Suspicious ipinfo Requests:** Flags suspicious IP information requests.
*   **Suspicious Direct File Downloads:** Detects direct file download attempts.
*   **Suspicious HTTP Requests:**  Identifies potentially malicious HTTP requests.
*   **Port Scanning:** Detects port scanning activity.
*   **DNS Resource Exhaustion:** Identifies DNS resource exhaustion attacks.
*   **Data Leakage:** Identifies potential data exfiltration attempts.
*   **False Positives:**  Acknowledges that false positives can occur and provides guidance.

## Best Practice(s)

See original README.

1.  Install Maltrail following steps in original README.
2.  Configure environment following steps in original README.
3.  Set running environment following steps in original README.
4.  Enable as systemd services (Linux only) following steps in original README.

## License

Maltrail is licensed under the MIT License. See the [LICENSE](https://github.com/stamparm/maltrail/blob/master/LICENSE) file.

## Sponsors

*   [Sansec](https://sansec.io/) (2024-)
*   [Sansec](https://sansec.io/) (2020-2021)

## Developers

*   Miroslav Stampar ([@stamparm](https://github.com/stamparm))
*   Mikhail Kasimov ([@MikhailKasimov](https://github.com/MikhailKasimov))

## Presentations

*   47th TF-CSIRT Meeting, Prague (Czech Republic), 2016 ([slides](https://www.terena.org/activities/tf-csirt/meeting47/M.Stampar-Maltrail.pdf))

## Publications

*   Detect attacks on your network with Maltrail, Linux Magazine, 2022 ([Annotation](https://www.linux-magazine.com/Issues/2022/258/Maltrail))
*   Best Cyber Threat Intelligence Feeds ([SilentPush Review, 2022](https://www.silentpush.com/blog/best-cyber-threat-intelligence-feeds))
*   Research on Network Malicious Traffic Detection System Based on Maltrail ([Nanotechnology Perceptions, ISSN 1660-6795, 2024](https://nano-ntp.com/index.php/nano/article/view/1915/1497))

## Blacklist

*   Maltrail's malware-related domain blacklist is available [here](https://raw.githubusercontent.com/stamparm/aux/master/maltrail-malware-domains.txt).

## Thank You

*   Thomas Kristner
*   Eduardo Arcusa Les
*   James Lay
*   Ladislav Baco (@laciKE)
*   John Kristoff (@jtkdpu)
*   Michael M&uuml;nz (@mimugmail)
*   David Brush
*   @Godwottery
*   Chris Wild (@briskets)

## Third-party integrations

*   [FreeBSD Port](https://www.freshports.org/security/maltrail)
*   [OPNSense Gateway Plugin](https://github.com/opnsense/plugins/pull/1257)
*   [D4 Project](https://www.d4-project.org/2019/09/25/maltrail-integration.html)
*   [BlackArch Linux](https://github.com/BlackArch/blackarch/blob/master/packages/maltrail/PKGBUILD)
*   [Validin LLC](https://twitter.com/ValidinLLC/status/1719666086390517762)
*   [Maltrail Add-on for Splunk](https://splunkbase.splunk.com/app/7211)
*   [Maltrail decoder and rules for Wazuh](https://github.com/MikhailKasimov/maltrail-wazuh-decoder-and-rules)
*   [GScan](https://github.com/grayddq/GScan) <sup>1</sup>
*   [MalwareWorld](https://www.malwareworld.com/) <sup>1</sup>
*   [oisd | domain blocklist](https://oisd.nl/?p=inc) <sup>1</sup>
*   [NextDNS](https://github.com/nextdns/metadata/blob/e0c9c7e908f5d10823b517ad230df214a7251b13/security/threat-intelligence-feeds.json) <sup>1</sup>
*   [NoTracking](https://github.com/notracking/hosts-blocklists/blob/master/SOURCES.md) <sup>1</sup>
*   [OWASP Mobile Audit](https://github.com/mpast/mobileAudit#environment-variables) <sup>1</sup>
*   [Mobile-Security-Framework-MobSF](https://github.com/MobSF/Mobile-Security-Framework-MobSF/commit/12b07370674238fa4281fc7989b34decc2e08876) <sup>1</sup>
*   [pfBlockerNG-devel](https://github.com/pfsense/FreeBSD-ports/blob/devel/net/pfSense-pkg-pfBlockerNG-devel/files/usr/local/www/pfblockerng/pfblockerng_feeds.json) <sup>1</sup>
*   [Sansec eComscan](https://sansec.io/kb/about-ecomscan/ecomscan-license)<sup>1</sup>
*   [Palo Alto Networks Cortex XSOAR](https://xsoar.pan.dev/docs/reference/integrations/github-maltrail-feed)<sup>2</sup>
 
<sup>1</sup> Using (only) trails

<sup>2</sup> Connector to trails (only)