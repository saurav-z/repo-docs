# System Design Primer: Your Guide to Building Scalable Systems

**Learn how to design and build large-scale systems with this comprehensive, community-driven resource. Improve your system design interview skills and understand the principles behind building robust, scalable applications.  Explore the original repository [here](https://github.com/donnemartin/system-design-primer).**

This repository provides a structured, organized collection of resources to help you master system design concepts, prepare for technical interviews, and enhance your understanding of building scalable systems.

## Key Features

*   **Comprehensive Coverage:** Explore a wide range of system design topics.
*   **Interview Preparation:** Prepare for system design interviews with practice questions and solutions.
*   **Community Driven:** Contribute to a continually updated, open-source project.
*   **Anki Flashcards:** Utilize spaced repetition with provided Anki decks.
*   **Real-World Examples:** Study architectures of popular companies and systems.

## Index of System Design Topics

> Summaries of various system design topics, including pros and cons. **Everything is a trade-off.** Each section contains links to more in-depth resources.

*   [System Design Topics: Start Here](#system-design-topics-start-here)
    *   [Step 1: Review the Scalability Video Lecture](#step-1-review-the-scalability-video-lecture)
    *   [Step 2: Review the Scalability Article](#step-2-review-the-scalability-article)
    *   [Next Steps](#next-steps)
*   [Performance vs. Scalability](#performance-vs-scalability)
*   [Latency vs. Throughput](#latency-vs-throughput)
*   [Availability vs. Consistency](#availability-vs-consistency)
    *   [CAP Theorem](#cap-theorem)
        *   [CP - Consistency and Partition Tolerance](#cp---consistency-and-partition-tolerance)
        *   [AP - Availability and Partition Tolerance](#ap---availability-and-partition-tolerance)
*   [Consistency Patterns](#consistency-patterns)
    *   [Weak Consistency](#weak-consistency)
    *   [Eventual Consistency](#eventual-consistency)
    *   [Strong Consistency](#strong-consistency)
*   [Availability Patterns](#availability-patterns)
    *   [Fail-Over](#fail-over)
    *   [Replication](#replication)
    *   [Availability in Numbers](#availability-in-numbers)
*   [Domain Name System](#domain-name-system)
*   [Content Delivery Network](#content-delivery-network)
    *   [Push CDNs](#push-cdns)
    *   [Pull CDNs](#pull-cdns)
*   [Load Balancer](#load-balancer)
    *   [Active-Passive](#active-passive)
    *   [Active-Active](#active-active)
    *   [Layer 4 Load Balancing](#layer-4-load-balancing)
    *   [Layer 7 Load Balancing](#layer-7-load-balancing)
    *   [Horizontal Scaling](#horizontal-scaling)
*   [Reverse Proxy (Web Server)](#reverse-proxy-web-server)
    *   [Load Balancer vs. Reverse Proxy](#load-balancer-vs-reverse-proxy)
*   [Application Layer](#application-layer)
    *   [Microservices](#microservices)
    *   [Service Discovery](#service-discovery)
*   [Database](#database)
    *   [Relational Database Management System (RDBMS)](#relational-database-management-system-rdbms)
        *   [Master-Slave Replication](#master-slave-replication)
        *   [Master-Master Replication](#master-master-replication)
        *   [Federation](#federation)
        *   [Sharding](#sharding)
        *   [Denormalization](#denormalization)
        *   [SQL Tuning](#sql-tuning)
    *   [NoSQL](#nosql)
        *   [Key-Value Store](#key-value-store)
        *   [Document Store](#document-store)
        *   [Wide Column Store](#wide-column-store)
        *   [Graph Database](#graph-database)
    *   [SQL or NoSQL](#sql-or-nosql)
*   [Cache](#cache)
    *   [Client Caching](#client-caching)
    *   [CDN Caching](#cdn-caching)
    *   [Web Server Caching](#web-server-caching)
    *   [Database Caching](#database-caching)
    *   [Application Caching](#application-caching)
    *   [Caching at the Database Query Level](#caching-at-the-database-query-level)
    *   [Caching at the Object Level](#caching-at-the-object-level)
    *   [When to Update the Cache](#when-to-update-the-cache)
        *   [Cache-Aside](#cache-aside)
        *   [Write-Through](#write-through)
        *   [Write-Behind (Write-Back)](#write-behind-write-back)
        *   [Refresh-Ahead](#refresh-ahead)
*   [Asynchronism](#asynchronism)
    *   [Message Queues](#message-queues)
    *   [Task Queues](#task-queues)
    *   [Back Pressure](#back-pressure)
*   [Communication](#communication)
    *   [Transmission Control Protocol (TCP)](#transmission-control-protocol-tcp)
    *   [User Datagram Protocol (UDP)](#user-datagram-protocol-udp)
    *   [Remote Procedure Call (RPC)](#remote-procedure-call-rpc)
    *   [Representational State Transfer (REST)](#representational-state-transfer-rest)
*   [Security](#security)
*   [Appendix](#appendix)
    *   [Powers of Two Table](#powers-of-two-table)
    *   [Latency Numbers Every Programmer Should Know](#latency-numbers-every-programmer-should-know)
    *   [Additional System Design Interview Questions](#additional-system-design-interview-questions)
    *   [Real World Architectures](#real-world-architectures)
    *   [Company Architectures](#company-architectures)
    *   [Company Engineering Blogs](#company-engineering-blogs)
*   [Under Development](#under-development)
*   [Credits](#credits)
*   [Contact Info](#contact-info)
*   [License](#license)

## Study Guide

> Suggested topics to review based on your interview timeline (short, medium, long).

![Imgur](images/OfVllex.png)

**Q: For interviews, do I need to know everything here?**

**A: No, you don't need to know everything here to prepare for the interview.**

Start broad and go deeper in a few areas. It helps to know a little about various key system design topics. Adjust the following guide based on your timeline, experience, what positions you are interviewing for, and which companies you are interviewing with.

*   **Short timeline** - Aim for **breadth** with system design topics. Practice by solving **some** interview questions.
*   **Medium timeline** - Aim for **breadth** and **some depth** with system design topics. Practice by solving **many** interview questions.
*   **Long timeline** - Aim for **breadth** and **more depth** with system design topics. Practice by solving **most** interview questions.

|                                                                                                                                                     | Short   | Medium | Long   |
| :-------------------------------------------------------------------------------------------------------------------------------------------------- | :------ | :----- | :----- |
| Read through the [System design topics](#index-of-system-design-topics) to get a broad understanding of how systems work                                  | :+1:    | :+1:   | :+1:   |
| Read through a few articles in the [Company engineering blogs](#company-engineering-blogs) for the companies you are interviewing with                | :+1:    | :+1:   | :+1:   |
| Read through a few [Real world architectures](#real-world-architectures)                                                                           | :+1:    | :+1:   | :+1:   |
| Review [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question)                                    | :+1:    | :+1:   | :+1:   |
| Work through [System design interview questions with solutions](#system-design-interview-questions-with-solutions)                                  | Some    | Many   | Most   |
| Work through [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions)               | Some    | Many   | Most   |
| Review [Additional system design interview questions](#additional-system-design-interview-questions)                                                | Some    | Many   | Most   |

## How to Approach a System Design Interview Question

> How to tackle a system design interview question.

You can use the following steps to guide the discussion. To help solidify this process, work through the [System design interview questions with solutions](#system-design-interview-questions-with-solutions) section using the following steps.

### Step 1: Outline Use Cases, Constraints, and Assumptions

Gather requirements and scope the problem. Ask questions to clarify use cases and constraints. Discuss assumptions.

*   Who is going to use it?
*   How are they going to use it?
*   How many users are there?
*   What does the system do?
*   What are the inputs and outputs of the system?
*   How much data do we expect to handle?
*   How many requests per second do we expect?
*   What is the expected read to write ratio?

### Step 2: Create a High-Level Design

Outline a high-level design with all important components.

*   Sketch the main components and connections
*   Justify your ideas

### Step 3: Design Core Components

Dive into details for each core component.

### Step 4: Scale the Design

Identify and address bottlenecks, given the constraints.

### Back-of-the-Envelope Calculations

Refer to the [Appendix](#appendix) for resources:

*   [Use back of the envelope calculations](http://highscalability.com/blog/2011/1/26/google-pro-tip-use-back-of-the-envelope-calculations-to-choo.html)
*   [Powers of two table](#powers-of-two-table)
*   [Latency numbers every programmer should know](#latency-numbers-every-programmer-should-know)

### Source(s) and Further Reading

*   [How to ace a systems design interview](https://www.palantir.com/2011/10/how-to-rock-a-systems-design-interview/)
*   [The system design interview](http://www.hiredintech.com/system-design)
*   [Intro to Architecture and Systems Design Interviews](https://www.youtube.com/watch?v=ZgdS0EUmn70)
*   [System design template](https://leetcode.com/discuss/career/229177/My-System-Design-Template)

## System Design Interview Questions with Solutions

> Common system design interview questions with sample discussions, code, and diagrams. Solutions are linked to content in the `solutions/` folder.

| Question                                          |                                                             |
| :------------------------------------------------ | :---------------------------------------------------------- |
| Design Pastebin.com (or Bit.ly)                  | [Solution](solutions/system_design/pastebin/README.md)      |
| Design the Twitter timeline and search (or Facebook feed and search) | [Solution](solutions/system_design/twitter/README.md)      |
| Design a web crawler                                | [Solution](solutions/system_design/web_crawler/README.md) |
| Design Mint.com                                   | [Solution](solutions/system_design/mint/README.md)          |
| Design the data structures for a social network    | [Solution](solutions/system_design/social_graph/README.md) |
| Design a key-value store for a search engine      | [Solution](solutions/system_design/query_cache/README.md)   |
| Design Amazon's sales ranking by category feature | [Solution](solutions/system_design/sales_rank/README.md)    |
| Design a system that scales to millions of users on AWS  | [Solution](solutions/system_design/scaling_aws/README.md)      |
| Add a system design question                      | [Contribute](#contributing)                                |

### Design Pastebin.com (or Bit.ly)

[View exercise and solution](solutions/system_design/pastebin/README.md)

![Imgur](images/4edXG0T.png)

### Design the Twitter timeline and search (or Facebook feed and search)

[View exercise and solution](solutions/system_design/twitter/README.md)

![Imgur](images/jrUBAF7.png)

### Design a web crawler

[View exercise and solution](solutions/system_design/web_crawler/README.md)

![Imgur](images/bWxPtQA.png)

### Design Mint.com

[View exercise and solution](solutions/system_design/mint/README.md)

![Imgur](images/V5q57vU.png)

### Design the data structures for a social network

[View exercise and solution](solutions/system_design/social_graph/README.md)

![Imgur](images/cdCv5g7.png)

### Design a key-value store for a search engine

[View exercise and solution](solutions/system_design/query_cache/README.md)

![Imgur](images/4j99mhe.png)

### Design Amazon's sales ranking by category feature

[View exercise and solution](solutions/system_design/sales_rank/README.md)

![Imgur](images/MzExP06.png)

### Design a system that scales to millions of users on AWS

[View exercise and solution](solutions/system_design/scaling_aws/README.md)

![Imgur](images/jj3A5N8.png)

## Object-Oriented Design Interview Questions with Solutions

> Common object-oriented design interview questions with sample discussions, code, and diagrams. Solutions are linked to content in the `solutions/` folder.

>**Note: This section is under development**

| Question                               |                                                      |
| :------------------------------------- | :--------------------------------------------------- |
| Design a hash map                      | [Solution](solutions/object_oriented_design/hash_table/hash_map.ipynb)  |
| Design a least recently used cache     | [Solution](solutions/object_oriented_design/lru_cache/lru_cache.ipynb)  |
| Design a call center                   | [Solution](solutions/object_oriented_design/call_center/call_center.ipynb)  |
| Design a deck of cards                 | [Solution](solutions/object_oriented_design/deck_of_cards/deck_of_cards.ipynb)  |
| Design a parking lot                   | [Solution](solutions/object_oriented_design/parking_lot/parking_lot.ipynb)  |
| Design a chat server                   | [Solution](solutions/object_oriented_design/online_chat/online_chat.ipynb)  |
| Design a circular array                | [Contribute](#contributing)                          |
| Add an object-oriented design question | [Contribute](#contributing)                          |

## System Design Topics: Start Here

New to system design?

First, you'll need a basic understanding of common principles, learning about what they are, how they are used, and their pros and cons.

### Step 1: Review the Scalability Video Lecture

[Scalability Lecture at Harvard](https://www.youtube.com/watch?v=-W9F__D3oY4)

*   Topics covered:
    *   Vertical scaling
    *   Horizontal scaling
    *   Caching
    *   Load balancing
    *   Database replication
    *   Database partitioning

### Step 2: Review the Scalability Article

[Scalability](https://web.archive.org/web/20221030091841/http://www.lecloud.net/tagged/scalability/chrono)

*   Topics covered:
    *   [Clones](https://web.archive.org/web/20220530193911/https://www.lecloud.net/post/7295452622/scalability-for-dummies-part-1-clones)
    *   [Databases](https://web.archive.org/web/20220602114024/https://www.lecloud.net/post/7994751381/scalability-for-dummies-part-2-database)
    *   [Caches](https://web.archive.org/web/20230126233752/https://www.lecloud.net/post/9246290032/scalability-for-dummies-part-3-cache)
    *   [Asynchronism](https://web.archive.org/web/20220926171507/https://www.lecloud.net/post/9699762917/scalability-for-dummies-part-4-asynchronism)

### Next Steps

Next, we'll look at high-level trade-offs:

*   **Performance** vs. **Scalability**
*   **Latency** vs. **Throughput**
*   **Availability** vs. **Consistency**

Keep in mind that **everything is a trade-off**.

Then we'll dive into more specific topics such as DNS, CDNs, and load balancers.

## Performance vs. Scalability

A service is **scalable** if it results in increased **performance** in a manner proportional to resources added. Generally, increasing performance means serving more units of work, but it can also be to handle larger units of work, such as when datasets grow.<sup><a href=http://www.allthingsdistributed.com/2006/03/a_word_on_scalability.html>1</a></sup>

Another way to look at performance vs. scalability:

*   If you have a **performance** problem, your system is slow for a single user.
*   If you have a **scalability** problem, your system is fast for a single user but slow under heavy load.

### Source(s) and Further Reading

*   [A word on scalability](http://www.allthingsdistributed.com/2006/03/a_word_on_scalability.html)
*   [Scalability, availability, stability, patterns](http://www.slideshare.net/jboner/scalability-availability-stability-patterns/)

## Latency vs. Throughput

**Latency** is the time to perform some action or to produce some result.

**Throughput** is the number of such actions or results per unit of time.

Generally, you should aim for **maximal throughput** with **acceptable latency**.

### Source(s) and Further Reading

*   [Understanding latency vs. throughput](https://community.cadence.com/cadence_blogs_8/b/fv/posts/understanding-latency-vs-throughput)

## Availability vs. Consistency

### CAP Theorem

<p align="center">
  <img src="images/bgLMI2u.png">
  <br/>
  <i><a href=http://robertgreiner.com/2014/08/cap-theorem-revisited>Source: CAP theorem revisited</a></i>
</p>

In a distributed computer system, you can only support two of the following guarantees:

*   **Consistency** - Every read receives the most recent write or an error
*   **Availability** - Every request receives a response, without guarantee that it contains the most recent version of the information
*   **Partition Tolerance** - The system continues to operate despite arbitrary partitioning due to network failures

*Networks aren't reliable, so you'll need to support partition tolerance. You'll need to make a software tradeoff between consistency and availability.*

#### CP - Consistency and Partition Tolerance

Waiting for a response from the partitioned node might result in a timeout error. CP is a good choice if your business needs require atomic reads and writes.

#### AP - Availability and Partition Tolerance

Responses return the most readily available version of the data available on any node, which might not be the latest. Writes might take some time to propagate when the partition is resolved.

AP is a good choice if the business needs to allow for [eventual consistency](#eventual-consistency) or when the system needs to continue working despite external errors.

### Source(s) and Further Reading

*   [CAP theorem revisited](http://robertgreiner.com/2014/08/cap-theorem-revisited/)
*   [A plain english introduction to CAP theorem](http://ksat.me/a-plain-english-introduction-to-cap-theorem)
*   [CAP FAQ](https://github.com/henryr/cap-faq)
*   [The CAP theorem](https://www.youtube.com/watch?v=k-Yaq8AHlFA)

## Consistency Patterns

With multiple copies of the same data, we are faced with options on how to synchronize them so clients have a consistent view of the data. Recall the definition of consistency from the [CAP theorem](#cap-theorem) - Every read receives the most recent write or an error.

### Weak Consistency

After a write, reads may or may not see it. A best effort approach is taken.

This approach is seen in systems such as memcached. Weak consistency works well in real time use cases such as VoIP, video chat, and realtime multiplayer games. For example, if you are on a phone call and lose reception for a few seconds, when you regain connection you do not hear what was spoken during connection loss.

### Eventual Consistency

After a write, reads will eventually see it (typically within milliseconds). Data is replicated asynchronously.

This approach is seen in systems such as DNS and email. Eventual consistency works well in highly available systems.

### Strong Consistency

After a write, reads will see it. Data is replicated synchronously.

This approach is seen in file systems and RDBMSes. Strong consistency works well in systems that need transactions.

### Source(s) and Further Reading

*   [Transactions across data centers](http://snarfed.org/transactions_across_datacenters_io.html)

## Availability Patterns

There are two complementary patterns to support high availability: **fail-over** and **replication**.

### Fail-Over

#### Active-Passive

With active-passive fail-over, heartbeats are sent between the active and the passive server on standby. If the heartbeat is interrupted, the passive server takes over the active's IP address and resumes service.

The length of downtime is determined by whether the passive server is already running in 'hot' standby or whether it needs to start up from 'cold' standby. Only the active server handles traffic.

Active-passive failover can also be referred to as master-slave failover.

#### Active-Active

In active-active, both servers are managing traffic, spreading the load between them.

If the servers are public-facing, the DNS would need to know about the public IPs of both servers. If the servers are internal-facing, application logic would need to know about both servers.

Active-active failover can also be referred to as master-master failover.

### Disadvantage(s): failover

*   Fail-over adds more hardware and additional complexity.
*   There is a potential for loss of data if the active system fails before any newly written data can be replicated to the passive.

### Replication

#### Master-Slave and Master-Master

This topic is further discussed in the [Database](#database) section:

*   [Master-slave replication](#master-slave-replication)
*   [Master-master replication](#master-master-replication)

### Availability in Numbers

Availability is often quantified by uptime (or downtime) as a percentage of time the service is available. Availability is generally measured in number of 9s--a service with 99.99% availability is described as having four 9s.

#### 99.9% availability - three 9s

| Duration            | Acceptable downtime |
| :------------------ | :------------------ |
| Downtime per year   | 8h 45min 57s        |
| Downtime per month  | 43m 49.7s           |
| Downtime per week   | 10m 4.8s            |
| Downtime per day    | 1m 26.4s            |

#### 99.99% availability - four 9s

| Duration            | Acceptable downtime |
| :------------------ | :------------------ |
| Downtime per year   | 52min 35.7s         |
| Downtime per month  | 4m 23s              |
| Downtime per week   | 1m 5s               |
| Downtime per day    | 8.6s                |

#### Availability in Parallel vs. in Sequence

If a service consists of multiple components prone to failure, the service's overall availability depends on whether the components are in sequence or in parallel.

###### In Sequence

Overall availability decreases when two components with availability < 100% are in sequence:

```
Availability (Total) = Availability (Foo) * Availability (Bar)
```

If both `Foo` and `Bar` each had 99.9% availability, their total availability in sequence would be 99.8%.

###### In Parallel

Overall availability increases when two components with availability < 100% are in parallel:

```
Availability (Total) = 1 - (1 - Availability (Foo)) * (1 - Availability (Bar))
```

If both `Foo` and `Bar` each had 99.9% availability, their total availability in parallel would be 99.9999%.

## Domain Name System

<p align="center">
  <img src="images/IOyLj4i.jpg">
  <br/>
  <i><a href=http://www.slideshare.net/srikrupa5/dns-security-presentation-issa>Source: DNS security presentation</a></i>
</p>

A Domain Name System (DNS) translates a domain name such as www.example.com to an IP address.

DNS is hierarchical, with a few authoritative servers at the top level. Your router or ISP provides information about which DNS server(s) to contact when doing a lookup. Lower level DNS servers cache mappings, which could become stale due to DNS propagation delays. DNS results can also be cached by your browser or OS for a certain period of time, determined by the [time to live (TTL)](https://en.wikipedia.org/wiki/Time_to_live).

*   **NS record (name server)** - Specifies the DNS servers for your domain/subdomain.
*   **MX record (mail exchange)** - Specifies the mail servers for accepting messages.
*   **A record (address)** - Points a name to an IP address.
*   **CNAME (canonical)** - Points a name to another name or `CNAME` (example.com to www.example.com) or to an `A` record.

Services such as [CloudFlare](https://www.cloudflare.com/dns/) and [Route 53](https://aws.amazon.com/route53/) provide managed DNS services. Some DNS services can route traffic through various methods:

*   [Weighted round robin](https://www.jscape.com/blog/load-balancing-algorithms)
    *   Prevent traffic from going to servers under maintenance
    *   Balance between varying cluster sizes
    *   A/B testing
*   [Latency-based](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/routing-policy-latency.html)
*   [Geolocation-based](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/routing-policy-geo.html)

### Disadvantage(s): DNS

*   Accessing a DNS server introduces a slight delay, although mitigated by caching described above.
*   DNS server management could be complex and is generally managed by [governments, ISPs, and large companies](http://superuser.com/questions/472695/who-controls-the-dns-servers/472729).
*   DNS services have recently come under [DDoS attack](http://dyn.com/blog/dyn-analysis-summary-of-friday-october-21-attack/), preventing users from accessing websites such as Twitter without knowing Twitter's IP address(es).

### Source(s) and Further Reading

*   [DNS architecture](https://technet.microsoft.com/en-us/library/dd197427(v=ws.10).aspx)
*   [Wikipedia](https://en.wikipedia.org/wiki/Domain_Name_System)
*   [DNS articles](https://support.dnsimple.com/categories/dns/)

## Content Delivery Network

<p align="center">
  <img src="images/h9TAuGI.jpg">
  <br/>
  <i><a href=https://www.creative-artworks.eu/why-use-a-content-delivery-network-cdn/>Source: Why use a CDN</a></i>
</p>

A content delivery network (CDN) is a globally distributed network of proxy servers, serving content from locations closer to the user. Generally, static files such as HTML/CSS/JS, photos, and videos are served from CDN, although some CDNs such as Amazon's CloudFront support dynamic content. The site's DNS resolution will tell clients which server to contact.

Serving content from CDNs can significantly improve performance in two ways:

*   Users receive content from data centers close to them
*   Your servers do not have to serve requests that the CDN fulfills

### Push CDNs

Push CDNs receive new content whenever changes occur on your server. You take full responsibility for providing content, uploading directly to the CDN and rewriting URLs to point to the CDN. You can configure when content expires and when it is updated. Content is uploaded only when it is new or changed, minimizing traffic, but maximizing storage.

Sites with a small amount of traffic or sites with content that isn't often updated work well with push CDNs. Content is placed on the CDNs once, instead of being re-pulled at regular intervals.

### Pull CDNs

Pull CDNs grab new content from your server when the first user requests the content. You leave the content on your server and rewrite URLs to point to the CDN. This results in a slower request until the content is cached on the CDN.

A [time-to-live (TTL)](https://en.wikipedia.org/wiki/Time_to_live) determines how long content is cached. Pull CDNs minimize storage space on the CDN, but can create redundant traffic if files expire and are pulled before they have actually changed.

Sites with heavy traffic work well with pull CDNs, as traffic is spread out more evenly with only recently-requested content remaining on the CDN.

### Disadvantage(s): CDN

*   CDN costs could be significant depending on traffic, although this should be weighed with additional costs you would incur not using a CDN.
*   Content might be stale if it is updated before the TTL expires it.
*   CDNs require changing URLs for static content to point to the CDN.

### Source(s) and Further Reading

*   [Globally distributed content delivery](https://figshare.com/articles/Globally_distributed_content_delivery/6605972)
*   [The differences between push and pull CDNs](http://www.travelblogadvice.com/technical/the-differences-between-push-and-pull-cdns/)
*   [Wikipedia](https://en.wikipedia.org/wiki/Content_delivery_network)

## Load Balancer

<p align="center">
  <img src="images/h81n9iK.png">
  <br/>
  <i><a href=http://horicky.blogspot.com/2010/10/scalable-system-design-patterns.html>Source: Scalable system design patterns</a></i>
</p>

Load balancers distribute incoming client requests to computing resources such as application servers and databases. In each case, the load balancer returns the response from the computing resource to the appropriate client. Load balancers are effective at:

*   Preventing requests from going to unhealthy servers
*   Preventing overloading resources
*   Helping to eliminate a single point of failure

Load balancers can be implemented with hardware (expensive) or with software such as HAProxy.

Additional benefits include:

*   **SSL termination** - Decrypt incoming requests and encrypt server responses so backend servers do not have to perform these potentially expensive operations
    *   Removes the need to install [X.509 certificates](https://en.wikipedia.org/wiki/X.509) on each server
*   **Session persistence** - Issue cookies and route a specific client's requests to same instance if the web apps do not keep track of sessions

To protect against failures, it's common to set up multiple load balancers, either in [active-passive](#active-passive) or [active-active](#active-active) mode.

Load balancers can route traffic based on various metrics, including:

*   Random
*   Least loaded
*   Session/cookies
*   [Round robin or weighted round robin](https://www.g33kinfo.com/info/round-robin-vs-weighted-round-robin-lb)
*   [Layer 4](#layer-4-load-balancing)
*   [Layer 7](#layer-7-load-balancing)

### Layer 4 Load Balancing

Layer 4 load balancers look at info at the [transport layer](#communication) to decide how to distribute requests. Generally, this involves the source, destination IP addresses, and ports in the header, but not the contents of the packet. Layer 4 load balancers forward network packets to and from the upstream server, performing [Network Address Translation (NAT)](https://www.nginx.com/resources/glossary/layer-4-load-balancing/).

### Layer 7 Load Balancing

Layer 7 load balancers look at the [application layer](#communication) to decide how to distribute requests. This can involve contents of the header, message, and cookies. Layer 7 load balancers terminate network traffic, reads the message, makes a load-balancing decision, then opens a connection to the selected server. For example, a layer 7 load balancer can direct video traffic to servers that host videos while directing more sensitive user billing traffic to security-hardened servers.

At the cost of flexibility, layer 4 load balancing requires less time and computing resources than Layer 7, although the performance impact can be minimal on modern commodity hardware.

### Horizontal Scaling

Load balancers can also help with horizontal scaling, improving performance and availability. Scaling out using commodity machines is more cost efficient and results in higher availability than scaling up a single server on more expensive hardware, called **Vertical Scaling**. It is also easier to hire for talent working on commodity hardware than it is for specialized enterprise systems.

#### Disadvantage(s): horizontal scaling

*   Scaling horizontally introduces complexity and involves cloning servers
    *   Servers should be stateless: they should not contain any user-related data like sessions or profile pictures
    *   Sessions can be stored in a centralized data store such as a [database](#database) (SQL, NoSQL) or a persistent [cache](#cache) (Redis, Memcached)
*   Downstream servers such as caches and databases need to handle more simultaneous connections as upstream servers scale out

### Disadvantage(s): load balancer

*   The load balancer can become a performance bottleneck if it does not have enough resources or if it is not configured properly.
*   Introducing a load balancer to help eliminate a single point of failure results in increased complexity.
*   A single load balancer is a