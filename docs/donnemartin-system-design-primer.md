# System Design Primer: Ace Your Next Interview!

This repository provides a comprehensive guide to system design, helping you prepare for technical interviews and build large-scale systems.  Explore the resources at the original repo: [donnemartin/system-design-primer](https://github.com/donnemartin/system-design-primer).

## Key Features

*   **Comprehensive Coverage:** Learn about fundamental system design concepts and principles.
*   **Interview Prep:** Prepare for system design interview questions with sample solutions.
*   **Community Driven:** Benefit from an open-source project with contributions from the community.
*   **Anki Flashcards:** Utilize spaced repetition through Anki decks for key concept retention.
*   **Interactive Coding Challenges:** Complement your preparation with interactive coding challenges.

## Table of Contents

*   [Introduction](#system-design-primer-ace-your-next-interview)
    *   [Key Features](#key-features)
    *   [Table of Contents](#table-of-contents)
*   [System Design Topics: Start Here](#system-design-topics-start-here)
    *   [Step 1: Review the Scalability Video Lecture](#step-1-review-the-scalability-video-lecture)
    *   [Step 2: Review the Scalability Article](#step-2-review-the-scalability-article)
    *   [Next Steps](#next-steps)
*   [Performance vs Scalability](#performance-vs-scalability)
*   [Latency vs Throughput](#latency-vs-throughput)
*   [Availability vs Consistency](#availability-vs-consistency)
    *   [CAP Theorem](#cap-theorem)
    *   [CP - Consistency and Partition Tolerance](#cp---consistency-and-partition-tolerance)
    *   [AP - Availability and Partition Tolerance](#ap---availability-and-partition-tolerance)
*   [Consistency Patterns](#consistency-patterns)
    *   [Weak Consistency](#weak-consistency)
    *   [Eventual Consistency](#eventual-consistency)
    *   [Strong Consistency](#strong-consistency)
*   [Availability Patterns](#availability-patterns)
    *   [Fail-over](#fail-over)
        *   [Active-passive](#active-passive)
        *   [Active-active](#active-active)
    *   [Replication](#replication)
    *   [Availability in Numbers](#availability-in-numbers)
*   [Domain Name System](#domain-name-system)
*   [Content Delivery Network](#content-delivery-network)
    *   [Push CDNs](#push-cdns)
    *   [Pull CDNs](#pull-cdns)
*   [Load Balancer](#load-balancer)
    *   [Layer 4 Load Balancing](#layer-4-load-balancing)
    *   [Layer 7 Load Balancing](#layer-7-load-balancing)
    *   [Horizontal Scaling](#horizontal-scaling)
*   [Reverse Proxy (Web Server)](#reverse-proxy-web-server)
    *   [Load Balancer vs Reverse Proxy](#load-balancer-vs-reverse-proxy)
*   [Application Layer](#application-layer)
    *   [Microservices](#microservices)
    *   [Service Discovery](#service-discovery)
*   [Database](#database)
    *   [Relational Database Management System (RDBMS)](#relational-database-management-system-rdbms)
        *   [Master-slave Replication](#master-slave-replication)
        *   [Master-master Replication](#master-master-replication)
        *   [Federation](#federation)
        *   [Sharding](#sharding)
        *   [Denormalization](#denormalization)
        *   [SQL Tuning](#sql-tuning)
    *   [NoSQL](#nosql)
        *   [Key-value Store](#key-value-store)
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
        *   [Cache-aside](#cache-aside)
        *   [Write-through](#write-through)
        *   [Write-behind (Write-back)](#write-behind-write-back)
        *   [Refresh-ahead](#refresh-ahead)
*   [Asynchronism](#asynchronism)
    *   [Message Queues](#message-queues)
    *   [Task Queues](#task-queues)
    *   [Back Pressure](#back-pressure)
*   [Communication](#communication)
    *   [Hypertext Transfer Protocol (HTTP)](#hypertext-transfer-protocol-http)
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
*   [Study Guide](#study-guide)
*   [How to Approach a System Design Interview Question](#how-to-approach-a-system-design-interview-question)
*   [System Design Interview Questions with Solutions](#system-design-interview-questions-with-solutions)
    *   [Design Pastebin.com (or Bit.ly)](#design-pastebincom-or-bitly)
    *   [Design the Twitter Timeline and Search (or Facebook Feed and Search)](#design-the-twitter-timeline-and-search-or-facebook-feed-and-search)
    *   [Design a Web Crawler](#design-a-web-crawler)
    *   [Design Mint.com](#design-mintcom)
    *   [Design the Data Structures for a Social Network](#design-the-data-structures-for-a-social-network)
    *   [Design a Key-value Store for a Search Engine](#design-a-key-value-store-for-a-search-engine)
    *   [Design Amazon's Sales Ranking by Category Feature](#design-amazons-sales-ranking-by-category-feature)
    *   [Design a System that Scales to Millions of Users on AWS](#design-a-system-that-scales-to-millions-of-users-on-aws)
*   [Object-oriented Design Interview Questions with Solutions](#object-oriented-design-interview-questions-with-solutions)
*   [Under Development](#under-development)
*   [Credits](#credits)
*   [Contact Info](#contact-info)
*   [License](#license)

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

*   **Performance** vs **scalability**
*   **Latency** vs **throughput**
*   **Availability** vs **consistency**

Keep in mind that **everything is a trade-off**.

Then we'll dive into more specific topics such as DNS, CDNs, and load balancers.

## Performance vs Scalability

A service is **scalable** if it results in increased **performance** in a manner proportional to resources added. Generally, increasing performance means serving more units of work, but it can also be to handle larger units of work, such as when datasets grow.<sup><a href=http://www.allthingsdistributed.com/2006/03/a_word_on_scalability.html>1</a></sup>

Another way to look at performance vs scalability:

*   If you have a **performance** problem, your system is slow for a single user.
*   If you have a **scalability** problem, your system is fast for a single user but slow under heavy load.

### Source(s) and further reading

*   [A word on scalability](http://www.allthingsdistributed.com/2006/03/a_word_on_scalability.html)
*   [Scalability, availability, stability, patterns](http://www.slideshare.net/jboner/scalability-availability-stability-patterns/)

## Latency vs Throughput

**Latency** is the time to perform some action or to produce some result.

**Throughput** is the number of such actions or results per unit of time.

Generally, you should aim for **maximal throughput** with **acceptable latency**.

### Source(s) and further reading

*   [Understanding latency vs throughput](https://community.cadence.com/cadence_blogs_8/b/fv/posts/understanding-latency-vs-throughput)

## Availability vs Consistency

### CAP theorem

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

#### CP - consistency and partition tolerance

Waiting for a response from the partitioned node might result in a timeout error. CP is a good choice if your business needs require atomic reads and writes.

#### AP - availability and partition tolerance

Responses return the most readily available version of the data available on any node, which might not be the latest. Writes might take some time to propagate when the partition is resolved.

AP is a good choice if the business needs to allow for [eventual consistency](#eventual-consistency) or when the system needs to continue working despite external errors.

### Source(s) and further reading

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

### Source(s) and further reading

*   [Transactions across data centers](http://snarfed.org/transactions_across_datacenters_io.html)

## Availability Patterns

There are two complementary patterns to support high availability: **fail-over** and **replication**.

### Fail-over

#### Active-passive

With active-passive fail-over, heartbeats are sent between the active and the passive server on standby. If the heartbeat is interrupted, the passive server takes over the active's IP address and resumes service.

The length of downtime is determined by whether the passive server is already running in 'hot' standby or whether it needs to start up from 'cold' standby. Only the active server handles traffic.

Active-passive failover can also be referred to as master-slave failover.

#### Active-active

In active-active, both servers are managing traffic, spreading the load between them.

If the servers are public-facing, the DNS would need to know about the public IPs of both servers. If the servers are internal-facing, application logic would need to know about both servers.

Active-active failover can also be referred to as master-master failover.

### Disadvantage(s): failover

*   Fail-over adds more hardware and additional complexity.
*   There is a potential for loss of data if the active system fails before any newly written data can be replicated to the passive.

### Replication

#### Master-slave and master-master

This topic is further discussed in the [Database](#database) section:

*   [Master-slave replication](#master-slave-replication)
*   [Master-master replication](#master-master-replication)

### Availability in numbers

Availability is often quantified by uptime (or downtime) as a percentage of time the service is available. Availability is generally measured in number of 9s--a service with 99.99% availability is described as having four 9s.

#### 99.9% availability - three 9s

| Duration            | Acceptable downtime|
|---------------------|--------------------|
| Downtime per year   | 8h 45min 57s       |
| Downtime per month  | 43m 49.7s          |
| Downtime per week   | 10m 4.8s           |
| Downtime per day    | 1m 26.4s           |

#### 99.99% availability - four 9s

| Duration            | Acceptable downtime|
|---------------------|--------------------|
| Downtime per year   | 52min 35.7s        |
| Downtime per month  | 4m 23s             |
| Downtime per week   | 1m 5s              |
| Downtime per day    | 8.6s               |

#### Availability in parallel vs in sequence

If a service consists of multiple components prone to failure, the service's overall availability depends on whether the components are in sequence or in parallel.

###### In sequence

Overall availability decreases when two components with availability < 100% are in sequence:

```
Availability (Total) = Availability (Foo) * Availability (Bar)
```

If both `Foo` and `Bar` each had 99.9% availability, their total availability in sequence would be 99.8%.

###### In parallel

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

### Source(s) and further reading

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

### Source(s) and further reading

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

### Layer 4 load balancing

Layer 4 load balancers look at info at the [transport layer](#communication) to decide how to distribute requests. Generally, this involves the source, destination IP addresses, and ports in the header, but not the contents of the packet. Layer 4 load balancers forward network packets to and from the upstream server, performing [Network Address Translation (NAT)](https://www.nginx.com/resources/glossary/layer-4-load-balancing/).

### Layer 7 load balancing

Layer 7 load balancers look at the [application layer](#communication) to decide how to distribute requests. This can involve contents of the header, message, and cookies. Layer 7 load balancers terminate network traffic, reads the message, makes a load-balancing decision, then opens a connection to the selected server. For example, a layer 7 load balancer can direct video traffic to servers that host videos while directing more sensitive user billing traffic to security-hardened servers.

At the cost of flexibility, layer 4 load balancing requires less time and computing resources than Layer 7, although the performance impact can be minimal on modern commodity hardware.

### Horizontal scaling

Load balancers can also help with horizontal scaling, improving performance and availability. Scaling out using commodity machines is more cost efficient and results in higher availability than scaling up a single server on more expensive hardware, called **Vertical Scaling**. It is also easier to hire for talent working on commodity hardware than it is for specialized enterprise systems.

#### Disadvantage(s): horizontal scaling

*   Scaling horizontally introduces complexity and involves cloning servers
    *   Servers should be stateless: they should not contain any user-related data like sessions or profile pictures
    *   Sessions can be stored in a centralized data store such as a [database](#database) (SQL, NoSQL) or a persistent [cache](#cache) (Redis, Memcached)
*   Downstream servers such as caches and databases need to handle more simultaneous connections as upstream servers scale out

### Disadvantage(s): load balancer

*   The load balancer can become a performance bottleneck if it does not have enough resources or if it is not configured properly.
*   Introducing a load balancer to help eliminate a single point of failure results in increased complexity.
*   A single load balancer is a single point of failure, configuring multiple load balancers further increases complexity.

### Source(s) and further reading

*   [NGINX architecture](https://www.nginx.com/blog/inside-nginx-how-we-designed-for-performance-scale/)
*   [HAProxy architecture guide](http://www.haproxy.org/download/1.2/doc/architecture.txt)
*   [Scalability](https://web.archive.org/web/20220530193911/https://www.lecloud.net/post/7295452622/scalability-for-dummies-part-1-clones)
*   [Wikipedia](https://en.wikipedia.org/wiki/Load_balancing_(computing))
*   [Layer 4 load balancing](https://www.nginx.com/resources/glossary/layer-4-load-balancing/)
*   [Layer 7 load balancing](https://www.nginx.com/resources/glossary/layer-7-load-balancing/)
*   [ELB listener config](http://docs.aws.amazon.com/elasticloadbalancing/latest/classic/elb-listener-config.html)

## Reverse Proxy (Web Server)

<p align="center">
  <img src="images/n41Azff.png">
  <br/>
  <i><a href=https://upload.wikimedia.org/wikipedia/commons/6/67/Reverse_proxy_h2g2bob.svg>Source: Wikipedia</a></i>
  <br/>
</p>

A reverse proxy is a web server that centralizes internal services and provides unified interfaces to the public. Requests from clients are forwarded to a server that can fulfill it before the reverse proxy returns the server's response to the client.

Additional benefits include:

*   **Increased security** - Hide information about backend servers, blacklist IPs, limit number of connections per client
*   **Increased scalability and flexibility** - Clients only see the reverse proxy's IP, allowing you to scale servers or change their configuration
*   **SSL termination** - Decrypt incoming requests and encrypt server responses so backend servers do not have to perform these potentially expensive operations
    *   Removes the need to install [X.509 certificates](https://en.wikipedia.org/wiki/X.509) on each server
*   **Compression** - Compress server responses
*   **Caching** - Return the response for cached requests
*   **Static content** - Serve static content directly
    *   HTML/CSS/JS
    *   Photos
    *   Videos
    *   Etc

### Load balancer vs reverse proxy

*   Deploying a load balancer is useful when you have multiple servers. Often, load balancers route traffic to a set of servers serving the same function.
*   Reverse proxies can be useful even with just one web server or application server, opening up the benefits described in the previous section.
*   Solutions such as NGINX and HAProxy can support both layer 7 reverse proxying and load balancing.

### Disadvantage(s): reverse proxy

*   Introducing a reverse proxy results in increased complexity.
*   A single reverse proxy is a single point of failure, configuring multiple reverse proxies (ie a [failover](https://en.wikipedia.org/wiki/Failover)) further increases complexity.

### Source(s) and further reading

*   [Reverse proxy vs load balancer](https://www.nginx.com/resources/glossary/reverse-proxy-vs-load-balancer/)
*   [NGINX architecture](https://www.nginx.com/blog/inside-nginx-how-we-designed-for-performance-scale/)
*   [HAProxy architecture guide](http://www.haproxy.org/download/1.2/doc/architecture.txt)
*   [Wikipedia](https://en.wikipedia.org/wiki/Reverse_proxy)

## Application Layer

<p align="center">
  <img src="images/yB5SYwm.png">
  <br/>
  <i><a href=http://lethain.com/introduction-to-architecting-systems-for-scale/#platform_layer>Source: Intro to architecting systems for scale</a></i>
</p>

Separating out the web layer from the application layer (also known as platform layer) allows you to scale and configure both layers independently. Adding a new API results in adding application servers without necessarily adding additional web servers. The **single responsibility principle** advocates for small and autonomous services that work together. Small teams with small services can plan more aggressively for rapid growth.

Workers in the application layer also help enable [asynchronism](#asynchronism).

### Microservices

Related to this discussion are [microservices](https://en.wikipedia.org/wiki/Microservices), which can be described as a suite of independently deployable, small, modular services. Each service runs a unique process and communicates through a well-defined, lightweight mechanism to serve a business goal. <sup><a href=https://smartbear.com/learn/api-design/what-are-microservices>1</a></sup>

Pinterest, for example, could have the following microservices: user profile, follower, feed, search, photo upload, etc.

### Service Discovery

Systems such as [Consul](https://www.consul.io/docs/index.html), [Etcd](https://coreos.com/etcd/docs/latest), and [Zookeeper](http://www.slideshare.net/sauravhaloi/introduction-to-apache-zookeeper) can help services find each other by keeping track of registered names, addresses, and ports. [Health checks](https://www.consul.io/intro/getting-started/checks.html) help verify service integrity and are often done using an [HTTP](#hypertext-transfer-protocol-http) endpoint. Both Consul and Etcd have a built in [key-value store](#key-value-store) that can be useful for storing config values and other shared data.

### Disadvantage(s): application layer

*   Adding an application layer with loosely coupled services requires a different approach from an architectural, operations, and process viewpoint (vs a monolithic system).
*   Microservices can add complexity in terms of deployments and operations.

### Source(s) and further reading

*   [Intro to architecting systems for scale](http://lethain.com/introduction-to-architecting-systems-for-scale)
*   [Crack the system design interview](http://www.puncsky.com/blog/2016-02-13-crack-the-system-design-interview)
*   [Service oriented architecture](https://en.wikipedia.org/wiki/Service-oriented_architecture)
*   [Introduction to Zookeeper](http://www.slideshare.net/sauravhaloi/introduction-to-apache-zookeeper)
*   [Here's what you need to know about building microservices](https://cloudncode.wordpress.com/2016/07/22/msa-getting-started/)

## Database

<p align="center">
  <img src="images/Xkm5CXz.png">
  <br/>
  <i><a href=https://www.youtube.com/watch?v=kKjm4ehYiMs>Source: Scaling up to your first 10 million users</a></i>
</p>

### Relational Database Management System (RDBMS)

A relational database like SQL is a collection of data items organized in tables.

**ACID** is a set of properties of relational database [transactions](https://en.wikipedia.org/wiki/Database_transaction).

*   **Atomicity** - Each transaction is all or nothing
*   **Consistency** - Any transaction will bring the database from one valid state to another
*   **Isolation** - Executing transactions concurrently has the same results as if the transactions were executed serially