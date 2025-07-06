# System Design Primer: Your Comprehensive Guide to Building Scalable Systems

**Conquer system design interviews and master the art of building large-scale systems with this organized collection of resources.**  [Explore the original repository](https://github.com/donnemartin/system-design-primer)

## Key Features

*   **Organized Resources:** A curated collection of articles, diagrams, and solutions covering key system design topics.
*   **Interview Prep Focus:** Prepare for system design interviews with sample questions, discussions, code, and diagrams, with a special focus on solutions.
*   **Comprehensive Coverage:**  Understand the core principles of system design, including scalability, performance, consistency, and availability.
*   **Community-Driven:**  Benefit from an open-source project where contributions and translations are welcome.
*   **Anki Flashcards:** Reinforce your knowledge with spaced-repetition flashcard decks for system design concepts and exercises.

## Table of Contents

*   [Motivation](#motivation)
*   [Anki flashcards](#anki-flashcards)
*   [Contributing](#contributing)
*   [Index of system design topics](#index-of-system-design-topics)
*   [Study guide](#study-guide)
*   [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question)
*   [System design interview questions with solutions](#system-design-interview-questions-with-solutions)
*   [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions)
*   [System design topics: start here](#system-design-topics-start-here)
*   [Performance vs scalability](#performance-vs-scalability)
*   [Latency vs throughput](#latency-vs-throughput)
*   [Availability vs consistency](#availability-vs-consistency)
*   [Consistency patterns](#consistency-patterns)
*   [Availability patterns](#availability-patterns)
*   [Domain name system](#domain-name-system)
*   [Content delivery network](#content-delivery-network)
*   [Load balancer](#load-balancer)
*   [Reverse proxy (web server)](#reverse-proxy-web-server)
*   [Application layer](#application-layer)
*   [Database](#database)
*   [Cache](#cache)
*   [Asynchronism](#asynchronism)
*   [Communication](#communication)
*   [Security](#security)
*   [Appendix](#appendix)
*   [Under development](#under-development)
*   [Credits](#credits)
*   [Contact info](#contact-info)
*   [License](#license)

## Motivation

This project provides a structured approach to learning system design, preparing you for interviews and helping you build scalable systems.

### Learn how to design large-scale systems

Learning how to design scalable systems will help you become a better engineer.

System design is a broad topic.  There is a **vast amount of resources scattered throughout the web** on system design principles.

This repo is an **organized collection** of resources to help you learn how to build systems at scale.

### Learn from the open source community

This is a continually updated, open source project.

[Contributions](#contributing) are welcome!

### Prep for the system design interview

In addition to coding interviews, system design is a **required component** of the **technical interview process** at many tech companies.

**Practice common system design interview questions** and **compare** your results with **sample solutions**: discussions, code, and diagrams.

Additional topics for interview prep:

*   [Study guide](#study-guide)
*   [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question)
*   [System design interview questions, **with solutions**](#system-design-interview-questions-with-solutions)
*   [Object-oriented design interview questions, **with solutions**](#object-oriented-design-interview-questions-with-solutions)
*   [Additional system design interview questions](#additional-system-design-interview-questions)

## Anki flashcards

The provided [Anki flashcard decks](https://apps.ankiweb.net/) use spaced repetition to help you retain key system design concepts.

*   [System design deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/System%20Design.apkg)
*   [System design exercises deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/System%20Design%20Exercises.apkg)
*   [Object oriented design exercises deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/OO%20Design.apkg)

Great for use while on-the-go.

### Coding Resource: Interactive Coding Challenges

Looking for resources to help you prep for the [**Coding Interview**](https://github.com/donnemartin/interactive-coding-challenges)?

Check out the sister repo [**Interactive Coding Challenges**](https://github.com/donnemartin/interactive-coding-challenges), which contains an additional Anki deck:

*   [Coding deck](https://github.com/donnemartin/interactive-coding-challenges/tree/master/anki_cards/Coding.apkg)

## Contributing

> Learn from the community.

Feel free to submit pull requests to help:

*   Fix errors
*   Improve sections
*   Add new sections
*   [Translate](https://github.com/donnemartin/system-design-primer/issues/28)

Content that needs some polishing is placed [under development](#under-development).

Review the [Contributing Guidelines](CONTRIBUTING.md).

## Index of system design topics

> Summaries of various system design topics, including pros and cons.  **Everything is a trade-off**.
>
> Each section contains links to more in-depth resources.

*   [System design topics: start here](#system-design-topics-start-here)
    *   [Step 1: Review the scalability video lecture](#step-1-review-the-scalability-video-lecture)
    *   [Step 2: Review the scalability article](#step-2-review-the-scalability-article)
    *   [Next steps](#next-steps)
*   [Performance vs scalability](#performance-vs-scalability)
*   [Latency vs throughput](#latency-vs-throughput)
*   [Availability vs consistency](#availability-vs-consistency)
    *   [CAP theorem](#cap-theorem)
        *   [CP - consistency and partition tolerance](#cp---consistency-and-partition-tolerance)
        *   [AP - availability and partition tolerance](#ap---availability-and-partition-tolerance)
*   [Consistency patterns](#consistency-patterns)
    *   [Weak consistency](#weak-consistency)
    *   [Eventual consistency](#eventual-consistency)
    *   [Strong consistency](#strong-consistency)
*   [Availability patterns](#availability-patterns)
    *   [Fail-over](#fail-over)
        *   [Active-passive](#active-passive)
        *   [Active-active](#active-active)
    *   [Replication](#replication)
    *   [Availability in numbers](#availability-in-numbers)
*   [Domain name system](#domain-name-system)
*   [Content delivery network](#content-delivery-network)
    *   [Push CDNs](#push-cdns)
    *   [Pull CDNs](#pull-cdns)
*   [Load balancer](#load-balancer)
    *   [Layer 4 load balancing](#layer-4-load-balancing)
    *   [Layer 7 load balancing](#layer-7-load-balancing)
    *   [Horizontal scaling](#horizontal-scaling)
*   [Reverse proxy (web server)](#reverse-proxy-web-server)
    *   [Load balancer vs reverse proxy](#load-balancer-vs-reverse-proxy)
*   [Application layer](#application-layer)
    *   [Microservices](#microservices)
    *   [Service discovery](#service-discovery)
*   [Database](#database)
    *   [Relational database management system (RDBMS)](#relational-database-management-system-rdbms)
        *   [Master-slave replication](#master-slave-replication)
        *   [Master-master replication](#master-master-replication)
        *   [Federation](#federation)
        *   [Sharding](#sharding)
        *   [Denormalization](#denormalization)
        *   [SQL tuning](#sql-tuning)
    *   [NoSQL](#nosql)
        *   [Key-value store](#key-value-store)
        *   [Document store](#document-store)
        *   [Wide column store](#wide-column-store)
        *   [Graph Database](#graph-database)
    *   [SQL or NoSQL](#sql-or-nosql)
*   [Cache](#cache)
    *   [Client caching](#client-caching)
    *   [CDN caching](#cdn-caching)
    *   [Web server caching](#web-server-caching)
    *   [Database caching](#database-caching)
    *   [Application caching](#application-caching)
    *   [Caching at the database query level](#caching-at-the-database-query-level)
    *   [Caching at the object level](#caching-at-the-object-level)
    *   [When to update the cache](#when-to-update-the-cache)
        *   [Cache-aside](#cache-aside)
        *   [Write-through](#write-through)
        *   [Write-behind (write-back)](#write-behind-write-back)
        *   [Refresh-ahead](#refresh-ahead)
*   [Asynchronism](#asynchronism)
    *   [Message queues](#message-queues)
    *   [Task queues](#task-queues)
    *   [Back pressure](#back-pressure)
*   [Communication](#communication)
    *   [Transmission control protocol (TCP)](#transmission-control-protocol-tcp)
    *   [User datagram protocol (UDP)](#user-datagram-protocol-udp)
    *   [Remote procedure call (RPC)](#remote-procedure-call-rpc)
    *   [Representational state transfer (REST)](#representational-state-transfer-rest)
*   [Security](#security)
*   [Appendix](#appendix)
    *   [Powers of two table](#powers-of-two-table)
    *   [Latency numbers every programmer should know](#latency-numbers-every-programmer-should-know)
    *   [Additional system design interview questions](#additional-system-design-interview-questions)
    *   [Real world architectures](#real-world-architectures)
    *   [Company architectures](#company-architectures)
    *   [Company engineering blogs](#company-engineering-blogs)
*   [Under development](#under-development)
*   [Credits](#credits)
*   [Contact info](#contact-info)
*   [License](#license)

## Study guide

> Suggested topics to review based on your interview timeline (short, medium, long).

**Q: For interviews, do I need to know everything here?**

**A: No, you don't need to know everything here to prepare for the interview**.

What you are asked in an interview depends on variables such as:

*   How much experience you have
*   What your technical background is
*   What positions you are interviewing for
*   Which companies you are interviewing with
*   Luck

More experienced candidates are generally expected to know more about system design. Architects or team leads might be expected to know more than individual contributors. Top tech companies are likely to have one or more design interview rounds.

Start broad and go deeper in a few areas. It helps to know a little about various key system design topics. Adjust the following guide based on your timeline, experience, what positions you are interviewing for, and which companies you are interviewing with.

*   **Short timeline** - Aim for **breadth** with system design topics. Practice by solving **some** interview questions.
*   **Medium timeline** - Aim for **breadth** and **some depth** with system design topics. Practice by solving **many** interview questions.
*   **Long timeline** - Aim for **breadth** and **more depth** with system design topics. Practice by solving **most** interview questions.

| | Short | Medium | Long |
|---|---|---|---|
| Read through the [System design topics](#index-of-system-design-topics) to get a broad understanding of how systems work | :+1: | :+1: | :+1: |
| Read through a few articles in the [Company engineering blogs](#company-engineering-blogs) for the companies you are interviewing with | :+1: | :+1: | :+1: |
| Read through a few [Real world architectures](#real-world-architectures) | :+1: | :+1: | :+1: |
| Review [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question) | :+1: | :+1: | :+1: |
| Work through [System design interview questions with solutions](#system-design-interview-questions-with-solutions) | Some | Many | Most |
| Work through [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions) | Some | Many | Most |
| Review [Additional system design interview questions](#additional-system-design-interview-questions) | Some | Many | Most |

## How to approach a system design interview question

> How to tackle a system design interview question.

The system design interview is an **open-ended conversation**. You are expected to lead it.

You can use the following steps to guide the discussion. To help solidify this process, work through the [System design interview questions with solutions](#system-design-interview-questions-with-solutions) section using the following steps.

### Step 1: Outline use cases, constraints, and assumptions

Gather requirements and scope the problem. Ask questions to clarify use cases and constraints. Discuss assumptions.

*   Who is going to use it?
*   How are they going to use it?
*   How many users are there?
*   What does the system do?
*   What are the inputs and outputs of the system?
*   How much data do we expect to handle?
*   How many requests per second do we expect?
*   What is the expected read to write ratio?

### Step 2: Create a high level design

Outline a high level design with all important components.

*   Sketch the main components and connections
*   Justify your ideas

### Step 3: Design core components

Dive into details for each core component. For example, if you were asked to [design a url shortening service](solutions/system_design/pastebin/README.md), discuss:

*   Generating and storing a hash of the full url
    *   [MD5](solutions/system_design/pastebin/README.md) and [Base62](solutions/system_design/pastebin/README.md)
    *   Hash collisions
    *   SQL or NoSQL
    *   Database schema
*   Translating a hashed url to the full url
    *   Database lookup
*   API and object-oriented design

### Step 4: Scale the design

Identify and address bottlenecks, given the constraints. For example, do you need the following to address scalability issues?

*   Load balancer
*   Horizontal scaling
*   Caching
*   Database sharding

Discuss potential solutions and trade-offs. Everything is a trade-off. Address bottlenecks using [principles of scalable system design](#index-of-system-design-topics).

### Back-of-the-envelope calculations

You might be asked to do some estimates by hand. Refer to the [Appendix](#appendix) for the following resources:

*   [Use back of the envelope calculations](http://highscalability.com/blog/2011/1/26/google-pro-tip-use-back-of-the-envelope-calculations-to-choo.html)
*   [Powers of two table](#powers-of-two-table)
*   [Latency numbers every programmer should know](#latency-numbers-every-programmer-should-know)

### Source(s) and further reading

Check out the following links to get a better idea of what to expect:

*   [How to ace a systems design interview](https://www.palantir.com/2011/10/how-to-rock-a-systems-design-interview/)
*   [The system design interview](http://www.hiredintech.com/system-design)
*   [Intro to Architecture and Systems Design Interviews](https://www.youtube.com/watch?v=ZgdS0EUmn70)
*   [System design template](https://leetcode.com/discuss/career/229177/My-System-Design-Template)

## System design interview questions with solutions

> Common system design interview questions with sample discussions, code, and diagrams.
>
> Solutions linked to content in the `solutions/` folder.

| Question |  |
|---|---|
| Design Pastebin.com (or Bit.ly) | [Solution](solutions/system_design/pastebin/README.md) |
| Design the Twitter timeline and search (or Facebook feed and search) | [Solution](solutions/system_design/twitter/README.md) |
| Design a web crawler | [Solution](solutions/system_design/web_crawler/README.md) |
| Design Mint.com | [Solution](solutions/system_design/mint/README.md) |
| Design the data structures for a social network | [Solution](solutions/system_design/social_graph/README.md) |
| Design a key-value store for a search engine | [Solution](solutions/system_design/query_cache/README.md) |
| Design Amazon's sales ranking by category feature | [Solution](solutions/system_design/sales_rank/README.md) |
| Design a system that scales to millions of users on AWS | [Solution](solutions/system_design/scaling_aws/README.md) |
| Add a system design question | [Contribute](#contributing) |

### Design Pastebin.com (or Bit.ly)

[View exercise and solution](solutions/system_design/pastebin/README.md)

### Design the Twitter timeline and search (or Facebook feed and search)

[View exercise and solution](solutions/system_design/twitter/README.md)

### Design a web crawler

[View exercise and solution](solutions/system_design/web_crawler/README.md)

### Design Mint.com

[View exercise and solution](solutions/system_design/mint/README.md)

### Design the data structures for a social network

[View exercise and solution](solutions/system_design/social_graph/README.md)

### Design a key-value store for a search engine

[View exercise and solution](solutions/system_design/query_cache/README.md)

### Design Amazon's sales ranking by category feature

[View exercise and solution](solutions/system_design/sales_rank/README.md)

### Design a system that scales to millions of users on AWS

[View exercise and solution](solutions/system_design/scaling_aws/README.md)

## Object-oriented design interview questions with solutions

> Common object-oriented design interview questions with sample discussions, code, and diagrams.
>
> Solutions linked to content in the `solutions/` folder.

>**Note: This section is under development**

| Question |  |
|---|---|
| Design a hash map | [Solution](solutions/object_oriented_design/hash_table/hash_map.ipynb)  |
| Design a least recently used cache | [Solution](solutions/object_oriented_design/lru_cache/lru_cache.ipynb)  |
| Design a call center | [Solution](solutions/object_oriented_design/call_center/call_center.ipynb)  |
| Design a deck of cards | [Solution](solutions/object_oriented_design/deck_of_cards/deck_of_cards.ipynb)  |
| Design a parking lot | [Solution](solutions/object_oriented_design/parking_lot/parking_lot.ipynb)  |
| Design a chat server | [Solution](solutions/object_oriented_design/online_chat/online_chat.ipynb)  |
| Design a circular array | [Contribute](#contributing)  |
| Add an object-oriented design question | [Contribute](#contributing) |

## System design topics: start here

New to system design?

First, you'll need a basic understanding of common principles, learning about what they are, how they are used, and their pros and cons.

### Step 1: Review the scalability video lecture

[Scalability Lecture at Harvard](https://www.youtube.com/watch?v=-W9F__D3oY4)

*   Topics covered:
    *   Vertical scaling
    *   Horizontal scaling
    *   Caching
    *   Load balancing
    *   Database replication
    *   Database partitioning

### Step 2: Review the scalability article

[Scalability](https://web.archive.org/web/20221030091841/http://www.lecloud.net/tagged/scalability/chrono)

*   Topics covered:
    *   [Clones](https://web.archive.org/web/20220530193911/https://www.lecloud.net/post/7295452622/scalability-for-dummies-part-1-clones)
    *   [Databases](https://web.archive.org/web/20220602114024/https://www.lecloud.net/post/7994751381/scalability-for-dummies-part-2-database)
    *   [Caches](https://web.archive.org/web/20230126233752/https://www.lecloud.net/post/9246290032/scalability-for-dummies-part-3-cache)
    *   [Asynchronism](https://web.archive.org/web/20220926171507/https://www.lecloud.net/post/9699762917/scalability-for-dummies-part-4-asynchronism)

### Next steps

Next, we'll look at high-level trade-offs:

*   **Performance** vs **scalability**
*   **Latency** vs **throughput**
*   **Availability** vs **consistency**

Keep in mind that **everything is a trade-off**.

Then we'll dive into more specific topics such as DNS, CDNs, and load balancers.

## Performance vs scalability

A service is **scalable** if it results in increased **performance** in a manner proportional to resources added. Generally, increasing performance means serving more units of work, but it can also be to handle larger units of work, such as when datasets grow.<sup><a href=http://www.allthingsdistributed.com/2006/03/a_word_on_scalability.html>1</a></sup>

Another way to look at performance vs scalability:

*   If you have a **performance** problem, your system is slow for a single user.
*   If you have a **scalability** problem, your system is fast for a single user but slow under heavy load.

### Source(s) and further reading

*   [A word on scalability](http://www.allthingsdistributed.com/2006/03/a_word_on_scalability.html)
*   [Scalability, availability, stability, patterns](http://www.slideshare.net/jboner/scalability-availability-stability-patterns/)

## Latency vs throughput

**Latency** is the time to perform some action or to produce some result.

**Throughput** is the number of such actions or results per unit of time.

Generally, you should aim for **maximal throughput** with **acceptable latency**.

### Source(s) and further reading

*   [Understanding latency vs throughput](https://community.cadence.com/cadence_blogs_8/b/fv/posts/understanding-latency-vs-throughput)

## Availability vs consistency

### CAP theorem

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

## Consistency patterns

With multiple copies of the same data, we are faced with options on how to synchronize them so clients have a consistent view of the data. Recall the definition of consistency from the [CAP theorem](#cap-theorem) - Every read receives the most recent write or an error.

### Weak consistency

After a write, reads may or may not see it. A best effort approach is taken.

This approach is seen in systems such as memcached. Weak consistency works well in real time use cases such as VoIP, video chat, and realtime multiplayer games. For example, if you are on a phone call and lose reception for a few seconds, when you regain connection you do not hear what was spoken during connection loss.

### Eventual consistency

After a write, reads will eventually see it (typically within milliseconds). Data is replicated asynchronously.

This approach is seen in systems such as DNS and email. Eventual consistency works well in highly available systems.

### Strong consistency

After a write, reads will see it. Data is replicated synchronously.

This approach is seen in file systems and RDBMSes. Strong consistency works well in systems that need transactions.

### Source(s) and further reading

*   [Transactions across data centers](http://snarfed.org/transactions_across_datacenters_io.html)

## Availability patterns

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

| Duration | Acceptable downtime |
|---|---|
| Downtime per year | 8h 45min 57s |
| Downtime per month | 43m 49.7s |
| Downtime per week | 10m 4.8s |
| Downtime per day | 1m 26.4s |

#### 99.99% availability - four 9s

| Duration | Acceptable downtime |
|---|---|
| Downtime per year | 52min 35.7s |
| Downtime per month | 4m 23s |
| Downtime per week | 1m 5s |
| Downtime per day | 8.6s |

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

## Domain name system

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

## Content delivery network

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
*   [The differences between push and pull CDNs](http://www.travelblogadvice.com/technical/the-differences-