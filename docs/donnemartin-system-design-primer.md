# System Design Primer: Your Guide to Building Scalable Systems

**Master system design for interviews and real-world projects with this comprehensive and community-driven resource. 
[Check out the original repo for more details!](https://github.com/donnemartin/system-design-primer)**

This primer provides a structured approach to understanding and building large-scale systems. It’s perfect for interview preparation and for becoming a better software engineer.

**Key Features:**

*   **Organized Resources:** Access a curated collection of resources on system design principles, avoiding scattered information.
*   **Community Driven:** Benefit from a continually updated, open-source project with contributions welcome from the community.
*   **Interview Prep:** Practice common system design interview questions with solutions, discussions, code, and diagrams, including object-oriented design questions.
*   **Anki Flashcards:** Utilize spaced repetition with provided Anki flashcard decks to help retain key concepts on the go.
*   **Interactive Coding Challenges:** Access the sister repo [**Interactive Coding Challenges**](https://github.com/donnemartin/interactive-coding-challenges) with additional Anki deck for Coding.

**Core Topics Covered:**

*   Scalability, Performance, and Reliability Principles
*   Database Design (SQL vs. NoSQL)
*   Caching Strategies
*   Load Balancing and CDNs
*   Message Queues and Asynchronous Systems
*   REST and RPC
*   Security Considerations

## Study Guide for System Design

This study guide provides suggested topics to review based on your interview timeline (short, medium, long) to help you.

|  | Short | Medium | Long |
| --- | --- | --- | --- |
| Get a broad understanding of how systems work  | :+1: | :+1: | :+1: |
| Read through a few articles in the [Company engineering blogs](#company-engineering-blogs) for the companies you are interviewing with | :+1: | :+1: | :+1: |
| Read through a few [Real world architectures](#real-world-architectures) | :+1: | :+1: | :+1: |
| Review [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question) | :+1: | :+1: | :+1: |
| Work through [System design interview questions with solutions](#system-design-interview-questions-with-solutions) | Some | Many | Most |
| Work through [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions) | Some | Many | Most |
| Review [Additional system design interview questions](#additional-system-design-interview-questions) | Some | Many | Most |

## How to Approach a System Design Interview

Follow these steps to lead the discussion:

1.  **Use Cases, Constraints, and Assumptions:** Define requirements, scope the problem, and clarify use cases and constraints.
2.  **High-Level Design:** Sketch the main components and connections.
3.  **Design Core Components:** Dive into details for each core component.
4.  **Scale the Design:** Address bottlenecks and discuss trade-offs, considering load balancing, caching, and database sharding.
5.  **Back-of-the-Envelope Calculations:** Refer to the [Appendix](#appendix) for handy resources like the Powers of Two table and latency numbers.

## System Design Interview Questions with Solutions

*   Design Pastebin.com (or Bit.ly)
*   Design the Twitter timeline and search (or Facebook feed and search)
*   Design a web crawler
*   Design Mint.com
*   Design the data structures for a social network
*   Design a key-value store for a search engine
*   Design Amazon's sales ranking by category feature
*   Design a system that scales to millions of users on AWS

## Object-Oriented Design Interview Questions with Solutions

*   Design a hash map
*   Design a least recently used cache
*   Design a call center
*   Design a deck of cards
*   Design a parking lot
*   Design a chat server
*   Design a circular array

**Note: This section is under development**

## Index of System Design Topics

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
    *   [Replication](#replication)
    *   [Availability in numbers](#availability-in-numbers)
*   [Domain name system](#domain-name-system)
*   [Content delivery network](#content-delivery-network)
    *   [Push CDNs](#push-cdns)
    *   [Pull CDNs](#pull-cdns)
*   [Load balancer](#load-balancer)
    *   [Active-passive](#active-passive)
    *   [Active-active](#active-active)
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

**[Contribute](#contributing) to this valuable resource and level up your system design skills!**