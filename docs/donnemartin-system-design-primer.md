# System Design Primer: Ace Your System Design Interview!

**Learn how to design large-scale systems and master your system design interview prep with this comprehensive, open-source resource. [Explore the original repository](https://github.com/donnemartin/system-design-primer).**

## Key Features

*   **Comprehensive Coverage:** Dive into a wide range of system design topics, from fundamental concepts to advanced architectures.
*   **Interview Prep Focus:** Get ready for your system design interview with detailed explanations, practical examples, and sample solutions.
*   **Community-Driven:** Benefit from a continually updated, open-source project with contributions from a vast community of engineers.
*   **Anki Flashcards:** Use the provided [Anki flashcard decks](https://apps.ankiweb.net/) to reinforce key system design concepts.
*   **Real-World Architectures:** Learn from the design of popular services like Twitter, Instagram, and Netflix.

## Overview

This repository provides a comprehensive guide to system design, covering the key concepts, principles, and real-world examples needed to excel in system design interviews and become a better engineer. This is the go-to resource for system design, with organized collections, resources and an emphasis on the [system design interview](https://github.com/donnemartin/system-design-primer)

**Why System Design Matters:**

*   **Become a Better Engineer:**  Understand how to build robust, scalable, and efficient systems.
*   **Ace the Technical Interview:** Successfully navigate system design questions, a critical part of the interview process at many top tech companies.

## Getting Started

### Study Guide

Follow these steps to maximize your preparation:

*   **Breadth First:** Review the [System Design Topics](#index-of-system-design-topics) for a broad understanding.
*   **Real-World Insights:** Explore [Company Engineering Blogs](#company-engineering-blogs) to see how real systems work.
*   **Approach the Interview:**  Understand [How to Approach a System Design Interview Question](#how-to-approach-a-system-design-interview-question).
*   **Practice, Practice, Practice:** Work through [System Design Interview Questions with Solutions](#system-design-interview-questions-with-solutions) and [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions).
*   **Deepen Your Knowledge:** Review [Additional System Design Interview Questions](#additional-system-design-interview-questions) to further your knowledge.

## Index of System Design Topics

(Summaries of various system design topics, including pros and cons.  **Everything is a trade-off**.)

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

## How to Approach a System Design Interview Question

1.  **Understand the Problem:** Gather requirements, clarify use cases, and define constraints.
2.  **Design a High-Level Architecture:** Outline key components and their interactions.
3.  **Dive into Core Components:** Detail the design of essential elements, such as databases and caches.
4.  **Address Scalability:** Identify bottlenecks and implement solutions like load balancing and sharding.

## System Design Interview Questions with Solutions

(Common system design interview questions with sample discussions, code, and diagrams.)

| Question                                                     | Solution                                                                   |
| :----------------------------------------------------------- | :------------------------------------------------------------------------- |
| Design Pastebin.com (or Bit.ly)                            | [Solution](solutions/system_design/pastebin/README.md)                      |
| Design the Twitter timeline and search (or Facebook feed and search) | [Solution](solutions/system_design/twitter/README.md)                       |
| Design a web crawler                                         | [Solution](solutions/system_design/web_crawler/README.md)                   |
| Design Mint.com                                              | [Solution](solutions/system_design/mint/README.md)                         |
| Design the data structures for a social network              | [Solution](solutions/system_design/social_graph/README.md)                 |
| Design a key-value store for a search engine               | [Solution](solutions/system_design/query_cache/README.md)                   |
| Design Amazon's sales ranking by category feature          | [Solution](solutions/system_design/sales_rank/README.md)                    |
| Design a system that scales to millions of users on AWS    | [Solution](solutions/system_design/scaling_aws/README.md)                   |
| Add a system design question | [Contribute](#contributing) |

## Object-oriented design interview questions with solutions

(Common object-oriented design interview questions with sample discussions, code, and diagrams.)

>**Note: This section is under development**

| Question | |
|---|---|
| Design a hash map | [Solution](solutions/object_oriented_design/hash_table/hash_map.ipynb)  |
| Design a least recently used cache | [Solution](solutions/object_oriented_design/lru_cache/lru_cache.ipynb)  |
| Design a call center | [Solution](solutions/object_oriented_design/call_center/call_center.ipynb)  |
| Design a deck of cards | [Solution](solutions/object_oriented_design/deck_of_cards/deck_of_cards.ipynb)  |
| Design a parking lot | [Solution](solutions/object_oriented_design/parking_lot/parking_lot.ipynb)  |
| Design a chat server | [Solution](solutions/object_oriented_design/online_chat/online_chat.ipynb)  |
| Design a circular array | [Contribute](#contributing)  |
| Add an object-oriented design question | [Contribute](#contributing) |

## Contributing

This is a collaborative project!  Help improve the guide by:

*   Fixing errors and improving existing sections
*   Adding new topics and content
*   [Translating](https://github.com/donnemartin/system-design-primer/issues/28) the guide into other languages

Review the [Contributing Guidelines](CONTRIBUTING.md).

## Additional Resources

*   [Additional System Design Interview Questions](#additional-system-design-interview-questions)
*   [Real World Architectures](#real-world-architectures)
*   [Company Architectures](#company-architectures)
*   [Company Engineering Blogs](#company-engineering-blogs)
* [Anki flashcard decks](https://apps.ankiweb.net/)

---

**(Links to credits, contact info, and license will remain in your final README)**