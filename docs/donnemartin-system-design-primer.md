# System Design Primer: Your Guide to Designing Large-Scale Systems

**Master the art of building scalable systems and ace your system design interviews with this comprehensive, community-driven resource.**  [Explore the original repository](https://github.com/donnemartin/system-design-primer) for a deeper dive into system design concepts.

---

## Key Features:

*   **Comprehensive Coverage:** Dive deep into essential system design topics.
*   **Interview Prep:** Master system design interview questions with solutions.
*   **Community-Driven:** Benefit from a continually updated, open-source project.
*   **Practical Resources:** Utilize Anki flashcards for effective knowledge retention.
*   **Real-World Examples:** Learn from detailed analyses of real-world system architectures.

---

## Table of Contents

*   **[Motivation](#motivation)**
    *   Learn how to design large-scale systems
    *   Prep for the system design interview
*   **[Anki Flashcards](#anki-flashcards)**
    *   System design deck
    *   System design exercises deck
    *   Object-oriented design exercises deck
*   **[Contributing](#contributing)**
*   **[Index of System Design Topics](#index-of-system-design-topics)**
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
*   **[Study Guide](#study-guide)**
*   **[How to Approach a System Design Interview Question](#how-to-approach-a-system-design-interview-question)**
*   **[System Design Interview Questions with Solutions](#system-design-interview-questions-with-solutions)**
*   **[Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions)**
*   [Under Development](#under-development)
*   [Credits](#credits)
*   [Contact Info](#contact-info)
*   [License](#license)

---

## Motivation

This repository serves as a comprehensive guide to system design, offering resources to help you:

*   **Learn how to design large-scale systems:** Understand the core principles of building scalable and reliable systems.
*   **Prep for the system design interview:** Prepare for the system design interviews with practice questions and solutions.

---

## Anki Flashcards

Enhance your retention with provided Anki flashcard decks using spaced repetition:

*   [System design deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/System%20Design.apkg)
*   [System design exercises deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/System%20Design%20Exercises.apkg)
*   [Object oriented design exercises deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/OO%20Design.apkg)

---

## Contributing

This project is open-source and welcomes contributions! Help improve the guide by:

*   Fixing errors
*   Improving sections
*   Adding new sections
*   [Translating](https://github.com/donnemartin/system-design-primer/issues/28)

---

## Index of System Design Topics

*   **[System design topics: start here](#system-design-topics-start-here)**
    *   [Step 1: Review the scalability video lecture](#step-1-review-the-scalability-video-lecture)
    *   [Step 2: Review the scalability article](#step-2-review-the-scalability-article)
    *   [Next steps](#next-steps)
*   **[Performance vs scalability](#performance-vs-scalability)**
*   **[Latency vs throughput](#latency-vs-throughput)**
*   **[Availability vs consistency](#availability-vs-consistency)**
    *   [CAP theorem](#cap-theorem)
        *   [CP - consistency and partition tolerance](#cp---consistency-and-partition-tolerance)
        *   [AP - availability and partition tolerance](#ap---availability-and-partition-tolerance)
*   **[Consistency patterns](#consistency-patterns)**
    *   [Weak consistency](#weak-consistency)
    *   [Eventual consistency](#eventual-consistency)
    *   [Strong consistency](#strong-consistency)
*   **[Availability patterns](#availability-patterns)**
    *   [Fail-over](#fail-over)
    *   [Replication](#replication)
    *   [Availability in numbers](#availability-in-numbers)
*   **[Domain name system](#domain-name-system)**
*   **[Content delivery network](#content-delivery-network)**
    *   [Push CDNs](#push-cdns)
    *   [Pull CDNs](#pull-cdns)
*   **[Load balancer](#load-balancer)**
    *   [Active-passive](#active-passive)
    *   [Active-active](#active-active)
    *   [Layer 4 load balancing](#layer-4-load-balancing)
    *   [Layer 7 load balancing](#layer-7-load-balancing)
    *   [Horizontal scaling](#horizontal-scaling)
*   **[Reverse proxy (web server)](#reverse-proxy-web-server)**
    *   [Load balancer vs reverse proxy](#load-balancer-vs-reverse-proxy)
*   **[Application layer](#application-layer)**
    *   [Microservices](#microservices)
    *   [Service discovery](#service-discovery)
*   **[Database](#database)**
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
*   **[Cache](#cache)**
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
*   **[Asynchronism](#asynchronism)**
    *   [Message queues](#message-queues)
    *   [Task queues](#task-queues)
    *   [Back pressure](#back-pressure)
*   **[Communication](#communication)**
    *   [Transmission control protocol (TCP)](#transmission-control-protocol-tcp)
    *   [User datagram protocol (UDP)](#user-datagram-protocol-udp)
    *   [Remote procedure call (RPC)](#remote-procedure-call-rpc)
    *   [Representational state transfer (REST)](#representational-state-transfer-rest)
*   **[Security](#security)**
*   **[Appendix](#appendix)**
    *   [Powers of two table](#powers-of-two-table)
    *   [Latency numbers every programmer should know](#latency-numbers-every-programmer-should-know)
    *   [Additional system design interview questions](#additional-system-design-interview-questions)
    *   [Real world architectures](#real-world-architectures)
    *   [Company architectures](#company-architectures)
    *   [Company engineering blogs](#company-engineering-blogs)
*   **[Under development](#under-development)**
*   **[Credits](#credits)**
*   **[Contact Info](#contact-info)**
*   **[License](#license)**

---

## Study Guide

A recommended study guide based on your interview timeline.

*   **Short Timeline**: Aim for breadth with system design topics. Practice by solving some interview questions.
*   **Medium Timeline**: Aim for breadth and some depth with system design topics. Practice by solving many interview questions.
*   **Long Timeline**: Aim for breadth and more depth with system design topics. Practice by solving most interview questions.

| Task | Short | Medium | Long |
|---|---|---|---|
| Read through the [System design topics](#index-of-system-design-topics) to get a broad understanding of how systems work | :+1: | :+1: | :+1: |
| Read through a few articles in the [Company engineering blogs](#company-engineering-blogs) for the companies you are interviewing with | :+1: | :+1: | :+1: |
| Read through a few [Real world architectures](#real-world-architectures) | :+1: | :+1: | :+1: |
| Review [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question) | :+1: | :+1: | :+1: |
| Work through [System design interview questions with solutions](#system-design-interview-questions-with-solutions) | Some | Many | Most |
| Work through [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions) | Some | Many | Most |
| Review [Additional system design interview questions](#additional-system-design-interview-questions) | Some | Many | Most |

---

## How to Approach a System Design Interview Question

Follow these steps to ace your system design interviews:

1.  **Outline use cases, constraints, and assumptions:** Gather requirements, scope the problem, and discuss assumptions.
2.  **Create a high-level design:** Sketch the main components and connections, justifying your ideas.
3.  **Design core components:** Dive into details for each core component.
4.  **Scale the design:** Identify and address bottlenecks, discussing potential solutions and trade-offs.
---
## System Design Interview Questions with Solutions

Get hands-on experience with common system design interview questions.

*   [Design Pastebin.com (or Bit.ly)](solutions/system_design/pastebin/README.md)
*   [Design the Twitter timeline and search (or Facebook feed and search)](solutions/system_design/twitter/README.md)
*   [Design a web crawler](solutions/system_design/web_crawler/README.md)
*   [Design Mint.com](solutions/system_design/mint/README.md)
*   [Design the data structures for a social network](solutions/system_design/social_graph/README.md)
*   [Design a key-value store for a search engine](solutions/system_design/query_cache/README.md)
*   [Design Amazon's sales ranking by category feature](solutions/system_design/sales_rank/README.md)
*   [Design a system that scales to millions of users on AWS](solutions/system_design/scaling_aws/README.md)
---
## Object-oriented design interview questions with solutions

>**Note: This section is under development**

Get hands-on experience with common system design interview questions.

*   Design a hash map
*   Design a least recently used cache
*   Design a call center
*   Design a deck of cards
*   Design a parking lot
*   Design a chat server

---
## Appendix
*   **[Powers of two table](#powers-of-two-table)**
*   **[Latency numbers every programmer should know](#latency-numbers-every-programmer-should-know)**
*   **[Additional system design interview questions](#additional-system-design-interview-questions)**
*   **[Real world architectures](#real-world-architectures)**
*   **[Company architectures](#company-architectures)**
*   **[Company engineering blogs](#company-engineering-blogs)**

---

## Credits

The contributions of the community and the following resources are gratefully acknowledged in this repository.

*   [Hired in tech](http://www.hiredintech.com/system-design/the-system-design-process/)
*   [Cracking the coding interview](https://www.amazon.com/dp/0984782850/)
*   [High scalability](http://highscalability.com/)
*   [checkcheckzz/system-design-interview](https://github.com/checkcheckzz/system-design-interview)
*   [shashank88/system_design](https://github.com/shashank88/system_design)
*   [mmcgrana/services-engineering](https://github.com/mmcgrana/services-engineering)
*   [System design cheat sheet](https://gist.github.com/vasanthk/485d1c25737e8e72759f)
*   [A distributed systems reading list](http://dancres.github.io/Pages/)
*   [Cracking the system design interview](http://www.puncsky.com/blog/2016-02-13-crack-the-system-design-interview)

---

## Contact Info

Feel free to contact me to discuss any issues, questions, or comments.

My contact info can be found on my [GitHub page](https://github.com/donnemartin).

---

## License

*I am providing code and resources in this repository to you under an open source license. Because this is my personal repository, the license you receive to my code and resources is from me and not my employer (Facebook).*

    Copyright 2017 Donne Martin

    Creative Commons Attribution 4.0 International License (CC BY 4.0)

    http://creativecommons.org/licenses/by/4.0/