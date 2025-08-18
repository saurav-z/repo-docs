# System Design Primer: Your Guide to Building Scalable Systems

**Master the art of designing large-scale systems and ace your system design interviews with this comprehensive resource.**  [Explore the original repository](https://github.com/donnemartin/system-design-primer) for in-depth knowledge and practical examples.

## Key Features:

*   **Comprehensive Coverage:** Dive deep into the essential principles, concepts, and trade-offs of system design.
*   **Organized Resources:**  Access a curated collection of resources to learn how to build scalable systems.
*   **Interview Preparation:**  Prepare for technical interviews with practice questions, sample solutions, and object-oriented design questions.
*   **Community-Driven:**  Benefit from a continually updated, open-source project, with contributions welcome.
*   **Anki Flashcards:**  Utilize spaced repetition to master key concepts with provided Anki decks.

## Table of Contents

1.  [Motivation](#motivation)
    *   [Learn how to design large-scale systems](#learn-how-to-design-large-scale-systems)
    *   [Learn from the open source community](#learn-from-the-open-source-community)
    *   [Prep for the system design interview](#prep-for-the-system-design-interview)
2.  [Anki flashcards](#anki-flashcards)
3.  [Contributing](#contributing)
4.  [Index of system design topics](#index-of-system-design-topics)
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
5.  [Study guide](#study-guide)
6.  [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question)
7.  [System design interview questions with solutions](#system-design-interview-questions-with-solutions)
8.  [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions)
9.  [System design topics: start here](#system-design-topics-start-here)
    *   [Step 1: Review the scalability video lecture](#step-1-review-the-scalability-video-lecture)
    *   [Step 2: Review the scalability article](#step-2-review-the-scalability-article)
    *   [Next steps](#next-steps)
10. [Performance vs scalability](#performance-vs-scalability)
11. [Latency vs throughput](#latency-vs-throughput)
12. [Availability vs consistency](#availability-vs-consistency)
    *   [CAP theorem](#cap-theorem)
    *   [CP - consistency and partition tolerance](#cp---consistency-and-partition-tolerance)
    *   [AP - availability and partition tolerance](#ap---availability-and-partition-tolerance)
13. [Consistency patterns](#consistency-patterns)
    *   [Weak consistency](#weak-consistency)
    *   [Eventual consistency](#eventual-consistency)
    *   [Strong consistency](#strong-consistency)
14. [Availability patterns](#availability-patterns)
    *   [Fail-over](#fail-over)
    *   [Replication](#replication)
    *   [Availability in numbers](#availability-in-numbers)
15. [Domain name system](#domain-name-system)
16. [Content delivery network](#content-delivery-network)
    *   [Push CDNs](#push-cdns)
    *   [Pull CDNs](#pull-cdns)
17. [Load balancer](#load-balancer)
    *   [Active-passive](#active-passive)
    *   [Active-active](#active-active)
    *   [Layer 4 load balancing](#layer-4-load-balancing)
    *   [Layer 7 load balancing](#layer-7-load-balancing)
    *   [Horizontal scaling](#horizontal-scaling)
18. [Reverse proxy (web server)](#reverse-proxy-web-server)
    *   [Load balancer vs reverse proxy](#load-balancer-vs-reverse-proxy)
19. [Application layer](#application-layer)
    *   [Microservices](#microservices)
    *   [Service discovery](#service-discovery)
20. [Database](#database)
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
21. [Cache](#cache)
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
22. [Asynchronism](#asynchronism)
    *   [Message queues](#message-queues)
    *   [Task queues](#task-queues)
    *   [Back pressure](#back-pressure)
23. [Communication](#communication)
    *   [Transmission control protocol (TCP)](#transmission-control-protocol-tcp)
    *   [User datagram protocol (UDP)](#user-datagram-protocol-udp)
    *   [Remote procedure call (RPC)](#remote-procedure-call-rpc)
    *   [Representational state transfer (REST)](#representational-state-transfer-rest)
24. [Security](#security)
25. [Appendix](#appendix)
    *   [Powers of two table](#powers-of-two-table)
    *   [Latency numbers every programmer should know](#latency-numbers-every-programmer-should-know)
    *   [Additional system design interview questions](#additional-system-design-interview-questions)
    *   [Real world architectures](#real-world-architectures)
    *   [Company architectures](#company-architectures)
    *   [Company engineering blogs](#company-engineering-blogs)
26. [Under development](#under-development)
27. [Credits](#credits)
28. [Contact info](#contact-info)
29. [License](#license)

---

## Motivation

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

## Study guide

> Suggested topics to review based on your interview timeline (short, medium, long).

![Imgur](images/OfVllex.png)

**Q: For interviews, do I need to know everything here?**

**A: No, you don't need to know everything here to prepare for the interview**.

What you are asked in an interview depends on variables such as:

*   How much experience you have
*   What your technical background is
*   What positions you are interviewing for
*   Which companies you are interviewing with
*   Luck

More experienced candidates are generally expected to know more about system design.  Architects or team leads might be expected to know more than individual contributors.  Top tech companies are likely to have one or more design interview rounds.

Start broad and go deeper in a few areas.  It helps to know a little about various key system design topics.  Adjust the following guide based on your timeline, experience, what positions you are interviewing for, and which companies you are interviewing with.

*   **Short timeline** - Aim for **breadth** with system design topics.  Practice by solving **some** interview questions.
*   **Medium timeline** - Aim for **breadth** and **some depth** with system design topics.  Practice by solving **many** interview questions.
*   **Long timeline** - Aim for **breadth** and **more depth** with system design topics.  Practice by solving **most** interview questions.

|                                                                                                         | Short | Medium | Long |
| :------------------------------------------------------------------------------------------------------ | :---- | :----- | :--- |
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

| Question                                              |                                                         |
| :---------------------------------------------------- | :------------------------------------------------------ |
| Design Pastebin.com (or Bit.ly)                      | [Solution](solutions/system_design/pastebin/README.md)  |
| Design the Twitter timeline and search (or Facebook feed and search) | [Solution](solutions/system_design/twitter/README.md) |
| Design a web crawler                                 | [Solution](solutions/system_design/web_crawler/README.md) |
| Design Mint.com                                      | [Solution](solutions/system_design/mint/README.md)       |
| Design the data structures for a social network       | [Solution](solutions/system_design/social_graph/README.md) |
| Design a key-value store for a search engine         | [Solution](solutions/system_design/query_cache/README.md) |
| Design Amazon's sales ranking by category feature    | [Solution](solutions/system_design/sales_rank/README.md) |
| Design a system that scales to millions of users on AWS | [Solution](solutions/system_design/scaling_aws/README.md) |
| Add a system design question                          | [Contribute](#contributing)                            |

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

## Object-oriented design interview questions with solutions

> Common object-oriented design interview questions with sample discussions, code, and diagrams.
>
> Solutions linked to content in the `solutions/` folder.

>**Note: This section is under development**

| Question                                        |                                                             |
| :---------------------------------------------- | :---------------------------------------------------------- |
| Design a hash map                             | [Solution](solutions/object_oriented_design/hash_table/hash_map.ipynb)  |
| Design a least recently used cache             | [Solution](solutions/object_oriented_design/lru_cache/lru_cache.ipynb)  |
| Design a call center                           | [Solution](solutions/object_oriented_design/call_center/call_center.ipynb)  |
| Design a deck of cards                         | [Solution](solutions/object_oriented_design/deck_of_cards/deck_of_cards.ipynb)  |
| Design a parking lot                           | [Solution](solutions/object_oriented_design/parking_lot/parking_lot.ipynb)  |
| Design a chat server                           | [Solution](solutions/object_oriented_design/online_chat/online_chat.ipynb)  |
| Design a circular array                       | [Contribute](#contributing)                                   |
| Add an object-oriented design question          | [Contribute](#contributing)                                   |

---

## Credits

Credits and sources are provided throughout this repo.

Special thanks to:

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