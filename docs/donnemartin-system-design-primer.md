# System Design Primer: Your Guide to Architecting Scalable Systems

**Tired of feeling lost in system design interviews?** This open-source resource provides an organized collection of resources to help you master the art of building large-scale systems.  

[View the original repo](https://github.com/donnemartin/system-design-primer)

## Key Features

*   **Comprehensive Coverage:** Dive into essential system design topics.
    *   Explore fundamental concepts such as performance vs. scalability, latency vs. throughput, and the CAP theorem.
    *   Understand various consistency and availability patterns.
    *   Learn about core components like CDNs, load balancers, databases, and caching strategies.
*   **Interview Prep Focused:** Ace your system design interviews with confidence.
    *   Learn how to approach system design interview questions with a step-by-step methodology.
    *   Practice common system design questions, including solutions, diagrams, and code examples.
    *   Prepare for object-oriented design questions with sample solutions.
*   **Community Driven:** Benefit from a continually updated, open-source project.
    *   Contribute by fixing errors, improving sections, adding new sections, and translating the guide.
    *   Leverage insights from the community to enhance your understanding.
*   **Practical Resources:** Access valuable tools and references to solidify your knowledge.
    *   Utilize Anki flashcard decks for spaced repetition and better retention.
    *   Explore a comprehensive index of system design topics with summaries, pros and cons, and in-depth resources.
    *   Review a suggested study guide tailored to different interview timelines (short, medium, long).
    *   Access back-of-the-envelope calculation guides for quick estimations.
    *   Reference real-world and company architectures, along with curated engineering blogs, for deeper insights.

## Table of Contents

*   [Motivation](#motivation)
    *   [Learn how to design large-scale systems](#learn-how-to-design-large-scale-systems)
    *   [Learn from the open source community](#learn-from-the-open-source-community)
    *   [Prep for the system design interview](#prep-for-the-system-design-interview)
*   [Anki flashcards](#anki-flashcards)
    *   [Coding Resource: Interactive Coding Challenges](#coding-resource-interactive-coding-challenges)
*   [Contributing](#contributing)
*   [Index of system design topics](#index-of-system-design-topics)
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
*   [Study guide](#study-guide)
*   [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question)
*   [System design interview questions with solutions](#system-design-interview-questions-with-solutions)
    *   [Design Pastebin.com (or Bit.ly)](#design-pastebincom-or-bitly)
    *   [Design the Twitter timeline and search (or Facebook feed and search)](#design-the-twitter-timeline-and-search-or-facebook-feed-and-search)
    *   [Design a web crawler](#design-a-web-crawler)
    *   [Design Mint.com](#design-mintcom)
    *   [Design the data structures for a social network](#design-the-data-structures-for-a-social-network)
    *   [Design a key-value store for a search engine](#design-a-key-value-store-for-a-search-engine)
    *   [Design Amazon's sales ranking by category feature](#design-amazons-sales-ranking-by-category-feature)
    *   [Design a system that scales to millions of users on AWS](#design-a-system-that-scales-to-millions-of-users-on-aws)
*   [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions)
*   [System design topics: start here](#system-design-topics-start-here)
    *   [Step 1: Review the scalability video lecture](#step-1-review-the-scalability-video-lecture)
    *   [Step 2: Review the scalability article](#step-2-review-the-scalability-article)
    *   [Next steps](#next-steps)
*   [Under development](#under-development)
*   [Credits](#credits)
*   [Contact info](#contact-info)
*   [License](#license)
```

**SEO Optimization Notes:**

*   **Keywords:** "System Design", "Scalable Systems", "System Design Interview", "Architecture", "Database", "Caching", "Load Balancing", "Microservices", "REST", "NoSQL".
*   **Headings:** Clear, concise headings and subheadings help with readability and keyword targeting.
*   **Bullet Points:**  Use bullet points to highlight key features and benefits.
*   **Concise Language:** Short, impactful sentences for clarity.
*   **Internal Linking:**  Linking within the document to various sections will help the search engines understand the content.
*   **External Linking:**  Linking to the original repo is crucial, as well as to useful external resources (where appropriate).