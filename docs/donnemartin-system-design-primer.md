# System Design Primer: Your Guide to Building Scalable Systems

**Conquer system design interviews and master the art of designing large-scale systems with this comprehensive resource. ([Original Repo](https://github.com/donnemartin/system-design-primer))**

## Key Features

*   **Organized Resources:** A curated collection of articles, diagrams, and code examples covering essential system design topics.
*   **Interview Preparation:**  Master common system design interview questions with sample solutions, code, and diagrams.
*   **Community-Driven:** An open-source project with room for contributions from the community.
*   **Anki Flashcards:**  Spaced repetition flashcards to solidify your understanding of key concepts.
*   **Comprehensive Coverage:** Explores everything from fundamental principles to practical architectures.

## Core Topics

### 1. Introduction

*   **Motivation:**
    *   Learn how to design large-scale systems and prep for system design interviews.
    *   A broad topic that has a vast amount of resources scattered throughout the web on system design principles.
    *   An organized collection of resources to help you learn how to build systems at scale.
    *   This is a continually updated, open source project.
*   **Learn from the open source community**
    *   This is a continually updated, open source project.
    *   [Contributions](#contributing) are welcome!
*   **Prep for the system design interview**
    *   In addition to coding interviews, system design is a **required component** of the **technical interview process** at many tech companies.
    *   **Practice common system design interview questions** and **compare** your results with **sample solutions**: discussions, code, and diagrams.
    *   Additional topics for interview prep:
        *   [Study guide](#study-guide)
        *   [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question)
        *   [System design interview questions, **with solutions**](#system-design-interview-questions-with-solutions)
        *   [Object-oriented design interview questions, **with solutions**](#object-oriented-design-interview-questions-with-solutions)
        *   [Additional system design interview questions](#additional-system-design-interview-questions)

### 2. Core Concepts

*   **Performance vs. Scalability:** The difference between improving speed for a single user vs. handling increased load.
*   **Latency vs. Throughput:** Balancing the time for a task to complete (latency) with the rate of tasks completed (throughput).
*   **CAP Theorem:** Understanding the trade-offs between Consistency, Availability, and Partition Tolerance in distributed systems.
    *   **CP (Consistency, Partition Tolerance):** Prioritizes data accuracy in the presence of network failures.
    *   **AP (Availability, Partition Tolerance):** Prioritizes system uptime, even if data might be slightly stale.
*   **Consistency Patterns:**  Approaches to maintain data consistency across multiple copies.  (Weak, Eventual, Strong)
*   **Availability Patterns:** Techniques for ensuring system uptime and reliability. (Fail-over, Replication)
*   **DNS:** Domain Name System - Translates domain names to IP addresses.
*   **CDN:** Content Delivery Network - Distributes content globally for faster delivery. (Push/Pull CDNs)
*   **Load Balancer:** Distributes traffic across multiple servers. (Layer 4/7)
*   **Reverse Proxy (Web Server):** Centralizes and manages traffic to internal services.

### 3. System Components

*   **Application Layer:** Separating web logic for scalability. (Microservices, Service Discovery)
*   **Database:** Choosing the right database for the job. (RDBMS, NoSQL)
    *   **RDBMS:** Relational database systems. (Master-Slave, Federation, Sharding, Denormalization, SQL Tuning)
    *   **NoSQL:** Non-relational database systems. (Key-Value, Document, Wide Column, Graph)
    *   **SQL vs. NoSQL:**  Deciding when to use each.
*   **Caching:** Improving performance by storing frequently accessed data. (Client, CDN, Web Server, Database, Application)
    *   Caching Strategies: (Cache-Aside, Write-Through, Write-Behind, Refresh-Ahead)
*   **Asynchronism:**  Offloading tasks for improved responsiveness. (Message Queues, Task Queues, Back Pressure)
*   **Communication:** Protocols used for data transfer. (HTTP, TCP, UDP, RPC, REST)
*   **Security:**  Basic security principles for system design.

### 4. Studying and Resources

*   **Study Guide:** Suggested topics based on your interview preparation timeline (short, medium, long).
*   **How to approach a system design interview question:** Step-by-step process.
*   **System Design Interview Questions with Solutions:**  Detailed solutions to common interview questions.
    *   Design Pastebin.com (or Bit.ly)
    *   Design the Twitter timeline and search (or Facebook feed and search)
    *   Design a web crawler
    *   Design Mint.com
    *   Design the data structures for a social network
    *   Design a key-value store for a search engine
    *   Design Amazon's sales ranking by category feature
    *   Design a system that scales to millions of users on AWS
*   **Object-oriented design interview questions with solutions:** Common object-oriented design interview questions with sample discussions, code, and diagrams.
*   **Additional System Design Interview Questions**
*   **Appendix:** Useful references like powers of two and latency numbers.
*   **Real World Architectures:** Case studies of architectures used by real companies.
*   **Company Engineering Blogs:** Links to engineering blogs.
*   **Credits:**  Acknowledgments to contributors and sources.

## Contribute

*   Fix errors
*   Improve sections
*   Add new sections
*   [Translate](https://github.com/donnemartin/system-design-primer/issues/28)

## Contact

*   My contact info can be found on my [GitHub page](https://github.com/donnemartin).

## License

Copyright 2017 Donne Martin

Creative Commons Attribution 4.0 International License (CC BY 4.0)
http://creativecommons.org/licenses/by/4.0/