# System Design Primer: Ace Your Next Technical Interview

**Learn how to design large-scale systems and prepare for system design interviews with this comprehensive and open-source resource.  [Access the original repo here.](https://github.com/donnemartin/system-design-primer)**

## Key Features

*   **Comprehensive Coverage:** Explore core system design concepts, including performance vs. scalability, latency vs. throughput, and the CAP theorem.
*   **Interview Prep:** Study essential system design topics and practice with common interview questions, complete with sample solutions, diagrams, and code.
*   **Community-Driven:**  This continually updated, open-source project welcomes contributions, including fixes, improvements, and translations.
*   **Anki Flashcards:**  Reinforce learning with provided Anki flashcard decks for system design and object-oriented design exercises.

## Key Topics Covered

*   **System Design Fundamentals:**  Understand the building blocks of scalable systems.
*   **Performance, Scalability & Consistency Patterns:** Explore vital trade-offs, including **CAP Theorem**.
*   **Infrastructure Components:** Dive into DNS, CDNs, load balancers, reverse proxies, and application layers.
*   **Databases:** Master relational databases, NoSQL options, and caching strategies.
*   **Asynchronism:**  Learn about message queues, task queues, and back pressure.
*   **Communication Protocols:**  Explore TCP, UDP, RPC, and REST.
*   **Security Basics:**  Gain a foundation in security best practices.
*   **Study Guide & Interview Preparation:**  A suggested study guide helps tailor your learning to your interview timeline.

## Index of System Design Topics

*   **System Design Topics: Start Here**
    *   Scalability Video Lecture & Article
    *   Next Steps
*   **Core Principles:**
    *   Performance vs. Scalability
    *   Latency vs. Throughput
    *   Availability vs. Consistency
        *   CAP Theorem
            *   CP - Consistency and Partition Tolerance
            *   AP - Availability and Partition Tolerance
*   **Consistency & Availability Patterns**
    *   Consistency Patterns:
        *   Weak Consistency
        *   Eventual Consistency
        *   Strong Consistency
    *   Availability Patterns:
        *   Fail-over (Active-Passive, Active-Active)
        *   Replication
        *   Availability in Numbers
*   **Infrastructure:**
    *   Domain Name System
    *   Content Delivery Network (Push & Pull CDNs)
    *   Load Balancer (Layer 4, Layer 7, Horizontal Scaling)
    *   Reverse Proxy (Web Server)
*   **Application & Database:**
    *   Application Layer
        *   Microservices
        *   Service Discovery
    *   Database:
        *   Relational Database Management System (RDBMS)
            *   Master-Slave/Master-Master Replication
            *   Federation/Sharding
            *   Denormalization/SQL Tuning
        *   NoSQL
            *   Key-Value Store
            *   Document Store
            *   Wide Column Store
            *   Graph Database
        *   SQL or NoSQL?
*   **Caching**
    *   Client/CDN/Web Server/Database/Application Caching
    *   Cache Update Strategies (Cache-Aside, Write-Through, Write-Behind, Refresh-Ahead)
*   **Asynchronism**
    *   Message Queues
    *   Task Queues
    *   Back Pressure
*   **Communication**
    *   Transmission Control Protocol (TCP)
    *   User Datagram Protocol (UDP)
    *   Remote Procedure Call (RPC)
    *   Representational State Transfer (REST)
*   **Security**
*   **Appendix**
    *   Powers of Two Table
    *   Latency Numbers Every Programmer Should Know
    *   Additional System Design Interview Questions
    *   Real World Architectures
    *   Company Architectures
    *   Company Engineering Blogs

## Study Guide

Prepare based on your interview timeline (short, medium, long).

*   **Short Timeline:** Focus on **breadth** across system design topics. Practice solving **some** interview questions.
*   **Medium Timeline:** Aim for **breadth and some depth** with system design topics. Practice solving **many** interview questions.
*   **Long Timeline:** Aim for **breadth and more depth** with system design topics. Practice solving **most** interview questions.

## How to Approach a System Design Interview Question

1.  **Outline use cases, constraints, and assumptions:** Clarify requirements.
2.  **Create a high-level design:**  Sketch the main components and connections.
3.  **Design core components:** Dive into the details of each component.
4.  **Scale the design:** Address bottlenecks, considering load balancing, caching, and sharding.
5.  **Back-of-the-envelope calculations:**  Estimate performance metrics.

## System Design Interview Questions with Solutions

Practice common system design interview questions with discussions, diagrams, and code.

*   Design Pastebin.com (or Bit.ly)
*   Design the Twitter timeline and search (or Facebook feed and search)
*   Design a web crawler
*   Design Mint.com
*   Design the data structures for a social network
*   Design a key-value store for a search engine
*   Design Amazon's sales ranking by category feature
*   Design a system that scales to millions of users on AWS

## Object-Oriented Design Interview Questions with Solutions (In Development)

*   Design a hash map
*   Design a least recently used cache
*   Design a call center
*   Design a deck of cards
*   Design a parking lot
*   Design a chat server
*   Design a circular array

**And more!**

[Contribute](#contributing) to add more solutions.

## Contributing

*   Submit pull requests to help:
    *   Fix errors
    *   Improve sections
    *   Add new sections
    *   Translate

## Credits, Contact & License

*   See original README for details.