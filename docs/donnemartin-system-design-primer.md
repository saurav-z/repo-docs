# System Design Primer: Your Guide to Building Scalable Systems

**Master system design concepts and ace your technical interviews with this comprehensive, open-source resource.**  [Explore the original repository on GitHub](https://github.com/donnemartin/system-design-primer).

## Key Features

*   **In-depth coverage:** Explore core concepts like performance vs. scalability, latency vs. throughput, and the CAP theorem.
*   **Practical examples:** Dive into system design interview questions with detailed solutions, diagrams, and code examples for real-world scenarios.
*   **Community-driven:**  Benefit from a constantly updated and improved open-source resource, with contributions welcome.
*   **Interview prep:**  Find study guides, object-oriented design questions, and valuable tips for acing system design interviews.
*   **Spaced Repetition:** Utilize Anki flashcards for effective knowledge retention of critical system design concepts.

## Core Concepts & Topics

This primer breaks down complex topics into manageable sections, providing a strong foundation for understanding system design principles.

### 1.  Foundational Principles

*   **Performance vs. Scalability:** Understanding the distinction between optimizing for a single user and handling increased load.
*   **Latency vs. Throughput:** Balancing the speed of responses with the volume of requests processed.
*   **Availability vs. Consistency (CAP Theorem):**  Navigating the trade-offs in distributed systems using the CAP theorem
    *   **CP (Consistency and Partition Tolerance)**
    *   **AP (Availability and Partition Tolerance)**

### 2.  Scalability Patterns

*   **Consistency Patterns:** Explore options for data synchronization.
    *   Weak Consistency
    *   Eventual Consistency
    *   Strong Consistency
*   **Availability Patterns:** Implement strategies for system resilience.
    *   Fail-over (Active-Passive, Active-Active)
    *   Replication
    *   Availability in numbers

### 3.  Essential Technologies & Architectures

*   **DNS (Domain Name System):** Understanding how domain names translate to IP addresses.
*   **CDN (Content Delivery Network):** Optimizing content delivery through distributed caching.
    *   Push CDNs
    *   Pull CDNs
*   **Load Balancer:** Distributing traffic across multiple servers.
    *   Layer 4 and Layer 7 Load Balancing
    *   Horizontal Scaling
*   **Reverse Proxy (Web Server):** Centralizing services and providing a unified interface.
*   **Application Layer:**  Decoupling the web and application layers.
    *   Microservices
    *   Service Discovery
*   **Database:**  Scaling data storage and retrieval.
    *   Relational Database Management System (RDBMS)
        *   Master-slave replication
        *   Master-master replication
        *   Federation
        *   Sharding
        *   Denormalization
        *   SQL tuning
    *   NoSQL: Understanding different NoSQL models.
        *   Key-value store
        *   Document store
        *   Wide column store
        *   Graph Database
    *   SQL or NoSQL:  Choosing the right database technology for the job.
*   **Cache:** Enhancing performance by storing frequently accessed data.
    *   Client caching
    *   CDN caching
    *   Web server caching
    *   Database caching
    *   Application caching
    *   Caching strategies (Cache-aside, Write-through, Write-behind, Refresh-ahead)
*   **Asynchronism:** Decoupling tasks and improving responsiveness.
    *   Message queues
    *   Task queues
    *   Back pressure
*   **Communication:** Protocols and methods for inter-system communication.
    *   TCP
    *   UDP
    *   RPC
    *   REST
*   **Security:** Principles for building secure systems.

### 4.  Interview Preparation

*   **Study Guide:** Suggested topics for your interview preparation based on your timeline.
*   **How to Approach a System Design Interview Question:** A step-by-step guide to tackling design problems.
*   **System Design Interview Questions with Solutions:**  Practical examples, diagrams, and code.
*   **Object-oriented design interview questions with solutions**
*   **Additional System Design Interview Questions:**  A comprehensive list of design problems.

## Contributing

This is a community-driven project, and your contributions are welcome!

*   Fix errors
*   Improve sections
*   Add new sections
*   [Translate](https://github.com/donnemartin/system-design-primer/issues/28)

Review the [Contributing Guidelines](CONTRIBUTING.md).

## Resources

*   **Anki Flashcards:**  Improve retention with spaced repetition.
    *   System Design Deck
    *   System Design Exercises Deck
    *   Object Oriented Design Exercises Deck
*   **Coding challenges:** Enhance your preparation with interactive coding challenges.
    *   Coding Deck

## Appendix

*   **Powers of Two Table**
*   **Latency Numbers Every Programmer Should Know**
*   **Additional System Design Interview Questions**
*   **Real World Architectures**
*   **Company Architectures**
*   **Company Engineering Blogs**

## Under Development

See what's coming next, and consider contributing.

## Credits & Contact

Find all the resources and individuals that helped create this primer in the Credits section of the [original README](https://github.com/donnemartin/system-design-primer).  Contact information can be found on the [author's GitHub page](https://github.com/donnemartin).

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).