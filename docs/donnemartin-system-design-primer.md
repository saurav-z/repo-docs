# System Design Primer: Your Guide to Designing Large-Scale Systems

**Master system design for interviews and real-world applications with this open-source collection of resources.** Find the original repo [here](https://github.com/donnemartin/system-design-primer).

## Key Features

*   **Comprehensive Resource Collection:** Organized collection of resources on system design principles.
*   **Interview Prep:**  Learn how to design systems at scale and prepare for system design interviews.
*   **Community Driven:** A continually updated, open-source project, allowing you to learn from and contribute to the community.
*   **Practical Solutions:** Common system design interview questions with sample discussions, code, and diagrams.
*   **Anki Flashcards:** Spaced repetition flashcards for efficient retention of key system design concepts.
*   **Study Guides:** Suggested topics and approaches to interview prep based on your timeline.

## Index of System Design Topics

*   **System Design Topics: Start Here**
    *   Step 1: Review the Scalability Video Lecture
    *   Step 2: Review the Scalability Article
    *   Next Steps
*   Performance vs Scalability
*   Latency vs Throughput
*   Availability vs Consistency
    *   CAP Theorem
        *   CP - Consistency and Partition Tolerance
        *   AP - Availability and Partition Tolerance
*   Consistency Patterns
    *   Weak Consistency
    *   Eventual Consistency
    *   Strong Consistency
*   Availability Patterns
    *   Fail-Over
        *   Active-Passive
        *   Active-Active
    *   Replication
        *   Master-Slave and Master-Master
    *   Availability in Numbers
*   Domain Name System
*   Content Delivery Network
    *   Push CDNs
    *   Pull CDNs
*   Load Balancer
    *   Active-Passive
    *   Active-Active
    *   Layer 4 Load Balancing
    *   Layer 7 Load Balancing
    *   Horizontal Scaling
*   Reverse Proxy (Web Server)
    *   Load Balancer vs Reverse Proxy
*   Application Layer
    *   Microservices
    *   Service Discovery
*   Database
    *   Relational Database Management System (RDBMS)
        *   Master-Slave Replication
        *   Master-Master Replication
        *   Federation
        *   Sharding
        *   Denormalization
        *   SQL Tuning
    *   NoSQL
        *   Key-Value Store
        *   Document Store
        *   Wide Column Store
        *   Graph Database
    *   SQL or NoSQL
*   Cache
    *   Client Caching
    *   CDN Caching
    *   Web Server Caching
    *   Database Caching
    *   Application Caching
    *   Caching at the Database Query Level
    *   Caching at the Object Level
    *   When to Update the Cache
        *   Cache-Aside
        *   Write-Through
        *   Write-Behind (Write-Back)
        *   Refresh-Ahead
*   Asynchronism
    *   Message Queues
    *   Task Queues
    *   Back Pressure
*   Communication
    *   Transmission Control Protocol (TCP)
    *   User Datagram Protocol (UDP)
    *   Remote Procedure Call (RPC)
    *   Representational State Transfer (REST)
*   Security
*   Appendix
    *   Powers of Two Table
    *   Latency Numbers Every Programmer Should Know
    *   Additional System Design Interview Questions
    *   Real World Architectures
    *   Company Architectures
    *   Company Engineering Blogs
*   Under Development
*   Credits
*   Contact Info
*   License

## Study Guide

Follow this suggested study guide.

| | Short | Medium | Long |
|---|---|---|---|
| Read through the [System design topics](#index-of-system-design-topics) to get a broad understanding of how systems work | :+1: | :+1: | :+1: |
| Read through a few articles in the [Company engineering blogs](#company-engineering-blogs) for the companies you are interviewing with | :+1: | :+1: | :+1: |
| Read through a few [Real world architectures](#real-world-architectures) | :+1: | :+1: | :+1: |
| Review [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question) | :+1: | :+1: | :+1: |
| Work through [System design interview questions with solutions](#system-design-interview-questions-with-solutions) | Some | Many | Most |
| Work through [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions) | Some | Many | Most |
| Review [Additional system design interview questions](#additional-system-design-interview-questions) | Some | Many | Most |

## How to Approach a System Design Interview Question

Follow these steps:

1.  **Outline use cases, constraints, and assumptions:** Define the problem.
2.  **Create a high-level design:** Sketch out the components.
3.  **Design core components:** Dive into details of each core component.
4.  **Scale the design:** Identify and address potential bottlenecks.

## System Design Interview Questions with Solutions

Here are some sample questions:

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

## System Design Topics: Start Here

*   Review the scalability video lecture.
*   Review the scalability article.
*   Then, review performance vs scalability, latency vs throughput, and availability vs consistency.

## Real-World Architectures

Study the common principles, common technologies, and patterns. Review the lessons learned.

| Type          | System                     | Reference(s)                                                                                                                                                                     |
|---------------|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data Processing | MapReduce                  | [research.google.com](http://static.googleusercontent.com/media/research.google.com/zh-CN/us/archive/mapreduce-osdi04.pdf)                                                 |
| Data Processing | Spark                      | [slideshare.net](http://www.slideshare.net/AGrishchenko/apache-spark-architecture)                                                                                          |
| Data Processing | Storm                      | [slideshare.net](http://www.slideshare.net/previa/storm-16094009)                                                                                                            |
| Data Store      | Bigtable                   | [harvard.edu](http://www.read.seas.harvard.edu/~kohler/class/cs239-w08/chang06bigtable.pdf)                                                                                |
| Data Store      | HBase                      | [slideshare.net](http://www.slideshare.net/alexbaranau/intro-to-hbase)                                                                                                       |
| Data Store      | Cassandra                  | [slideshare.net](http://www.slideshare.net/planetcassandra/cassandra-introduction-features-30103666)                                                                         |
| Data Store      | DynamoDB                   | [harvard.edu](http://www.read.seas.harvard.edu/~kohler/class/cs239-w08/decandia07dynamo.pdf)                                                                                |
| Data Store      | MongoDB                    | [slideshare.net](http://www.slideshare.net/mdirolf/introduction-to-mongodb)                                                                                                    |
| Data Store      | Spanner                    | [research.google.com](http://research.google.com/archive/spanner-osdi2012.pdf)                                                                                              |
| Data Store      | Memcached                  | [slideshare.net](http://www.slideshare.net/oemebamo/introduction-to-memcached)                                                                                                |
| Data Store      | Redis                      | [slideshare.net](http://www.slideshare.net/dvirsky/introduction-to-redis)                                                                                                     |
| Misc          | Chubby                     | [research.google.com](http://static.googleusercontent.com/external_content/untrusted_dlcp/research.google.com/en/us/archive/chubby-osdi06.pdf)                                 |
| Misc          | Dapper                     | [research.google.com](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36356.pdf)                                                               |
| Misc          | Kafka                      | [slideshare.net](http://www.slideshare.net/mumrah/kafka-talk-tri-hug)                                                                                                           |
| Misc          | Zookeeper                  | [slideshare.net](http://www.slideshare.net/sauravhaloi/introduction-to-apache-zookeeper)                                                                                        |

## Company Architectures

Here are some examples of company architectures.

*   Amazon
*   Cinchcast
*   DataSift
*   Dropbox
*   ESPN
*   Google
*   Instagram
*   Justin.tv
*   Facebook
*   Flickr
*   Mailbox
*   Netflix
*   Pinterest
*   Playfish
*   PlentyOfFish
*   Salesforce
*   Stack Overflow
*   TripAdvisor
*   Tumblr
*   Twitter
*   Uber
*   WhatsApp
*   YouTube

## Contributing

Contribute to this resource by submitting pull requests to fix errors, improve sections, or add new sections.

## Contact Info

Contact the author on his [GitHub page](https://github.com/donnemartin) with any questions.