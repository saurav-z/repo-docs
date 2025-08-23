# System Design Primer: Ace Your System Design Interviews!

**Prepare for system design interviews and learn to build scalable systems with this comprehensive, open-source guide.** [Visit the Original Repo](https://github.com/donnemartin/system-design-primer).

This guide provides a structured and organized collection of resources to help you master system design principles and prepare for technical interviews. Whether you're a software engineer, architect, or preparing for your next interview, this primer offers a wealth of information, including:

*   **In-depth explanations of core system design concepts** to help you build scalable systems.
*   **Practical interview preparation**, including sample questions, solutions, and study guides.
*   **A continually updated open-source project** that welcomes contributions from the community.
*   **Anki flashcards** to help you retain key system design concepts, making it perfect for use on-the-go.

## Key Features

*   **Comprehensive Coverage:** Covers a wide range of system design topics, from fundamental principles to advanced architectures.
*   **Interview Preparation:** Includes common system design interview questions, sample solutions, and guidance on approaching interview questions.
*   **Community-Driven:** An open-source project that encourages contributions from the community, ensuring the content remains up-to-date and relevant.
*   **Practical Resources:** Includes Anki flashcards, real-world architecture examples, and links to essential resources.
*   **Multilingual:** Available in multiple languages to make the content accessible to a global audience.

## Core System Design Concepts

This section introduces core system design concepts and trade-offs.  **Everything is a trade-off.**

### Performance vs. Scalability

A service is **scalable** if it results in increased **performance** in a manner proportional to resources added. Generally, increasing performance means serving more units of work, but it can also be to handle larger units of work, such as when datasets grow.

### Latency vs. Throughput

*   **Latency:** The time to perform some action or to produce some result.
*   **Throughput:** The number of such actions or results per unit of time.

Generally, you should aim for **maximal throughput** with **acceptable latency**.

### Availability vs. Consistency

The **CAP theorem** states that in a distributed computer system, you can only support two of the following guarantees:

*   **Consistency:** Every read receives the most recent write or an error.
*   **Availability:** Every request receives a response, without guarantee that it contains the most recent version of the information.
*   **Partition Tolerance:** The system continues to operate despite arbitrary partitioning due to network failures.

#### Consistency Patterns

*   **Weak Consistency:** After a write, reads may or may not see it.  A best effort approach is taken.
*   **Eventual Consistency:** After a write, reads will eventually see it (typically within milliseconds). Data is replicated asynchronously.
*   **Strong Consistency:** After a write, reads will see it. Data is replicated synchronously.

#### Availability Patterns

*   **Fail-over:** Techniques such as Active-Passive and Active-Active to ensure service availability.
*   **Replication:** Master-slave and master-master database replication strategies.

## In-depth System Design Topics

*   [Domain Name System (DNS)](#domain-name-system)
*   [Content Delivery Network (CDN)](#content-delivery-network)
*   [Load Balancer](#load-balancer)
*   [Reverse Proxy (Web Server)](#reverse-proxy-web-server)
*   [Application Layer](#application-layer)
*   [Database](#database)
*   [Cache](#cache)
*   [Asynchronism](#asynchronism)
*   [Communication](#communication)
*   [Security](#security)

## Study Guide

Choose a study approach that will best fit your needs:

*   **Short Timeline:** Aim for breadth with system design topics, and solve some interview questions.
*   **Medium Timeline:** Aim for breadth and some depth with system design topics, and solve many interview questions.
*   **Long Timeline:** Aim for breadth and more depth with system design topics, and solve most interview questions.

## How to Approach a System Design Interview Question

1.  **Outline use cases, constraints, and assumptions:** Gather requirements and scope the problem. Ask questions to clarify use cases and constraints. Discuss assumptions.
2.  **Create a high-level design:** Outline a high-level design with all important components.
3.  **Design core components:** Dive into details for each core component.
4.  **Scale the design:** Identify and address bottlenecks, given the constraints.

## System Design Interview Questions with Solutions

> Common system design interview questions with sample discussions, code, and diagrams.
>
> Solutions linked to content in the `solutions/` folder.

| Question | |
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

## Object-Oriented Design Interview Questions (Under Development)

> Common object-oriented design interview questions with sample discussions, code, and diagrams.

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

## Appendix

*   [Powers of two table](#powers-of-two-table)
*   [Latency numbers every programmer should know](#latency-numbers-every-programmer-should-know)
*   [Additional system design interview questions](#additional-system-design-interview-questions)
*   [Real world architectures](#real-world-architectures)
*   [Company architectures](#company-architectures)
*   [Company engineering blogs](#company-engineering-blogs)

---

**Ready to level up your system design skills? Dive into the [System Design Primer](https://github.com/donnemartin/system-design-primer) to gain the knowledge and practice needed for success!**
```

Key improvements and summaries:

*   **SEO Optimization:**  Includes relevant keywords (e.g., "system design," "interview," "scalable systems"). Uses headings and subheadings.
*   **Concise Hook:** Starts with a strong, single-sentence hook.
*   **Bulleted Key Features:** Highlights the main benefits of the guide.
*   **Structured Content:**  Uses clear headings, subheadings, and lists to improve readability and organization.
*   **Summarized Content:** Provides brief summaries of each section, making it easier to understand the scope.
*   **Actionable Language:** Uses calls to action ("Dive into," "Ready to") to encourage engagement.
*   **Links Back:**  The important link to the repo is at the top and bottom.
*   **Simplified:** Removed redundant text.
*   **Focus:** Focused on the most valuable aspects.
*   **Included the most important information from the original README.**
*   **Removed the translation links (moved to the bottom of the original README).**
*   **Removed some information that the average user won't need.**
*   **Simplified the study guide**