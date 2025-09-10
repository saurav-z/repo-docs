# System Design Primer: Your Guide to Building Scalable Systems

**Master the art of designing large-scale systems and ace your system design interview with this comprehensive, community-driven resource!**

[View the original repo on GitHub](https://github.com/donnemartin/system-design-primer)

This primer is an organized collection of resources and knowledge to help you understand and build scalable systems, prepare for system design interviews, and become a better engineer. Whether you're new to system design or a seasoned pro, this guide provides valuable insights and practical advice.

## Key Features

*   **Comprehensive Coverage:** Explore fundamental concepts and advanced topics in system design.
*   **Interview Prep:** Learn how to approach system design interview questions, with example solutions, diagrams, and code.
*   **Community-Driven:** Benefit from a continually updated, open-source project with contributions from a wide audience.
*   **Anki Flashcards:** Enhance your retention with Anki flashcards designed to reinforce key concepts.
*   **Practical Examples:** Analyze real-world architectures, company engineering blogs, and example problems.
*   **Multilingual Support:** Available in multiple languages, fostering a wider reach and deeper engagement.

## Getting Started

### System Design Fundamentals

*   **Learn how to design large-scale systems** - This primer guides you through the process.
*   **Practice for the system design interview** - This is an essential component of the technical interview process at many tech companies.
*   **Learn from the community** - This is a continually updated, open source project.

### Prepare for the System Design Interview

In addition to coding interviews, system design is a required component of the technical interview process at many tech companies.

**Practice common system design interview questions** and **compare** your results with **sample solutions**: discussions, code, and diagrams.

Additional topics for interview prep:

*   [Study guide](#study-guide)
*   [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question)
*   [System design interview questions, **with solutions**](#system-design-interview-questions-with-solutions)
*   [Object-oriented design interview questions, **with solutions**](#object-oriented-design-interview-questions-with-solutions)
*   [Additional system design interview questions](#additional-system-design-interview-questions)

### Study Guide

Suggested topics to review based on your interview timeline (short, medium, long).

![Imgur](images/OfVllex.png)

|                  | Short                                                                                                        | Medium                                                                                                            | Long                                                                                                             |
| :--------------- | :----------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| Study            | Read through the [System design topics](#index-of-system-design-topics) to get a broad understanding of how systems work. | Read through the [System design topics](#index-of-system-design-topics) to get a broad understanding of how systems work. | Read through the [System design topics](#index-of-system-design-topics) to get a broad understanding of how systems work. |
| Additional study            | Read through a few articles in the [Company engineering blogs](#company-engineering-blogs) for the companies you are interviewing with | Read through a few articles in the [Company engineering blogs](#company-engineering-blogs) for the companies you are interviewing with | Read through a few articles in the [Company engineering blogs](#company-engineering-blogs) for the companies you are interviewing with.                                                                                                            |
| Further study     | Read through a few [Real world architectures](#real-world-architectures)                                    | Read through a few [Real world architectures](#real-world-architectures)                                       | Read through a few [Real world architectures](#real-world-architectures)                                         |
| Review           | Review [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question) | Review [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question) | Review [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question) |
| Practice         | Work through [System design interview questions with solutions](#system-design-interview-questions-with-solutions)  | Work through [System design interview questions with solutions](#system-design-interview-questions-with-solutions) | Work through [System design interview questions with solutions](#system-design-interview-questions-with-solutions) |
| Additional Practice | Work through [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions) | Work through [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions) | Work through [Object-oriented design interview questions with solutions](#object-oriented-design-interview-questions-with-solutions)  |
| Further practice | Review [Additional system design interview questions](#additional-system-design-interview-questions)          | Review [Additional system design interview questions](#additional-system-design-interview-questions)         | Review [Additional system design interview questions](#additional-system-design-interview-questions)          |

### How to Approach a System Design Interview Question

The system design interview is an open-ended conversation.  You are expected to lead it.

You can use the following steps to guide the discussion.  To help solidify this process, work through the [System design interview questions with solutions](#system-design-interview-questions-with-solutions) section using the following steps.

#### Step 1: Outline use cases, constraints, and assumptions

Gather requirements and scope the problem.  Ask questions to clarify use cases and constraints.  Discuss assumptions.

*   Who is going to use it?
*   How are they going to use it?
*   How many users are there?
*   What does the system do?
*   What are the inputs and outputs of the system?
*   How much data do we expect to handle?
*   How many requests per second do we expect?
*   What is the expected read to write ratio?

#### Step 2: Create a high level design

Outline a high level design with all important components.

*   Sketch the main components and connections
*   Justify your ideas

#### Step 3: Design core components

Dive into details for each core component.  For example, if you were asked to [design a url shortening service](solutions/system_design/pastebin/README.md), discuss:

*   Generating and storing a hash of the full url
    *   [MD5](solutions/system_design/pastebin/README.md) and [Base62](solutions/system_design/pastebin/README.md)
    *   Hash collisions
    *   SQL or NoSQL
    *   Database schema
*   Translating a hashed url to the full url
    *   Database lookup
*   API and object-oriented design

#### Step 4: Scale the design

Identify and address bottlenecks, given the constraints.  For example, do you need the following to address scalability issues?

*   Load balancer
*   Horizontal scaling
*   Caching
*   Database sharding

Discuss potential solutions and trade-offs.  Everything is a trade-off.  Address bottlenecks using [principles of scalable system design](#index-of-system-design-topics).

### System design interview questions with solutions

> Common system design interview questions with sample discussions, code, and diagrams.
>
> Solutions linked to content in the `solutions/` folder.

| Question                                                              |                                                                                  |
| :-------------------------------------------------------------------- | :------------------------------------------------------------------------------- |
| Design Pastebin.com (or Bit.ly)                                        | [Solution](solutions/system_design/pastebin/README.md)                          |
| Design the Twitter timeline and search (or Facebook feed and search)   | [Solution](solutions/system_design/twitter/README.md)                           |
| Design a web crawler                                                  | [Solution](solutions/system_design/web_crawler/README.md)                         |
| Design Mint.com                                                        | [Solution](solutions/system_design/mint/README.md)                              |
| Design the data structures for a social network                         | [Solution](solutions/system_design/social_graph/README.md)                        |
| Design a key-value store for a search engine                           | [Solution](solutions/system_design/query_cache/README.md)                         |
| Design Amazon's sales ranking by category feature                     | [Solution](solutions/system_design/sales_rank/README.md)                          |
| Design a system that scales to millions of users on AWS              | [Solution](solutions/system_design/scaling_aws/README.md)                         |
| Add a system design question                                           | [Contribute](#contributing)                                                    |

### Object-oriented design interview questions with solutions

> Common object-oriented design interview questions with sample discussions, code, and diagrams.
>
> Solutions linked to content in the `solutions/` folder.

>**Note: This section is under development**

| Question                                           |                                                                        |
| :------------------------------------------------- | :--------------------------------------------------------------------- |
| Design a hash map                                   | [Solution](solutions/object_oriented_design/hash_table/hash_map.ipynb)   |
| Design a least recently used cache                  | [Solution](solutions/object_oriented_design/lru_cache/lru_cache.ipynb)  |
| Design a call center                                | [Solution](solutions/object_oriented_design/call_center/call_center.ipynb)  |
| Design a deck of cards                              | [Solution](solutions/object_oriented_design/deck_of_cards/deck_of_cards.ipynb)  |
| Design a parking lot                                | [Solution](solutions/object_oriented_design/parking_lot/parking_lot.ipynb)  |
| Design a chat server                                | [Solution](solutions/object_oriented_design/online_chat/online_chat.ipynb)  |
| Design a circular array                             | [Contribute](#contributing)                                          |
| Add an object-oriented design question              | [Contribute](#contributing)                                             |

### Index of System Design Topics

> Summaries of various system design topics, including pros and cons. Everything is a trade-off.
>
> Each section contains links to more in-depth resources.

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
*   [Under development](#under-development)
*   [Credits](#credits)
*   [Contact info](#contact-info)
*   [License](#license)

## Contribute

Help make this resource even better! Contribute by:

*   Fixing errors
*   Improving sections
*   Adding new sections
*   [Translate](https://github.com/donnemartin/system-design-primer/issues/28)

Content that needs some polishing is placed [under development](#under-development).

Review the [Contributing Guidelines](CONTRIBUTING.md).

## Other Resources

*   **Anki Flashcards**:  Provided [Anki flashcard decks](https://apps.ankiweb.net/) that use spaced repetition to help you retain key system design concepts.
    *   [System design deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/System%20Design.apkg)
    *   [System design exercises deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/System%20Design%20Exercises.apkg)
    *   [Object oriented design exercises deck](https://github.com/donnemartin/system-design-primer/tree/master/resources/flash_cards/OO%20Design.apkg)

*   **Coding Interview Prep**: If you are looking for resources to help you prep for the [**Coding Interview**](https://github.com/donnemartin/interactive-coding-challenges)
    *   [Coding deck](https://github.com/donnemartin/interactive-coding-challenges/tree/master/anki_cards/Coding.apkg)

## Index of System Design Topics

*   **Performance vs scalability** - Understand the core principles of each.
*   **Latency vs throughput** - Learn about the metrics.
*   **Availability vs consistency** - Explore the CAP theorem.
*   **Consistency patterns** - Weak, eventual, and strong.
*   **Availability patterns** - Failover and replication.
*   **Domain name system**
*   **Content delivery network**
    *   Push CDNs
    *   Pull CDNs
*   **Load balancer**
    *   Active-passive and active-active.
    *   Layer 4 and 7 load balancing
    *   Horizontal scaling
*   **Reverse proxy (web server)**
    *   Load balancer vs reverse proxy
*   **Application layer**
    *   Microservices
    *   Service discovery
*   **Database**
    *   Relational database management system (RDBMS)
        *   Master-slave and master-master replication
        *   Federation and sharding
        *   Denormalization
        *   SQL tuning
    *   NoSQL
        *   Key-value store
        *   Document store
        *   Wide column store
        *   Graph Database
    *   SQL or NoSQL
*   **Cache**
    *   Client, CDN, web server, database, and application caching
    *   Caching at the database query level and object level
    *   When to update the cache (Cache-aside, write-through, write-behind (write-back), refresh-ahead)
*   **Asynchronism**
    *   Message queues
    *   Task queues
    *   Back pressure
*   **Communication**
    *   Transmission control protocol (TCP)
    *   User datagram protocol (UDP)
    *   Remote procedure call (RPC)
    *   Representational state transfer (REST)
*   **Security** - General principles
*   **Appendix** - Valuable reference materials

## Further Reading

The sections below provide links and resources to further expand your knowledge.

### Additional system design interview questions

> Common system design interview questions, with links to resources on how to solve each.

| Question                                                                   | Reference(s)                                                                                                                                                                                                                                                                                                                                                                                       |
| :------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Design a file sync service like Dropbox                                       | [youtube.com](https://www.youtube.com/watch?v=PE4gwstWhmc)                                                                                                                                                                                                                                                                                                                                     |
| Design a search engine like Google                                          | [queue.acm.org](http://queue.acm.org/detail.cfm?id=988407)<br/>[stackexchange.com](http://programmers.stackexchange.com/questions/38324/interview-question-how-would-you-implement-google-search)<br/>[ardendertat.com](http://www.ardendertat.com/2012/01/11/implementing-search-engines/)<br/>[stanford.edu](http://infolab.stanford.edu/~backrub/google.html)                                         |
| Design a scalable web crawler like Google                                   | [quora.com](https://www.quora.com/How-can-I-build-a-web-crawler-from-scratch)                                                                                                                                                                                                                                                                                                                     |
| Design Google docs                                                         | [code.google.com](https://code.google.com/p/google-mobwrite/)<br/>[neil.fraser.name](https://neil.fraser.name/writing/sync/)                                                                                                                                                                                                                                                                        |
| Design a key-value store like Redis                                         | [slideshare.net](http://www.slideshare.net/dvirsky/introduction-to-redis)                                                                                                                                                                                                                                                                                                                     |
| Design a cache system like Memcached                                       | [slideshare.net](http://www.slideshare.net/oemebamo/introduction-to-memcached)                                                                                                                                                                                                                                                                                                                  |
| Design a recommendation system like Amazon's                               | [hulu.com](https://web.archive.org/web/20170406065247/http://tech.hulu.com/blog/2011/09/19/recommendation-system.html)<br/>[ijcai13.org](http://ijcai13.org/files/tutorial_slides/td3.pdf)                                                                                                                                                                                                          |
| Design a tinyurl system like Bitly                                          | [n00tc0d3r.blogspot.com](http://n00tc0d3r.blogspot.com/)                                                                                                                                                                                                                                                                                                                                        |
| Design a chat app like WhatsApp                                            | [highscalability.com](http://highscalability.com/blog/2014/2/26/the-whatsapp-architecture-facebook-bought-for-19-billion.html)                                                                                                                                                                                                                                                            |
| Design a picture sharing system like Instagram                              | [highscalability.com](http://highscalability.com/flickr-architecture)<br/>[highscalability.com](http://highscalability.com/blog/2011/12/6/instagram-architecture-14-million-users-terabytes-of-photos.html)                                                                                                                                                                             |
| Design the Facebook news feed function                                      | [quora.com](http://www.quora.com/What-are-best-practices-for-building-something-like-a-News-Feed)<br/>[quora.com](http://www.quora.com/Activity-Streams/What-are-the-scaling-issues-to-keep-in-mind-while-developing-a-social-network-feed)<br/>[slideshare.net](http://www.slideshare.net/danmckinley/etsy-activity-feeds-architecture)                                                                 |
| Design the Facebook timeline function                                       | [facebook.com](https://www.facebook.com/note.php?note_id=10150468255628920)<br/>[highscalability.com](http://highscalability.com/blog/2012/1/23/facebook-timeline-brought-to-you-by-the-power-of-denormaliza.html)                                                                                                                                                                                       |
| Design the Facebook chat function                                           | [erlang-factory.com](http://www.erlang-factory.com/upload/presentations/31/EugeneLetuchy-ErlangatFacebook.pdf)<br/>[facebook.com](https://www.facebook.com/note.php?note_id=14218138919&id=9445547199&index=0)                                                                                                                                                                                              |
| Design a graph search function like Facebook's                                | [facebook.com](https://www.facebook.com/notes/facebook-engineering/under-the-hood-building-out-the-infrastructure-for-graph-search/10151347573598920)<br/>[facebook.com](https://www.facebook.com/notes/facebook-engineering/under-the-hood-indexing-and-ranking-in-graph-search/10151361720763920)<br/>[facebook.com](https://www.facebook.com/notes/facebook-engineering/under-the-hood-the-natural-language-interface-of-graph-search/10151432733048920) |
| Design a content delivery network like CloudFlare                           | [figshare.com](https://figshare.com/articles/Globally_distributed_content_delivery/6605972)                                                                                                                                                                                                                                                                                                   |
| Design a trending topic system like Twitter's                                | [michael-noll.com](http://www.michael-noll.com/blog/2013/01/18/implementing-real-time-trending-topics-in-storm/)<br/>[snikolov .wordpress.com](http://snikolov.wordpress.com/2012/11/14/early-detection-of-twitter-trends/)                                                                                                                                                                      |
| Design a random ID generation system                                        | [blog.twitter.com](https://blog.twitter.com/2010/announcing-snowflake)<br/>[github.com](https://github.com/twitter/snowflake/)                                                                                                                                                                                                                                                                  |
| Return the top k requests during a time interval                             | [cs.ucsb.edu](https://www.cs.ucsb.edu/sites/default/files/documents/2005-23.pdf)<br/>[wpi.edu](http://davis.wpi.edu/xmdv/docs/EDBT11-diyang.pdf)                                                                                                                                                                                                                                                  |
| Design a system that serves data from multiple data centers                   | [highscalability.com](http://highscalability.com/blog/2009/8/24/how-google-serves-data-from-multiple-datacenters.html)                                                                                                                                                                                                                                                                       |
| Design an online multiplayer card game                                      | [indieflashblog.com](https://web.archive.org/web/20180929181117/http://www.indieflashblog.com/how-to-create-an-asynchronous-multiplayer-game.html)<br/>[buildnewgames.com](http://buildnewgames.com/real-time-multiplayer/)                                                                                                                                                                                 |
| Design a garbage collection system                                          | [stuffwithstuff.com](http://journal.stuffwithstuff.com/2013/12/08/babys-first-garbage-collector/)<br/>[washington.edu](http://courses.cs.washington.edu/courses/csep521/07wi/prj/rick.pdf)                                                                                                                                                                                               |
| Design an API rate limiter                                                    | [https://stripe.com/blog/](https://stripe.com/blog/rate-limiters)                                                                                                                                                                                                                                                                                                                              |
| Design a Stock Exchange (like NASDAQ or Binance)                                          | [Jane Street](https://youtu.be/b1e4t2k2KJY)<br/>[Golang Implementation](https://around25.com/blog/building-a-trading-engine-for-a-crypto-exchange/)<br/>[Go Implementation](http://bhomnick.net/building-a-simple-limit-order-in-go/)                                                                                                                                                                                                 |
| Add a system design question                                                   | [Contribute](#contributing)                                                                                                                                                                                                                                                                                                                                                                           |

### Real world architectures

> Articles on how real world systems are designed.

*   **Don't focus on nitty gritty details for the following articles, instead:**
    *   Identify shared principles, common technologies, and patterns within these articles
    *   Study what problems are solved by each component, where it works, where it doesn't
    *   Review the lessons learned

| Type           | System                 | Reference(s)                                                                                                                               |
| :------------- | :--------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| Data processing | **MapReduce**            | [research.google.com](http://static.googleusercontent.com/media/research.google.com/zh-CN/us/archive/mapreduce-osdi04.pdf)          |
| Data processing | **Spark**                | [slideshare.net](http://www.slideshare.net/AGrishchenko/apache-spark-architecture)                                                  |
| Data processing | **Storm**                | [slideshare.net](http://www.slideshare.net/previa/storm-16094009)                                                                     |
| Data store      | **Bigtable**             | [harvard.edu](http://www.read.seas.harvard.edu/~kohler/class/cs239-w08/chang06bigtable.pdf)                                             |
| Data store      | **HBase**                | [slideshare.net](http://www.slideshare.net/alexbaranau/intro-to-hbase)                                                                  |
| Data store      | **Cassandra**            | [slideshare.net](http://www.slideshare.net/planetcassandra/cassandra-introduction-features-30103666)                                   |
| Data store      | **DynamoDB**             | [harvard.edu](http://www.read.seas.harvard.edu/~kohler/class/cs239-w08/decandia07dynamo.pdf)                                             |
| Data store      | **MongoDB**              | [slideshare.net](http://www.slideshare.net/mdirolf/introduction-to-mongodb)                                                            |
| Data store      | **Spanner**              | [research.google.com](http://research.google.com/archive/spanner-osdi2012.pdf)                                                          |
| Data store      | **Memcached**            | [slideshare.net](http://www.slideshare.net/oemebamo/introduction-to-memcached)                                                           |
| Data store      | **Redis**                | [slideshare.net](http://www.slideshare.net/dvirsky/introduction-to-redis)                                                              |
| File system     | **Google File System (GFS)** | [research.google.com](http://static.googleusercontent.com/media/research.google.com/zh-CN/us/archive/gfs-sosp2003.pdf)          |
| File system     | **Hadoop File System (HDFS)** | [apache.org](http://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)                              |
| Misc            | **Chubby**               | [research.google.com](http://static.googleusercontent.com/external_content/untrusted_dlcp/research.google.com/en/us/archive/chubby-osdi06.pdf) |
| Misc            | **Dapper**               | [research.google.com](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36356.pdf)                        |
| Misc            | **Kafka**                | [slideshare.net](http://www.slideshare.net/mumrah/kafka-talk-tri-hug)                                                                    |
| Misc            | **Zookeeper**            | [slideshare.net](http://www.slideshare.net/sauravhaloi/introduction-to-apache-zookeeper)                                               |

### Company Architectures

| Company         | Reference(s)                                                                                                                                                                                                                                |
| :-------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Amazon          | [Amazon architecture](http://highscalability.com/amazon-architecture)                                                                                                                                                                     |
| Cinchcast       | [Producing 1,500 hours of audio every day](http://highscalability.com/blog/2012/7/16/cinchcast-architecture-producing-1500-hours-of-audio-every-d.html)                                                                                 |
| DataSift        | [Realtime datamining At 120,000 tweets per second](http://highscalability.com/blog/2011/11/29/datasift-architecture-realtime-datamining-at-120000-tweets-p.html)                                                                        |
| Dropbox         | [How we've scaled Dropbox](https://www.youtube.com/watch?v=PE4gwstWhmc)                                                                                                                                                                  |
| ESPN            | [Operating At 100,000 duh nuh nuhs per second](http://highscalability.com/blog/2013/11/4/espns-architecture-at-scale-operating-at-100000-duh-nuh-nuhs.html)                                                                               |
| Google          | [Google architecture](http://highscalability.com/google-architecture)                                                                                                                                                                     |
| Instagram       | [14 million users, terabytes of photos](http://highscalability.com/blog/2011/12/6/instagram-architecture-14-million-users-terabytes-of-photos.html)<br/>[What powers Instagram](http://instagram-engineering.tumblr.com/post/13649370142/what-powers-instagram-hundreds-of-instances) |
| Justin.tv       | [Justin.Tv's live video broadcasting architecture](http://highscalability.com/blog/2010/3/16/justintvs-live-video-broadcasting-architecture.html)                                                                                       |
| Facebook        | [Scaling memcached at Facebook](https://cs.uwaterloo.ca/~brecht/courses/854-Emerging-2014/readings/key-value/fb-memcached-nsdi-2013.pdf)<br/>[TAO: Facebook’s distributed data store for the social graph](https://cs.uwaterloo.ca/~brecht/courses/854-Emerging-2014/readings/data-store/tao-facebook-distributed-datastore-atc-2013.pdf)<br/>[Facebook’s photo storage](https://www.usenix.org/legacy/event/osdi10/tech/full_papers/Beaver.pdf)<br/>[How Facebook Live Streams To 800,000 Simultaneous Viewers](http://highscalability.com/blog/2016/6/27/how-facebook-live-streams-to-800000