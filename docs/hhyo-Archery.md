<div align="center">

# Archery: SQL Audit and Query Platform

**Archery is a powerful SQL audit and query platform designed to streamline database management and improve data security.**

[![Django CI](https://github.com/hhyo/Archery/actions/workflows/django.yml/badge.svg)](https://github.com/hhyo/Archery/actions/workflows/django.yml)
[![Release](https://img.shields.io/github/release/hhyo/archery.svg)](https://github.com/hhyo/archery/releases/)
[![codecov](https://codecov.io/gh/hhyo/archery/branch/master/graph/badge.svg)](https://codecov.io/gh/hhyo/archery)
[![version](https://img.shields.io/pypi/pyversions/django)](https://img.shields.io/pypi/pyversions/django/)
[![version](https://img.shields.io/badge/django-4.1-brightgreen.svg)](https://docs.djangoproject.com/zh-hans/4.1/)
[![Publish Docker image](https://github.com/hhyo/Archery/actions/workflows/docker-image.yml/badge.svg)](https://github.com/hhyo/Archery/actions/workflows/docker-image.yml)
[![docker_pulls](https://img.shields.io/docker/pulls/hhyo/archery.svg)](https://hub.docker.com/r/hhyo/archery/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](http://github.com/hhyo/archery/blob/master/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Documentation](https://archerydms.com/) | [FAQ](https://github.com/hhyo/archery/wiki/FAQ) | [Releases](https://github.com/hhyo/archery/releases/)

![](https://github.com/hhyo/Archery/wiki/images/dashboard.png)

</div>

## Key Features

*   **Comprehensive Database Support:**  Supports a wide range of databases including MySQL, MsSQL, Redis, PgSQL, Oracle, MongoDB, Phoenix, ODPS, ClickHouse, Cassandra, and Doris.
*   **SQL Auditing:** Review and approve SQL queries before execution to enhance data security.
*   **SQL Querying:** Execute and manage SQL queries with ease.
*   **Data Backup and Recovery:**  Facilitates data backup and recovery operations.
*   **Data Dictionary:** Provides access to data dictionaries for various databases.
*   **Slow Query Log Analysis:** Analyze and optimize database performance with slow query log support.
*   **Session Management:** Offers robust session management capabilities.
*   **User and Parameter Management:**  Manage users, roles, and database parameters efficiently.
*   **Data Archiving:** Supports data archiving for long-term data storage.

## Supported Databases (Feature Matrix)

| Database      | Query | Audit | Execute | Backup | Data Dictionary | Slow Log | Session Mgmt | User Mgmt | Parameter Mgmt | Data Archiving |
|---------------|-------|-------|---------|--------|-----------------|----------|--------------|-------------|----------------|----------------|
| MySQL         | √     | √     | √       | √      | √               | √        | √            | √           | √              | √              |
| MsSQL         | √     | ×     | √       | ×      | √               | ×        | ×            | ×           | ×              | ×              |
| Redis         | √     | ×     | √       | ×      | ×               | ×        | ×            | ×           | ×              | ×              |
| PgSQL         | √     | ×     | √       | ×      | ×               | ×        | ×            | ×           | ×              | ×              |
| Oracle        | √     | √     | √       | √      | √               | ×        | √            | ×           | ×              | ×              |
| MongoDB       | √     | √     | √       | ×      | ×               | ×        | √            | √           | ×              | ×              |
| Phoenix       | √     | ×     | √       | ×      | ×               | ×        | ×            | ×           | ×              | ×              |
| ODPS          | √     | ×     | ×       | ×      | ×               | ×        | ×            | ×           | ×              | ×              |
| ClickHouse    | √     | √     | √       | ×      | ×               | ×        | ×            | ×           | ×              | ×              |
| Cassandra     | √     | ×     | √       | ×      | ×               | ×        | ×            | ×           | ×              | ×              |
| Doris         | √     | ×     | √       | ×      | ×               | ×        | ×            | ×           | ×              | ×              |

## Quick Start

### System Experience

*   **Online Demo:** [https://demo.archerydms.com](https://demo.archerydms.com)

    | Account  | Password |
    | -------- | -------- |
    | archer   | archer   |

### Docker

*   Refer to the [Docker Wiki](https://github.com/hhyo/archery/wiki/docker) for instructions.

## Manual Installation

*   [Deployment Instructions](https://github.com/hhyo/archery/wiki/manual)

## Running Tests

```bash
python manage.py test -v 3
```

## Dependencies

### Frameworks

*   [Django](https://github.com/django/django)
*   [Bootstrap](https://github.com/twbs/bootstrap)
*   [jQuery](https://github.com/jquery/jquery)

### Frontend Components

*   Menu: [metisMenu](https://github.com/onokumus/metismenu)
*   Theme: [sb-admin-2](https://github.com/BlackrockDigital/startbootstrap-sb-admin-2)
*   Editor: [ace](https://github.com/ajaxorg/ace)
*   SQL Formatting: [sql-formatter](https://github.com/zeroturnaround/sql-formatter)
*   Table: [bootstrap-table](https://github.com/wenzhixin/bootstrap-table)
*   Table Editing: [bootstrap-editable](https://github.com/vitalets/x-editable)
*   Dropdown: [bootstrap-select](https://github.com/snapappointments/bootstrap-select)
*   File Upload: [bootstrap-fileinput](https://github.com/kartik-v/bootstrap-fileinput)
*   Date/Time Picker: [bootstrap-datetimepicker](https://github.com/smalot/bootstrap-datetimepicker)
*   Date Range Picker: [daterangepicker](https://github.com/dangrossman/daterangepicker)
*   Switch: [bootstrap-switch](https://github.com/Bttstrp/bootstrap-switch)
*   Markdown Display: [marked](https://github.com/markedjs/marked)

### Server-Side Components

*   Task Queue: [django-q](https://github.com/Koed00/django-q)
*   MySQL Connector: [mysqlclient-python](https://github.com/PyMySQL/mysqlclient-python)
*   MsSQL Connector: [pyodbc](https://github.com/mkleehammer/pyodbc)
*   Redis Connector: [redis-py](https://github.com/andymccurdy/redis-py)
*   PostgreSQL Connector: [psycopg2](https://github.com/psycopg/psycopg2)
*   Oracle Connector: [cx_Oracle](https://github.com/oracle/python-cx_Oracle)
*   MongoDB Connector: [pymongo](https://github.com/mongodb/mongo-python-driver)
*   Phoenix Connector: [phoenixdb](https://github.com/lalinsky/python-phoenixdb)
*   ODPS Connector: [pyodps](https://github.com/aliyun/aliyun-odps-python-sdk)
*   ClickHouse Connector: [clickhouse-driver](https://github.com/mymarilyn/clickhouse-driver)
*   SQL Parsing: [sqlparse](https://github.com/andialbrecht/sqlparse)
*   MySQL Binlog: [python-mysql-replication](https://github.com/noplay/python-mysql-replication)
*   LDAP: [django-auth-ldap](https://github.com/django-auth-ldap/django-auth-ldap)
*   Serialization: [simplejson](https://github.com/simplejson/simplejson)
*   Date/Time Utilities: [python-dateutil](https://github.com/paxan/python-dateutil)

### Feature Dependencies

*   Visualization: [pyecharts](https://github.com/pyecharts/pyecharts)
*   MySQL Audit/Execute/Backup: [goInception](https://github.com/hanchuanchuan/goInception) | [inception](https://github.com/hhyo/inception)
*   MySQL Index Optimization: [SQLAdvisor](https://github.com/Meituan-Dianping/SQLAdvisor)
*   SQL Optimization/Compression: [SOAR](https://github.com/XiaoMi/soar)
*   My2SQL: [my2sql](https://github.com/liuhr/my2sql)
*   Table Structure Sync: [SchemaSync](https://github.com/hhyo/SchemaSync)
*   Slow Log Analysis: [pt-query-digest](https://www.percona.com/doc/percona-toolkit/3.0/pt-query-digest.html) | [aquila_v2](https://github.com/thinkdb/aquila_v2)
*   Large Table DDL: [gh-ost](https://github.com/github/gh-ost) | [pt-online-schema-change](https://www.percona.com/doc/percona-toolkit/3.0/pt-online-schema-change.html)
*   MyBatis XML Parsing: [mybatis-mapper2sql](https://github.com/hhyo/mybatis-mapper2sql)
*   RDS Management: [aliyun-openapi-python-sdk](https://github.com/aliyun/aliyun-openapi-python-sdk)
*   Data Encryption: [django-mirage-field](https://github.com/luojilab/django-mirage-field)

## Contributing

Explore the development plan and dependencies to identify areas for contribution.  Submit PRs or respond to corresponding issues. Your contributions to Archery are appreciated.

Contribution methods include, but are not limited to:

*   [Wiki Documentation](https://github.com/hhyo/Archery/wiki) (Open for editing)
*   Bug Fixes
*   New Feature Submissions
*   Code Optimization
*   Test Case Improvements

## Feedback & Discussions

*   **Use Cases, Feature Requests, Questions:** [Discussions](https://github.com/hhyo/Archery/discussions)
*   **Bug Reports:** [Issues](https://github.com/hhyo/archery/issues)

## Acknowledgements

*   [archer](https://github.com/jly8866/archer): Archery is built upon the foundation of the archer project.
*   [goInception](https://github.com/hanchuanchuan/goInception): A MySQL operations tool with audit, execution, backup, and rollback generation capabilities.
*   [JetBrains Open Source](https://www.jetbrains.com/zh-cn/opensource/?from=archery) for providing free IDE licenses for the project.
  [<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.png" width="200"/>](https://www.jetbrains.com/opensource/)

##  [Back to the Archery Repository](https://github.com/hhyo/Archery)
```

Key improvements and SEO considerations:

*   **Clear, Concise Hook:** The opening sentence is a concise and benefit-driven description, great for drawing in users.
*   **Keyword-Rich Headings:** Uses relevant keywords like "SQL Audit," "Query Platform," and database names.
*   **Feature-Focused Structure:** The use of bullet points makes the key benefits easily scannable.
*   **Database-Specific Focus:** The "Supported Databases" section is very useful.
*   **Call to Action:** Directs users to the repo for more information at the bottom, instead of just a link in the beginning, which is the primary goal of the README.
*   **Well-Organized and Readable:** The structure is improved for readability.
*   **Clear Sectioning:**  Uses headings and subheadings to organize the document.
*   **Included full Dependency list** (helpful to users)
*   **Complete Feature Table**