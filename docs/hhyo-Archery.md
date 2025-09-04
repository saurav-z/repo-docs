<div align="center">

# Archery: Your Comprehensive SQL Audit and Query Platform

[<img src="https://github.com/hhyo/Archery/wiki/images/dashboard.png" alt="Archery Dashboard" width="800"/>](https://github.com/hhyo/Archery)

Archery provides a centralized platform for SQL auditing, query management, and database operations, simplifying database administration.  Check out the original repo [here](https://github.com/hhyo/Archery).

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

</div>

## Key Features

*   **SQL Auditing:** Review and monitor SQL queries for security and performance.
*   **Query Management:**  Centralized platform for managing and executing SQL queries.
*   **Database Support:**  Supports a wide range of databases.
*   **Database Operations:** Offers features for backup and data dictionary management.
*   **Performance Monitoring:** Monitor slow query logs.
*   **User and Parameter Management:** Manages user accounts and database parameters.

## Supported Databases

| Database        | Query | Audit | Execute | Backup | Data Dictionary | Slow Logs | Session Management | Account Management | Parameter Management | Data Archiving |
|-----------------|-------|-------|---------|--------|-----------------|-----------|--------------------|--------------------|----------------------|----------------|
| MySQL           | √     | √     | √       | √      | √               | √         | √                  | √                  | √                    | √              |
| MsSQL           | √     | ×     | √       | ×      | √               | ×         | ×                  | ×                  | ×                    | ×              |
| Redis           | √     | ×     | √       | ×      | ×               | ×         | ×                  | ×                  | ×                    | ×              |
| PgSQL           | √     | ×     | √       | ×      | ×               | ×         | ×                  | ×                  | ×                    | ×              |
| Oracle          | √     | √     | √       | √      | √               | ×         | √                  | ×                  | ×                    | ×              |
| MongoDB         | √     | √     | √       | ×      | ×               | ×         | √                  | √                  | ×                    | ×              |
| Phoenix         | √     | ×     | √       | ×      | ×               | ×         | ×                  | ×                  | ×                    | ×              |
| ODPS            | √     | ×     | ×       | ×      | ×               | ×         | ×                  | ×                  | ×                    | ×              |
| ClickHouse      | √     | √     | √       | ×      | ×               | ×         | ×                  | ×                  | ×                    | ×              |
| Cassandra       | √     | ×     | √       | ×      | ×               | ×         | ×                  | ×                  | ×                    | ×              |
| Doris           | √     | ×     | √       | ×      | ×               | ×         | ×                  | ×                  | ×                    | ×              |

## Quick Start

### System Demo
Try it out using the online demo!

[Online Demo](https://demo.archerydms.com)

| Account   | Password  |
| --------- | --------- |
| archer    | archer    |

### Docker

Refer to the documentation for using Docker: [Docker Instructions](https://github.com/hhyo/archery/wiki/docker)

## Manual Installation

Detailed installation instructions: [Manual Installation](https://github.com/hhyo/archery/wiki/manual)

## Running Tests

```bash
python manage.py test -v 3
```

## Dependencies

### Frameworks & Libraries

*   [Django](https://github.com/django/django)
*   [Bootstrap](https://github.com/twbs/bootstrap)
*   [jQuery](https://github.com/jquery/jquery)

### Frontend Components

*   Menu: [metisMenu](https://github.com/onokumus/metismenu)
*   Theme: [sb-admin-2](https://github.com/BlackrockDigital/startbootstrap-sb-admin-2)
*   Editor: [ace](https://github.com/ajaxorg/ace)
*   SQL Formatter: [sql-formatter](https://github.com/zeroturnaround/sql-formatter)
*   Tables: [bootstrap-table](https://github.com/wenzhixin/bootstrap-table)
*   Table Editing: [bootstrap-editable](https://github.com/vitalets/x-editable)
*   Dropdown: [bootstrap-select](https://github.com/snapappointments/bootstrap-select)
*   File Upload: [bootstrap-fileinput](https://github.com/kartik-v/bootstrap-fileinput)
*   Date/Time Picker: [bootstrap-datetimepicker](https://github.com/smalot/bootstrap-datetimepicker)
*   Date Range Picker: [daterangepicker](https://github.com/dangrossman/daterangepicker)
*   Switch: [bootstrap-switch](https://github.com/Bttstrp/bootstrap-switch)
*   Markdown: [marked](https://github.com/markedjs/marked)

### Backend Components

*   Queue: [django-q](https://github.com/Koed00/django-q)
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
*   Date/Time: [python-dateutil](https://github.com/paxan/python-dateutil)

### Feature Dependencies

*   Visualization: [pyecharts](https://github.com/pyecharts/pyecharts)
*   MySQL Audit/Execution/Backup: [goInception](https://github.com/hanchuanchuan/goInception) | [inception](https://github.com/hhyo/inception)
*   MySQL Index Optimization: [SQLAdvisor](https://github.com/Meituan-Dianping/SQLAdvisor)
*   SQL Optimization/Compression: [SOAR](https://github.com/XiaoMi/soar)
*   My2SQL: [my2sql](https://github.com/liuhr/my2sql)
*   Table Structure Sync: [SchemaSync](https://github.com/hhyo/SchemaSync)
*   Slow Log Parsing: [pt-query-digest](https://www.percona.com/doc/percona-toolkit/3.0/pt-query-digest.html) | [aquila_v2](https://github.com/thinkdb/aquila_v2)
*   Large Table DDL: [gh-ost](https://github.com/github/gh-ost) | [pt-online-schema-change](https://www.percona.com/doc/percona-toolkit/3.0/pt-online-schema-change.html)
*   MyBatis XML Parsing: [mybatis-mapper2sql](https://github.com/hhyo/mybatis-mapper2sql)
*   RDS Management: [aliyun-openapi-python-sdk](https://github.com/aliyun/aliyun-openapi-python-sdk)
*   Data Encryption: [django-mirage-field](https://github.com/luojilab/django-mirage-field)

## Contributing

Check the project's development plan and dependency list, and contribute by responding to corresponding Issues or submitting a PR.  Your contributions are highly valued!

Contributions include, but are not limited to:

*   [Wiki Documentation](https://github.com/hhyo/Archery/wiki) (Open for editing)
*   Bug fixes
*   New feature submissions
*   Code optimization
*   Test case improvements

## Feedback and Community

*   For usage questions and feature requests: [Discussions](https://github.com/hhyo/Archery/discussions)
*   To report bugs: [Issues](https://github.com/hhyo/archery/issues)

## Acknowledgements

*   [archer](https://github.com/jly8866/archer) - Archery is based on the secondary development of archer.
*   [goInception](https://github.com/hanchuanchuan/goInception) - A MySQL operation and maintenance tool that integrates audit, execution, backup, and rollback statement generation.
*   [JetBrains Open Source](https://www.jetbrains.com/zh-cn/opensource/?from=archery) - Provides free IDE licenses for the project.

  [<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.png" width="200"/>](https://www.jetbrains.com/opensource/)

```

Key improvements and explanations:

*   **SEO Optimization:**  The title is now a clear and concise "Archery: Your Comprehensive SQL Audit and Query Platform," using keywords relevant to the project's function. The description includes keywords as well (SQL audit, query management, database operations).
*   **Hook:** A concise, attention-grabbing first sentence that summarizes the project's core value proposition.
*   **Clear Headings:** Consistent and informative headings.
*   **Bulleted Key Features:** Uses bullet points for easy readability and to highlight the core functionalities.
*   **Concise Language:** The text is streamlined and avoids unnecessary jargon.
*   **Contextual Links:** Links are used to provide context and improve navigation.  The direct link back to the repo is prominent.
*   **Organization:** The information is presented in a logical and easy-to-scan format.
*   **Call to Action (Quick Start):**  Encourages immediate use with the demo.
*   **Focus on Value:** The revised README emphasizes the benefits of using Archery (simplified database administration).