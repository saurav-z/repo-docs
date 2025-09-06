<div align="center">

# Archery: Your SQL Audit and Query Platform

Archery is a powerful and versatile platform designed to audit and manage SQL queries, offering comprehensive features for database administrators and developers.

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

<div align="center">
    <img src="https://github.com/hhyo/Archery/wiki/images/dashboard.png" alt="Archery Dashboard" width="800">
</div>

## Key Features

*   **Comprehensive Database Support:**
    *   MySQL, MsSQL, Redis, PgSQL, Oracle, MongoDB, Phoenix, ODPS, ClickHouse, Cassandra, Doris.
*   **Core Functionality:**
    *   SQL Query Execution & Auditing
    *   Database Backup and Restore
    *   Data Dictionary Management
    *   Slow Query Log Analysis
    *   Session Management
    *   User and Role Management
    *   Parameter Management
    *   Data Archiving

### Database Feature Support Matrix

| Database     | Query | Audit | Execute | Backup | Data Dictionary | Slow Log | Session Management | User Management | Parameter Management | Data Archiving |
|--------------|-------|-------|---------|--------|-----------------|----------|--------------------|-----------------|--------------------|----------------|
| MySQL        | √     | √     | √       | √      | √               | √        | √                  | √               | √                  | √              |
| MsSQL        | √     | ×     | √       | ×      | √               | ×        | ×                  | ×               | ×                  | ×              |
| Redis        | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×               | ×                  | ×              |
| PgSQL        | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×               | ×                  | ×              |
| Oracle       | √     | √     | √       | √      | √               | ×        | √                  | ×               | ×                  | ×              |
| MongoDB      | √     | √     | √       | ×      | ×               | ×        | √                  | √               | ×                  | ×              |
| Phoenix      | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×               | ×                  | ×              |
| ODPS         | √     | ×     | ×       | ×      | ×               | ×        | ×                  | ×               | ×                  | ×              |
| ClickHouse   | √     | √     | √       | ×      | ×               | ×        | ×                  | ×               | ×                  | ×              |
| Cassandra    | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×               | ×                  | ×              |
| Doris        | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×               | ×                  | ×              |

## Getting Started

### Online Demo

*   **Try it now:** [Online Demo](https://demo.archerydms.com)
    *   **Username:** archer
    *   **Password:** archer

### Docker

*   See the [Docker documentation](https://github.com/hhyo/archery/wiki/docker) for easy deployment.

### Manual Installation

*   Detailed installation instructions can be found in the [manual](https://github.com/hhyo/archery/wiki/manual).

## Running Tests

```bash
python manage.py test -v 3
```

## Dependencies

### Frameworks
- [Django](https://github.com/django/django)
- [Bootstrap](https://github.com/twbs/bootstrap)
- [jQuery](https://github.com/jquery/jquery)

### Frontend Components

*   **Menu:** [metisMenu](https://github.com/onokumus/metismenu)
*   **Theme:** [sb-admin-2](https://github.com/BlackrockDigital/startbootstrap-sb-admin-2)
*   **Editor:** [ace](https://github.com/ajaxorg/ace)
*   **SQL Formatting:** [sql-formatter](https://github.com/zeroturnaround/sql-formatter)
*   **Tables:** [bootstrap-table](https://github.com/wenzhixin/bootstrap-table)
*   **Table Editing:** [bootstrap-editable](https://github.com/vitalets/x-editable)
*   **Dropdowns:** [bootstrap-select](https://github.com/snapappointments/bootstrap-select)
*   **File Upload:** [bootstrap-fileinput](https://github.com/kartik-v/bootstrap-fileinput)
*   **Date/Time Picker:** [bootstrap-datetimepicker](https://github.com/smalot/bootstrap-datetimepicker)
*   **Date Range Picker:** [daterangepicker](https://github.com/dangrossman/daterangepicker)
*   **Switches:** [bootstrap-switch](https://github.com/Bttstrp/bootstrap-switch)
*   **Markdown Display:** [marked](https://github.com/markedjs/marked)

### Backend Components

*   **Task Queue:** [django-q](https://github.com/Koed00/django-q)
*   **MySQL Connector:** [mysqlclient-python](https://github.com/PyMySQL/mysqlclient-python)
*   **MsSQL Connector:** [pyodbc](https://github.com/mkleehammer/pyodbc)
*   **Redis Connector:** [redis-py](https://github.com/andymccurdy/redis-py)
*   **PostgreSQL Connector:** [psycopg2](https://github.com/psycopg/psycopg2)
*   **Oracle Connector:** [cx_Oracle](https://github.com/oracle/python-cx_Oracle)
*   **MongoDB Connector:** [pymongo](https://github.com/mongodb/mongo-python-driver)
*   **Phoenix Connector:** [phoenixdb](https://github.com/lalinsky/python-phoenixdb)
*   **ODPS Connector:** [pyodps](https://github.com/aliyun/aliyun-odps-python-sdk)
*   **ClickHouse Connector:** [clickhouse-driver](https://github.com/mymarilyn/clickhouse-driver)
*   **SQL Parsing:** [sqlparse](https://github.com/andialbrecht/sqlparse)
*   **MySQL Binlog:** [python-mysql-replication](https://github.com/noplay/python-mysql-replication)
*   **LDAP:** [django-auth-ldap](https://github.com/django-auth-ldap/django-auth-ldap)
*   **Serialization:** [simplejson](https://github.com/simplejson/simplejson)
*   **Date/Time Handling:** [python-dateutil](https://github.com/paxan/python-dateutil)

### Functional Dependencies

*   **Visualization:** [pyecharts](https://github.com/pyecharts/pyecharts)
*   **MySQL Audit/Execution/Backup:** [goInception](https://github.com/hanchuanchuan/goInception) | [inception](https://github.com/hhyo/inception)
*   **MySQL Index Optimization:** [SQLAdvisor](https://github.com/Meituan-Dianping/SQLAdvisor)
*   **SQL Optimization:** [SOAR](https://github.com/XiaoMi/soar)
*   **My2SQL:** [my2sql](https://github.com/liuhr/my2sql)
*   **Schema Synchronization:** [SchemaSync](https://github.com/hhyo/SchemaSync)
*   **Slow Log Analysis:** [pt-query-digest](https://www.percona.com/doc/percona-toolkit/3.0/pt-query-digest.html) | [aquila_v2](https://github.com/thinkdb/aquila_v2)
*   **Large Table DDL:** [gh-ost](https://github.com/github/gh-ost) | [pt-online-schema-change](https://www.percona.com/doc/percona-toolkit/3.0/pt-online-schema-change.html)
*   **MyBatis XML Parsing:** [mybatis-mapper2sql](https://github.com/hhyo/mybatis-mapper2sql)
*   **RDS Management:** [aliyun-openapi-python-sdk](https://github.com/aliyun/aliyun-openapi-python-sdk)
*   **Data Encryption:** [django-mirage-field](https://github.com/luojilab/django-mirage-field)

## Contributing

We welcome contributions!  Review the project's [development plan](https://github.com/hhyo/Archery/wiki/development-plan) and the dependency list to get started.  You can claim issues or submit pull requests.

Contribution methods include:

*   [Wiki Documentation](https://github.com/hhyo/Archery/wiki) (Open for editing)
*   Bug Fixes
*   New Feature Submissions
*   Code Optimization
*   Test Case Improvements

## Get in Touch

*   **Discussions:** [Discussions](https://github.com/hhyo/Archery/discussions) for usage questions and feature requests.
*   **Bug Reports:** [Issues](https://github.com/hhyo/archery/issues) to report bugs.

## Acknowledgements

*   **archer** ([https://github.com/jly8866/archer](https://github.com/jly8866/archer)):  Archery is built upon the foundation of the archer project.
*   **goInception** ([https://github.com/hanchuanchuan/goInception](https://github.com/hanchuanchuan/goInception)): A MySQL operation tool that integrates audit, execution, backup, and rollback statement generation.
*   **JetBrains Open Source**  ([https://www.jetbrains.com/zh-cn/opensource/?from=archery](https://www.jetbrains.com/zh-cn/opensource/?from=archery)) Provides free IDE licenses for the project.
    [<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.png" width="200"/>](https://www.jetbrains.com/opensource/)

**[Back to the Archery Repository](https://github.com/hhyo/Archery)**
```
Key improvements and SEO optimizations:

*   **Concise Title:**  Kept the core title "Archery" and added a more descriptive subtitle.
*   **Hook:**  A clear, compelling one-sentence introduction to immediately explain the project's value.
*   **Keywords:** Incorporated relevant keywords throughout (SQL, audit, query, database, management, etc.)
*   **Structured Content:**  Organized the README with clear headings and subheadings for readability and SEO.
*   **Feature Highlighting:**  Used bullet points to make the key features easily scannable.  Made the feature list more readable and useful.
*   **Database Feature Support Table:** Added a table summarizing supported features by database for quick comparison.
*   **Call to Action:**  "Try it now" for the demo.
*   **Clear Links:**  Included links for documentation, FAQ, releases, and back to the original repo at the end.
*   **SEO-Friendly Formatting:** Used Markdown headings and bullet points effectively.
*   **Removed Redundancy:**  Combined similar sections for brevity.  Simplified the language where possible.
*   **Added Alt Text:** Added descriptive alt text to the image.
*   **Expanded on Functionality:** Clarified the various functions like execution and backup.
*   **Dependency Formatting:** Enhanced the dependency lists to make the titles more prominent.