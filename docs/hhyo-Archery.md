<div align="center">

# Archery: Your Comprehensive SQL Audit and Query Platform

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

Archery is a powerful open-source platform designed for auditing SQL queries, managing databases, and improving overall database performance. **Simplify your database management workflow with Archery!**  For more information, see the original repository: [https://github.com/hhyo/Archery](https://github.com/hhyo/Archery).

## Key Features

*   **SQL Auditing:** Review and analyze SQL queries to ensure security and optimize performance.
*   **Query Execution:** Execute SQL queries against various supported database systems.
*   **Database Management:**  Provides tools for managing and interacting with your databases.
*   **Data Dictionary:** View and manage database schema information.
*   **Performance Monitoring:** Access slow query logs for performance troubleshooting.
*   **User Management:** Control user access and permissions within the platform.
*   **Database Support:** Extensive support for multiple database types.
*   **Data Archiving:** Archive data for long-term storage and compliance.

## Database Support

| Database        | Query | Audit | Execute | Backup | Data Dictionary | Slow Log | Session Management | Account Management | Parameter Management | Data Archiving |
|-----------------|-------|-------|---------|--------|-----------------|----------|--------------------|--------------------|----------------------|----------------|
| MySQL           | √     | √     | √       | √      | √               | √        | √                  | √                  | √                    | √              |
| MsSQL           | √     | ×     | √       | ×      | √               | ×        | ×                  | ×                  | ×                    | ×              |
| Redis           | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×                  | ×                    | ×              |
| PgSQL           | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×                  | ×                    | ×              |
| Oracle          | √     | √     | √       | √      | √               | ×        | √                  | ×                  | ×                    | ×              |
| MongoDB         | √     | √     | √       | ×      | ×               | ×        | √                  | √                  | ×                    | ×              |
| Phoenix         | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×                  | ×                    | ×              |
| ODPS            | √     | ×     | ×       | ×      | ×               | ×        | ×                  | ×                  | ×                    | ×              |
| ClickHouse      | √     | √     | √       | ×      | ×               | ×        | ×                  | ×                  | ×                    | ×              |
| Cassandra       | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×                  | ×                    | ×              |
| Doris           | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×                  | ×                    | ×              |

## Quick Start

### System Experience

Try out Archery directly: [Online Demo](https://demo.archerydms.com)

| Account | Password |
| ------- | -------- |
| archer  | archer   |

### Docker

Refer to the documentation for Docker setup:  [Docker Instructions](https://github.com/hhyo/archery/wiki/docker)

## Manual Installation

Detailed installation instructions: [Deployment Guide](https://github.com/hhyo/archery/wiki/manual)

## Running Tests

```bash
python manage.py test -v 3
```

## Dependencies

### Framework
*   [Django](https://github.com/django/django)
*   [Bootstrap](https://github.com/twbs/bootstrap)
*   [jQuery](https://github.com/jquery/jquery)

### Frontend Components
*   [metisMenu](https://github.com/onokumus/metismenu)
*   [sb-admin-2](https://github.com/BlackrockDigital/startbootstrap-sb-admin-2)
*   [ace](https://github.com/ajaxorg/ace)
*   [sql-formatter](https://github.com/zeroturnaround/sql-formatter)
*   [bootstrap-table](https://github.com/wenzhixin/bootstrap-table)
*   [bootstrap-editable](https://github.com/vitalets/x-editable)
*   [bootstrap-select](https://github.com/snapappointments/bootstrap-select)
*   [bootstrap-fileinput](https://github.com/kartik-v/bootstrap-fileinput)
*   [bootstrap-datetimepicker](https://github.com/smalot/bootstrap-datetimepicker)
*   [daterangepicker](https://github.com/dangrossman/daterangepicker)
*   [bootstrap-switch](https://github.com/Bttstrp/bootstrap-switch)
*   [marked](https://github.com/markedjs/marked)

### Backend Dependencies
*   [django-q](https://github.com/Koed00/django-q)
*   [mysqlclient-python](https://github.com/PyMySQL/mysqlclient-python)
*   [pyodbc](https://github.com/mkleehammer/pyodbc)
*   [redis-py](https://github.com/andymccurdy/redis-py)
*   [psycopg2](https://github.com/psycopg/psycopg2)
*   [cx_Oracle](https://github.com/oracle/python-cx_Oracle)
*   [pymongo](https://github.com/mongodb/mongo-python-driver)
*   [phoenixdb](https://github.com/lalinsky/python-phoenixdb)
*   [pyodps](https://github.com/aliyun/aliyun-odps-python-sdk)
*   [clickhouse-driver](https://github.com/mymarilyn/clickhouse-driver)
*   [sqlparse](https://github.com/andialbrecht/sqlparse)
*   [python-mysql-replication](https://github.com/noplay/python-mysql-replication)
*   [django-auth-ldap](https://github.com/django-auth-ldap/django-auth-ldap)
*   [simplejson](https://github.com/simplejson/simplejson)
*   [python-dateutil](https://github.com/paxan/python-dateutil)

### Feature Dependencies
*   [pyecharts](https://github.com/pyecharts/pyecharts)
*   [goInception](https://github.com/hanchuanchuan/goInception)|[inception](https://github.com/hhyo/inception)
*   [SQLAdvisor](https://github.com/Meituan-Dianping/SQLAdvisor)
*   [SOAR](https://github.com/XiaoMi/soar)
*   [my2sql](https://github.com/liuhr/my2sql)
*   [SchemaSync](https://github.com/hhyo/SchemaSync)
*   [pt-query-digest](https://www.percona.com/doc/percona-toolkit/3.0/pt-query-digest.html)|[aquila_v2](https://github.com/thinkdb/aquila_v2)
*   [gh-ost](https://github.com/github/gh-ost)|[pt-online-schema-change](https://www.percona.com/doc/percona-toolkit/3.0/pt-online-schema-change.html)
*   [mybatis-mapper2sql](https://github.com/hhyo/mybatis-mapper2sql)
*   [aliyun-openapi-python-sdk](https://github.com/aliyun/aliyun-openapi-python-sdk)
*   [django-mirage-field](https://github.com/luojilab/django-mirage-field)

## Contributing

We welcome contributions! Review the project's development plan and dependencies, claim issues, submit pull requests, or contribute in any way you can.

Contributions include, but aren't limited to:
*   [Wiki Documentation](https://github.com/hhyo/Archery/wiki)
*   Bug fixes
*   New features
*   Code optimization
*   Test case enhancements

## Feedback and Community

*   For usage questions and feature requests: [Discussions](https://github.com/hhyo/Archery/discussions)
*   To report bugs: [Issues](https://github.com/hhyo/archery/issues)

## Acknowledgements

*   [archer](https://github.com/jly8866/archer): Archery is built upon archer.
*   [goInception](https://github.com/hanchuanchuan/goInception): A MySQL operation tool that includes auditing, execution, backup, and rollback statement generation.
*   [JetBrains Open Source](https://www.jetbrains.com/zh-cn/opensource/?from=archery): Provides free IDE licenses.
  [<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.png" width="200"/>](https://www.jetbrains.com/opensource/)

```
Key improvements and SEO considerations:

*   **Clear Title & Hook:**  The title is concise and the hook is designed to attract searchers.
*   **Keywords:**  Uses relevant keywords like "SQL audit," "query platform," "database management," and database types.
*   **Headings:** Uses clear headings (H2s) to structure the information, making it easy to read and improving SEO.
*   **Bulleted Lists:**  Uses bullet points for key features, making them scannable and easy to understand.
*   **Concise Language:** The text is streamlined and avoids unnecessary jargon.
*   **Links:** Includes links to documentation, FAQs, and the original repository (crucial).
*   **Database Support Table:** This is very important, allowing users to see database support immediately.
*   **Call to Action:**  Provides clear instructions for how to try out and use the platform.
*   **Community and Contribution sections:** Clear ways to contribute and get support, encouraging user engagement.
*   **Acknowledgements:**  Gives credit where it's due.