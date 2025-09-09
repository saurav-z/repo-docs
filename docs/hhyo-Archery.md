# Archery: Your Centralized SQL Audit and Query Platform

Archery is an open-source SQL auditing and query platform designed to streamline database management and improve data security.  [Explore the original repository](https://github.com/hhyo/Archery).

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

## Key Features

*   **Database Support:**
    *   MySQL
    *   MsSQL
    *   Redis
    *   PgSQL
    *   Oracle
    *   MongoDB
    *   Phoenix
    *   ODPS
    *   ClickHouse
    *   Cassandra
    *   Doris
*   **Core Functionality:**
    *   SQL Querying
    *   SQL Auditing
    *   SQL Execution
    *   Database Backup
    *   Data Dictionary
    *   Slow Query Log Analysis
    *   Session Management
    *   User Account Management
    *   Parameter Management
    *   Data Archiving

## Quick Start

### System Demo

Try out a live demo to experience Archery's capabilities firsthand.

*   **Demo URL:** [Online Demo](https://demo.archerydms.com)
*   **Login Credentials:**
    *   Username: `archer`
    *   Password: `archer`

### Docker

See the Docker documentation for easy deployment:  [Docker Instructions](https://github.com/hhyo/archery/wiki/docker)

## Manual Installation

Detailed installation instructions are available: [Manual Installation Guide](https://github.com/hhyo/archery/wiki/manual)

## Running Tests

Execute the test suite:

```bash
python manage.py test -v 3
```

## Dependencies

### Frameworks
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

### Backend Dependencies and Connectors
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

### Feature-Specific Dependencies
*   [pyecharts](https://github.com/pyecharts/pyecharts)
*   [goInception](https://github.com/hanchuanchuan/goInception)
*   [inception](https://github.com/hhyo/inception)
*   [SQLAdvisor](https://github.com/Meituan-Dianping/SQLAdvisor)
*   [SOAR](https://github.com/XiaoMi/soar)
*   [my2sql](https://github.com/liuhr/my2sql)
*   [SchemaSync](https://github.com/hhyo/SchemaSync)
*   [pt-query-digest](https://www.percona.com/doc/percona-toolkit/3.0/pt-query-digest.html)
*   [aquila_v2](https://github.com/thinkdb/aquila_v2)
*   [gh-ost](https://github.com/github/gh-ost)
*   [pt-online-schema-change](https://www.percona.com/doc/percona-toolkit/3.0/pt-online-schema-change.html)
*   [mybatis-mapper2sql](https://github.com/hhyo/mybatis-mapper2sql)
*   [aliyun-openapi-python-sdk](https://github.com/aliyun/aliyun-openapi-python-sdk)
*   [django-mirage-field](https://github.com/luojilab/django-mirage-field)

## Contributing

Contribute to Archery by:

*   Adding to the [Wiki documentation](https://github.com/hhyo/Archery/wiki)
*   Fixing bugs
*   Adding new features
*   Optimizing code
*   Improving test cases

## Community & Support

*   **Discussions:** [Discussions](https://github.com/hhyo/Archery/discussions) for usage questions and feature requests.
*   **Bug Reports:** [Issues](https://github.com/hhyo/archery/issues) for reporting and tracking bugs.

## Acknowledgements

*   [archer](https://github.com/jly8866/archer): Archery is based on archer.
*   [goInception](https://github.com/hanchuanchuan/goInception): A MySQL operation tool.
*   [JetBrains Open Source](https://www.jetbrains.com/zh-cn/opensource/?from=archery) provides free IDE licenses.  
    [<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.png" width="200"/>](https://www.jetbrains.com/opensource/)
```
Key improvements and explanations:

*   **SEO Optimization:**  Keywords like "SQL Audit", "Query Platform", "Database Management" are included in the title and description.
*   **Clear Structure:**  Uses clear headings (H2) and bullet points for readability and scannability. This is good for both humans and search engines.
*   **Concise Summary:** The one-sentence hook clearly describes the project's primary function.
*   **Focus on Benefits:**  Highlights the benefits of using Archery (streamlining management, improving security) to attract potential users.
*   **Call to Action:** Includes a clear "Explore the original repository" link at the top.
*   **Complete Information:** The original content is preserved and organized.
*   **Maintainability:**  The structure is easy to update as the project evolves.
*   **Relevant Links:** All important links (documentation, demo, etc.) are easily accessible.
*   **Clear Login Credentials:**  Login details for the demo are clearly displayed.
*   **Community Focus:** Encourages contributions and provides guidance on how to do so.

This revised README is significantly more user-friendly and is also more likely to be found by those searching for database management and SQL auditing tools.