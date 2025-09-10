<div align="center">

# Archery: Your Centralized SQL Audit and Query Platform

Archery is a powerful and versatile SQL audit and query platform designed to streamline database management. [View the original repository on GitHub](https://github.com/hhyo/Archery).

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

*   **Comprehensive Database Support:**
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
    *   SQL Query Execution
    *   SQL Audit and Review
    *   Data Backup and Restore
    *   Data Dictionary
    *   Slow Query Log Analysis
    *   Session Management
    *   User and Account Management
    *   Parameter Management
    *   Data Archiving

### Feature Matrix

| Database    | Query | Audit | Execute | Backup | Data Dictionary | Slow Log | Session Management | Account Management | Parameter Management | Data Archiving |
| ----------- | ----- | ----- | ------- | ------ | --------------- | -------- | ------------------ | ------------------ | ------------------ | -------------- |
| MySQL       | √     | √     | √       | √      | √               | √        | √                  | √                  | √                  | √              |
| MsSQL       | √     | ×     | √       | ×      | √               | ×        | ×                  | ×                  | ×                  | ×              |
| Redis       | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×                  | ×                  | ×              |
| PgSQL       | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×                  | ×                  | ×              |
| Oracle      | √     | √     | √       | √      | √               | ×        | √                  | ×                  | ×                  | ×              |
| MongoDB     | √     | √     | √       | ×      | ×               | ×        | √                  | √                  | ×                  | ×              |
| Phoenix     | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×                  | ×                  | ×              |
| ODPS        | √     | ×     | ×       | ×      | ×               | ×        | ×                  | ×                  | ×                  | ×              |
| ClickHouse  | √     | √     | √       | ×      | ×               | ×        | ×                  | ×                  | ×                  | ×              |
| Cassandra   | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×                  | ×                  | ×              |
| Doris       | √     | ×     | √       | ×      | ×               | ×        | ×                  | ×                  | ×                  | ×              |

## Quick Start

### System Demo

[Online Demo](https://demo.archerydms.com)

| Account  | Password |
| -------- | -------- |
| archer   | archer   |

### Docker

See the [Docker documentation](https://github.com/hhyo/archery/wiki/docker).

## Manual Installation

[Deployment Instructions](https://github.com/hhyo/archery/wiki/manual)

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
*   Tables: [bootstrap-table](https://github.com/wenzhixin/bootstrap-table)
*   Table Editing: [bootstrap-editable](https://github.com/vitalets/x-editable)
*   Dropdown: [bootstrap-select](https://github.com/snapappointments/bootstrap-select)
*   File Upload: [bootstrap-fileinput](https://github.com/kartik-v/bootstrap-fileinput)
*   Date/Time Picker: [bootstrap-datetimepicker](https://github.com/smalot/bootstrap-datetimepicker)
*   Date Range Picker: [daterangepicker](https://github.com/dangrossman/daterangepicker)
*   Switches: [bootstrap-switch](https://github.com/Bttstrp/bootstrap-switch)
*   Markdown Display: [marked](https://github.com/markedjs/marked)

### Backend Dependencies

*   Queue Tasks: [django-q](https://github.com/Koed00/django-q)
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
*   Date/Time Handling: [python-dateutil](https://github.com/paxan/python-dateutil)

### Functionality Dependencies

*   Visualization: [pyecharts](https://github.com/pyecharts/pyecharts)
*   MySQL Audit/Execution: [goInception](https://github.com/hanchuanchuan/goInception) | [inception](https://github.com/hhyo/inception)
*   MySQL Index Optimization: [SQLAdvisor](https://github.com/Meituan-Dianping/SQLAdvisor)
*   SQL Optimization: [SOAR](https://github.com/XiaoMi/soar)
*   My2SQL: [my2sql](https://github.com/liuhr/my2sql)
*   Table Schema Sync: [SchemaSync](https://github.com/hhyo/SchemaSync)
*   Slow Log Analysis: [pt-query-digest](https://www.percona.com/doc/percona-toolkit/3.0/pt-query-digest.html) | [aquila\_v2](https://github.com/thinkdb/aquila_v2)
*   Large Table DDL: [gh-ost](https://github.com/github/gh-ost) | [pt-online-schema-change](https://www.percona.com/doc/percona-toolkit/3.0/pt-online-schema-change.html)
*   MyBatis XML Parsing: [mybatis-mapper2sql](https://github.com/hhyo/mybatis-mapper2sql)
*   RDS Management: [aliyun-openapi-python-sdk](https://github.com/aliyun/aliyun-openapi-python-sdk)
*   Data Encryption: [django-mirage-field](https://github.com/luojilab/django-mirage-field)

## Contributing

Review the project's development plan and dependency list, claim issues, submit pull requests, or contribute in other ways.

Ways to contribute:

*   [Wiki Documentation](https://github.com/hhyo/Archery/wiki) (Open for editing)
*   Bug Fixes
*   New Feature Submissions
*   Code Optimization
*   Test Case Improvements

## Feedback and Discussions

*   Usage questions and feature requests: [Discussions](https://github.com/hhyo/Archery/discussions)
*   Bug Reports: [Issues](https://github.com/hhyo/archery/issues)

## Acknowledgements

*   [archer](https://github.com/jly8866/archer): Archery is built based on the second development of archer.
*   [goInception](https://github.com/hanchuanchuan/goInception): MySQL operation and maintenance tool that integrates audit, execution, backup, and rollback statement generation.
*   [JetBrains Open Source](https://www.jetbrains.com/zh-cn/opensource/?from=archery) provides free IDE licenses for the project.

  [<img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.png" width="200"/>](https://www.jetbrains.com/opensource/)