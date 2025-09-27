# FastStream: Effortlessly Integrate Event Streams into Your Microservices

**Build robust, real-time microservices with ease using FastStream – the Python framework designed for seamless event stream integration.**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
<br/><br/>
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
<br/>
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
<br/>
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
<img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json" alt="FastStream"/>
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
<br/>
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features of FastStream

*   ✅ **Multiple Broker Support:** Works seamlessly with Kafka, RabbitMQ, NATS, and Redis.
*   ✅ **Pydantic Validation:**  Built-in data validation using Pydantic for robust message handling.
*   ✅ **Automatic Documentation:** Generates AsyncAPI documentation automatically, simplifying integration.
*   ✅ **Intuitive Development:**  Benefit from full-typed editor support, catching errors early.
*   ✅ **Dependency Injection:**  Efficiently manage service dependencies with FastStream's built-in system.
*   ✅ **Testability:** Supports in-memory testing, enabling fast and reliable CI/CD pipelines.
*   ✅ **Extensibility:** Leverage extensions for lifespans, custom serialization, and middleware.
*   ✅ **Framework Integrations:** Full compatibility with any HTTP framework, especially [FastAPI](#fastapi-plugin).

## What is FastStream?

FastStream simplifies building event-driven microservices by handling message queue interactions, parsing, and documentation generation automatically.  Designed with developers in mind, FastStream offers ease of use while remaining powerful enough for advanced use cases.  It is your go-to framework for creating scalable, data-centric microservices with minimal effort.

**Ready to get started?** Check out the [FastStream documentation](https://faststream.ag2.ai/latest/) and [source code](https://github.com/ag2ai/faststream).

---

## Core Functionality

### Writing App Code

FastStream provides convenient function decorators (`@broker.subscriber` and `@broker.publisher`) to streamline the process of consuming and producing data to event queues. These decorators enable you to focus on core business logic by simplifying message handling and data serialization/deserialization.

FastStream seamlessly integrates with [Pydantic](https://docs.pydantic.dev/) for input JSON data parsing into Python objects. This facilitates working with structured data within your applications, achieved through type annotations for input messages.

```python
from faststream import FastStream
from faststream.kafka import KafkaBroker

broker = KafkaBroker("localhost:9092")
app = FastStream(broker)

@broker.subscriber("in")
@broker.publisher("out")
async def handle_msg(user: str, user_id: int) -> str:
    return f"User: {user_id} - {user} registered"
```

### Testing the Service

The `TestBroker` context managers simplify service testing.  It redirects `subscriber` and `publisher` functions to in-memory brokers, making testing fast and independent of a running broker.

### Running the Application

The application can be started and managed via the FastStream CLI. Install it via `pip install "faststream[cli]"`.

To run the service, use the CLI and specify the module and application symbol:
```shell
faststream run basic:app
```

Features include hot reload (`--reload`) and multiprocessing (`--workers 3`). More about the CLI [here](https://faststream.ag2.ai/latest/getting-started/cli/).

### Project Documentation

FastStream automatically generates documentation according to the [AsyncAPI](https://www.asyncapi.com/) specification.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

### Dependencies

FastStream has a dependency management system, similar to `pytest fixtures` and `FastAPI Depends`.

```python
from typing import Annotated
from faststream import Depends, Logger

async def base_dep(user_id: int) -> bool:
    return True

@broker.subscriber("in-test")
async def base_handler(user: str,
                       logger: Logger,
                       dep: Annotated[bool, Depends(base_dep)]):
    assert dep is True
    logger.info(user)
```

### HTTP Frameworks Integrations

#### Any Framework

FastStream `MQBrokers` can be used independently of a `FastStream` application:

```python
from aiohttp import web

from faststream.kafka import KafkaBroker

broker = KafkaBroker("localhost:9092")

@broker.subscriber("test")
async def base_handler(body):
    print(body)

async def start_broker(app):
    await broker.start()

async def stop_broker(app):
    await broker.stop()

async def hello(request):
    return web.Response(text="Hello, world")

app = web.Application()
app.add_routes([web.get("/", hello)])
app.on_startup.append(start_broker)
app.on_cleanup.append(stop_broker)

if __name__ == "__main__":
    web.run_app(app)
```

#### FastAPI Plugin

```python
from fastapi import FastAPI
from pydantic import BaseModel

from faststream.kafka.fastapi import KafkaRouter

router = KafkaRouter("localhost:9092")

class Incoming(BaseModel):
    m: dict

@router.subscriber("test")
@router.publisher("response")
async def hello(m: Incoming):
    return {"response": "Hello, world!"}

app = FastAPI()
app.include_router(router)
```
More integration features can be found [here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/)

---

## Installation

FastStream is compatible with **Linux**, **macOS**, **Windows**, and most Unix-style operating systems.

```sh
pip install 'faststream[kafka]'
# or
pip install 'faststream[rabbit]'
# or
pip install 'faststream[nats]'
# or
pip install 'faststream[redis]'
```

FastStream leverages **PydanticV2**, but it can work with **PydanticV1** on platforms without Rust support.

---

## Stay Connected

*   ⭐ Give the [GitHub repository](https://github.com/ag2ai/faststream/) a star.
*   💬 Join our [EN Discord server](https://discord.gg/qFm6aSqq59).
*   💬 Join our [RU Telegram group](https://t.me/python_faststream).

---

## Contributors

Thank you to all the amazing contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>