# FastStream: Effortlessly Integrate Event Streams for Microservices

**Simplify your microservice architecture with FastStream, a powerful framework for building robust and scalable event-driven applications. Access the original repo [here](https://github.com/ag2ai/faststream).**

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Python Versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram RU](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features of FastStream

*   **Multi-Broker Support:** Integrate seamlessly with popular message brokers, including [Kafka](https://kafka.apache.org/), [RabbitMQ](https://www.rabbitmq.com/), [NATS](https://nats.io/), and [Redis](https://redis.io/).
*   **Pydantic Validation:** Leverage Pydantic for efficient data validation and serialization, ensuring data integrity in your event streams.
*   **Automatic Documentation:** Generate [AsyncAPI](https://www.asyncapi.com/) documentation automatically, making your services self-documenting and easier to integrate.
*   **Intuitive Developer Experience:** Benefit from full-typed editor support, catching errors early and streamlining your development workflow.
*   **Dependency Injection:** Use FastStream's built-in dependency injection system to manage and inject dependencies in your services.
*   **Simplified Testing:** Utilize in-memory tests for quick and reliable testing, accelerating your CI/CD pipelines.
*   **Extensible Architecture:** Customize your streaming applications using extensions for lifespans, custom serialization and middleware.
*   **Flexible Integrations:**  Compatible with any HTTP framework and offers dedicated plugins for seamless integration with frameworks like [FastAPI](https://fastapi.tiangolo.com/).

---

## Getting Started

### Installation

Install FastStream with your desired broker dependencies using `pip`:

```bash
pip install 'faststream[kafka]'   # For Kafka
pip install 'faststream[rabbit]'  # For RabbitMQ
pip install 'faststream[nats]'    # For NATS
pip install 'faststream[redis]'   # For Redis
```

### Basic Usage

FastStream simplifies event stream integration using function decorators like `@broker.subscriber` and `@broker.publisher`. These decorators handle parsing, networking, and documentation generation, allowing you to focus on your core business logic.

**Example:**

```python
from faststream import FastStream
from faststream.kafka import KafkaBroker

broker = KafkaBroker("localhost:9092")  # Replace with your broker's connection string
app = FastStream(broker)

@broker.subscriber("in")
@broker.publisher("out")
async def handle_msg(user: str, user_id: int) -> str:
    return f"User: {user_id} - {user} registered"
```

### Data Modeling with Pydantic

FastStream integrates with Pydantic, allowing you to define message structures using a declarative syntax:

```python
from pydantic import BaseModel
from faststream import FastStream
from faststream.kafka import KafkaBroker

broker = KafkaBroker("localhost:9092")
app = FastStream(broker)

class User(BaseModel):
    user: str
    user_id: int

@broker.subscriber("in")
@broker.publisher("out")
async def handle_msg(data: User) -> str:
    return f"User: {data.user} - {data.user_id} registered"
```

### Testing Your Service

Test your service using `TestKafkaBroker` context managers.

```python
import pytest
import pydantic
from faststream.kafka import TestKafkaBroker

@pytest.mark.asyncio
async def test_correct():
    async with TestKafkaBroker(broker) as br:
        await br.publish({
            "user": "John",
            "user_id": 1,
        }, "in")
```

### Running the Application

Use the FastStream CLI to run your application:

1.  Install the CLI:  `pip install "faststream[cli]"`
2.  Run your app:  `faststream run <your_module>:<your_app>` (e.g., `faststream run basic:app`)

FastStream also supports hot reloading (`--reload`) and multiprocessing (`--workers`) for enhanced development and performance.

---

## Project Documentation

FastStream automatically generates documentation for your project in accordance with the [**AsyncAPI**](https://www.asyncapi.com/) specification.

You can see the generated docs, facilitating the integration of services by clearly presenting channels and message formats.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Advanced Features

### Dependency Injection

FastStream utilizes `FastDepends` for dependency management, supporting a system similar to `pytest fixtures` and `FastAPI Depends`.

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

### HTTP Framework Integrations

*   **Any Framework:** Use FastStream `MQBrokers` in conjunction with other frameworks. Start and stop them during the application's lifespan.

*   **FastAPI Plugin:** FastStream integrates with FastAPI via `KafkaRouter` and allows you to declare message handlers using `@router.subscriber(...)` and `@router.publisher(...)` decorators.

---

## Stay Connected

*   [GitHub Repository](https://github.com/ag2ai/faststream/)
*   [EN Discord server](https://discord.gg/qFm6aSqq59)
*   [RU Telegram group](https://t.me/python_faststream)

---

## Contributors

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>