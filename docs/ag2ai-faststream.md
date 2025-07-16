# FastStream: Effortlessly Integrate Event Streams for Microservices

**Simplify your microservices architecture with FastStream, a Python framework designed for seamless event stream integration.** [Check out the FastStream GitHub repository](https://github.com/ag2ai/faststream/) for the latest updates and contributions.

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Test Passing](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

*   **Multiple Broker Support:** Seamlessly integrate with Kafka, RabbitMQ, NATS, and Redis using a unified API.
*   **Pydantic Validation:** Utilize Pydantic for robust data validation and serialization of incoming messages.
*   **Automatic Documentation:** Generate AsyncAPI documentation automatically for easy service integration.
*   **Intuitive Development:** Benefit from full-typed editor support, catching errors early in development.
*   **Dependency Injection:** Manage service dependencies efficiently with FastStream's built-in DI system.
*   **Simplified Testing:** Test your services easily with in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensibility:** Customize lifespans, serialization, and middleware through extensions.
*   **Framework Compatibility:** Integrate with any HTTP framework, including a dedicated FastAPI plugin.

---

## Getting Started

### Installation

Install FastStream using pip, with support for your desired broker:

```bash
pip install 'faststream[kafka]'  # for Kafka
# or
pip install 'faststream[rabbit]' # for RabbitMQ
# or
pip install 'faststream[nats]'   # for NATS
# or
pip install 'faststream[redis]'  # for Redis
```

### Writing Application Code

FastStream provides decorators (`@broker.subscriber` and `@broker.publisher`) to simplify message queue interactions. Leverage Pydantic for data parsing and validation.

```python
from faststream import FastStream
from faststream.kafka import KafkaBroker
# from faststream.rabbit import RabbitBroker

broker = KafkaBroker("localhost:9092")
# broker = RabbitBroker("amqp://guest:guest@localhost:5672/")

app = FastStream(broker)

@broker.subscriber("in")
@broker.publisher("out")
async def handle_msg(user: str, user_id: int) -> str:
    return f"User: {user_id} - {user} registered"
```

Use Pydantic's `BaseModel` for structured message definitions:

```python
from pydantic import BaseModel, Field, PositiveInt
from faststream import FastStream
from faststream.kafka import KafkaBroker

broker = KafkaBroker("localhost:9092")
app = FastStream(broker)

class User(BaseModel):
    user: str = Field(..., examples=["John"])
    user_id: PositiveInt = Field(..., examples=["1"])

@broker.subscriber("in")
@broker.publisher("out")
async def handle_msg(data: User) -> str:
    return f"User: {data.user} - {data.user_id} registered"
```

### Testing

Use `TestBroker` for in-memory testing:

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

@pytest.mark.asyncio
async def test_invalid():
    async with TestKafkaBroker(broker) as br:
        with pytest.raises(pydantic.ValidationError):
            await br.publish("wrong message", "in")
```

### Running the Application

Install the FastStream CLI:

```bash
pip install "faststream[cli]"
```

Run your application using the CLI:

```bash
faststream run basic:app
```

Use the `--reload` flag for hot reloading and `--workers` for multiprocessing scaling.

---

## Project Documentation

Access automatically generated AsyncAPI documentation to easily understand service integrations.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Advanced Features

### Dependency Injection

Utilize FastStream's dependency management system, similar to `pytest fixtures` and `FastAPI Depends`:

```python
from faststream import Depends, Logger

async def base_dep(user_id: int) -> bool:
    return True

@broker.subscriber("in-test")
async def base_handler(user: str,
                       logger: Logger,
                       dep: bool = Depends(base_dep)):
    assert dep is True
    logger.info(user)
```

### HTTP Framework Integrations

*   **Any Framework:** Integrate `MQBrokers` by starting and stopping them with your application's lifecycle.
*   **FastAPI Plugin:** Use the `KafkaRouter` to seamlessly integrate with FastAPI.

```python
# Example: FastAPI integration
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

---

## Learn More

*   **Documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)
*   **CLI Documentation:** [https://faststream.ag2.ai/latest/getting-started/cli/](https://faststream.ag2.ai/latest/getting-started/cli/)
*   **FastAPI Integration:** [https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/)

---

## Stay in Touch

*   **GitHub:** [https://github.com/ag2ai/faststream/](https://github.com/ag2ai/faststream/)
*   **Discord:** [https://discord.gg/qFm6aSqq59](https://discord.gg/qFm6aSqq59)
*   **Telegram:** [https://t.me/python_faststream](https://t.me/python_faststream)

---

## Contributors

Thank you to all the contributors!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>