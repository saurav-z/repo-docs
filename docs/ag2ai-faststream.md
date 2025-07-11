# FastStream: Effortlessly Integrate Event Streams for Microservices

Simplify event stream integration with **FastStream**, a powerful Python framework designed to make building and scaling microservices easy and efficient. [Explore the FastStream repository](https://github.com/ag2ai/faststream/) for more details.

---

[![Trendshift](https://trendshift.io/api/badge/repositories/3043)](https://trendshift.io/repositories/3043)
[![Tests](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main)](https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml)
[![Coverage](https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg)](https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream)
[![Downloads](https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month)](https://www.pepy.tech/projects/faststream)
[![PyPI](https://img.shields.io/pypi/v/faststream?label=PyPI)](https://pypi.org/project/faststream)
[![Python Versions](https://img.shields.io/pypi/pyversions/faststream.svg)](https://pypi.org/project/faststream)
[![CodeQL](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml)
[![Dependency Review](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg)](https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml)
[![License](https://img.shields.io/github/license/ag2ai/faststream.svg)](https://github.com/ag2ai/faststream/blob/main/LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN)](https://discord.gg/qFm6aSqq59)
[![FastStream](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json)](https://faststream.ag2.ai/latest/)
[![Telegram](https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU)](https://t.me/python_faststream)
[![Gurubase](https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF)](https://gurubase.io/g/faststream)

---

## Key Features

*   **Multiple Broker Support:** Seamlessly work with Kafka, RabbitMQ, NATS, and Redis using a unified API.
*   **Pydantic Validation:**  Leverage Pydantic for efficient data validation and serialization of messages.
*   **Automatic Documentation:** Generate AsyncAPI documentation automatically to streamline service integration.
*   **Type-Safe Development:** Benefit from full-typed editor support, catching errors early and improving developer experience.
*   **Dependency Injection:** Utilize a powerful dependency injection system for managing service dependencies.
*   **Simplified Testing:** Easily test your services with in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensibility:** Use extensions for lifespans, custom serialization, and middleware.
*   **Framework Compatibility:** Fully compatible with any HTTP framework ([FastAPI](#fastapi-plugin) plugin included).

---

## Install

FastStream is compatible with Linux, macOS, Windows, and most Unix-like operating systems. Install it easily using `pip`:

```bash
pip install 'faststream[kafka]'       # For Kafka
pip install 'faststream[rabbit]'      # For RabbitMQ
pip install 'faststream[nats]'        # For NATS
pip install 'faststream[redis]'       # For Redis
```

By default, FastStream uses PydanticV2, but if your platform lacks Rust support, it will work seamlessly with PydanticV1.

---

## Writing App Code

FastStream provides convenient function decorators (`@broker.subscriber` and `@broker.publisher`) to handle message consumption and production. This simplifies your code, letting you focus on your core business logic:

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

FastStream uses Pydantic for data parsing, allowing you to serialize input messages with type annotations:

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

---

## Testing Your Service

Use `TestBroker` context managers for in-memory testing, enabling fast and reliable testing without a running broker:

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

---

## Running the Application

Start your FastStream application using the built-in CLI command:

1.  Install the CLI: `pip install "faststream[cli]"`
2.  Run your app: `faststream run your_module:app`

FastStream also offers hot reload (`--reload`) and multiprocessing (`--workers`) features for enhanced development.

Learn more about CLI features [here](https://faststream.ag2.ai/latest/getting-started/cli/).

---

## Project Documentation

FastStream automatically generates documentation following the AsyncAPI specification.  This streamlines service integration by providing clear documentation of your application's channels and message formats.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream features a dependency management system, similar to `pytest fixtures` and `FastAPI Depends`, using `FastDepends`:

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

---

## HTTP Framework Integrations

### Any Framework

Use FastStream `MQBrokers` with any HTTP framework:

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

### FastAPI Plugin

Integrate FastStream with FastAPI:

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

Find more integration details [here](https://faststream.ag2.ai/latest/getting-started/integrations/fastapi/).

---

## Stay in Touch

Support FastStream and stay updated:

*   Star our [GitHub repository](https://github.com/ag2ai/faststream/).
*   Join our [EN Discord server](https://discord.gg/qFm6aSqq59).
*   Join our [RU Telegram group](https://t.me/python_faststream).

---

## Contributors

Special thanks to all contributors:
<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>