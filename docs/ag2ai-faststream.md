# FastStream: Build Modern Microservices with Effortless Event Stream Integration

**FastStream simplifies event stream integration, enabling you to build robust and scalable microservices with ease.**

---

<p align="center">
  <a href="https://trendshift.io/repositories/3043" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3043" alt="ag2ai%2Ffaststream | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

  <br/>
  <br/>

  <a href="https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml" target="_blank">
    <img src="https://github.com/ag2ai/faststream/actions/workflows/pr_tests.yaml/badge.svg?branch=main" alt="Test Passing"/>
  </a>

  <a href="https://coverage-badge.samuelcolvin.workers.dev/redirect/ag2ai/faststream" target="_blank">
      <img src="https://coverage-badge.samuelcolvin.workers.dev/ag2ai/faststream.svg" alt="Coverage"/>
  </a>

  <a href="https://www.pepy.tech/projects/faststream" target="_blank">
    <img src="https://static.pepy.tech/personalized-badge/faststream?period=month&units=international_system&left_color=grey&right_color=green&left_text=downloads/month" alt="Downloads"/>
  </a>

  <a href="https://pypi.org/project/faststream" target="_blank">
    <img src="https://img.shields.io/pypi/v/faststream?label=PyPI" alt="Package version"/>
  </a>

  <a href="https://pypi.org/project/faststream" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/faststream.svg" alt="Supported Python versions"/>
  </a>

  <br/>

  <a href="https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml" target="_blank">
    <img src="https://github.com/ag2ai/faststream/actions/workflows/pr_codeql.yaml/badge.svg" alt="CodeQL"/>
  </a>

  <a href="https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml" target="_blank">
    <img src="https://github.com/ag2ai/faststream/actions/workflows/pr_dependency-review.yaml/badge.svg" alt="Dependency Review"/>
  </a>

  <a href="https://github.com/ag2ai/faststream/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/ag2ai/faststream.svg" alt="License"/>
  </a>

  <a href="https://github.com/ag2ai/faststream/blob/main/CODE_OF_CONDUCT.md" target="_blank">
    <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Code of Conduct"/>
  </a>

  <br/>

  <a href="https://discord.gg/qFm6aSqq59" target="_blank">
      <img alt="Discord" src="https://img.shields.io/discord/1085457301214855171?logo=discord&label=EN"/>
  </a>

  <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fag2ai%2Ffaststream%2Fmain%2Fdocs%2Fdocs%2Fassets%2Fimg%2Fshield.json" alt="FastStream"/>

  <a href="https://t.me/python_faststream" target="_blank">
    <img alt="Telegram" src="https://img.shields.io/badge/-telegram-black?color=blue&logo=telegram&label=RU"/>
  </a>

  <br/>

  <a href="https://gurubase.io/g/faststream" target="_blank">
    <img src="https://img.shields.io/badge/Gurubase-Ask%20FastStream%20Guru-006BFF" alt="Gurubase"/>
  </a>
</p>

---

## Key Features

*   **Multiple Broker Support:** Seamlessly integrate with popular message brokers like Kafka, RabbitMQ, NATS, and Redis.
*   **Pydantic Validation:** Leverage Pydantic for data validation and serialization, ensuring data integrity.
*   **Automatic AsyncAPI Documentation:** Generate comprehensive documentation to simplify service integration.
*   **Type-Safe Development:** Benefit from full-typed editor support, catching errors early in the development cycle.
*   **Dependency Injection:** Utilize a built-in dependency injection system for efficient service management.
*   **Simplified Testing:** Test your services easily with in-memory tests.
*   **Extensible Architecture:** Use extensions for lifespans, custom serialization and middleware.
*   **Framework Integrations:** Integrate with any HTTP framework, with specific support for FastAPI.

---

**For more details and advanced usage, check out the documentation:** [https://faststream.ag2.ai/latest/](https://faststream.ag2.ai/latest/)

---

## Getting Started

### Installation

Install FastStream with your preferred message broker dependencies using pip:

```bash
pip install 'faststream[kafka]'  # For Kafka
pip install 'faststream[rabbit]' # For RabbitMQ
pip install 'faststream[nats]'   # For NATS
pip install 'faststream[redis]'  # For Redis
```

### Example Usage

Here's a basic example of how to use FastStream:

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

### Testing

Test your services with ease using the `TestKafkaBroker`:

```python
import pytest
import pydantic
from faststream.kafka import TestKafkaBroker

@pytest.mark.asyncio
async def test_correct():
    async with TestKafkaBroker(broker) as br:
        await br.publish({"user": "John", "user_id": 1}, "in")

@pytest.mark.asyncio
async def test_invalid():
    async with TestKafkaBroker(broker) as br:
        with pytest.raises(pydantic.ValidationError):
            await br.publish("wrong message", "in")
```

### Running Your Application

Use the FastStream CLI to run your application:

```bash
pip install "faststream[cli]"
faststream run your_module:app
```

### Hot Reload and Multiprocessing

FastStream offers hot reload and multiprocessing features to improve your development experience:

```bash
faststream run your_module:app --reload  # Hot reload
faststream run your_module:app --workers 3 # Multiprocessing
```

---

## Project Background

FastStream is based on the experience from [FastKafka](https://github.com/airtai/fastkafka) and [Propan](https://github.com/lancetnik/propan), aiming to provide a unified solution for processing streamed data.

---

## Integrations

### Any Framework

You can use **FastStream** `MQBrokers` without a `FastStream` application.
Just *start* and *stop* them according to your application's lifespan.

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

### **FastAPI** Plugin

Also, **FastStream** can be used as part of **FastAPI**.

Just import a **StreamRouter** you need and declare the message handler with the same `@router.subscriber(...)` and `@router.publisher(...)` decorators.

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

## Stay Connected

Show your support and get the latest updates:

*   Star our [GitHub repository](https://github.com/ag2ai/faststream/)
*   Join our [Discord server](https://discord.gg/qFm6aSqq59)
*   Join our [Telegram group](https://t.me/python_faststream)

---

## Contributors

Thanks to all of these amazing people who made the project better!

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>