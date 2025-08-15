# FastStream: Effortlessly Integrate Event Streams in Your Microservices 🚀

**FastStream** simplifies event stream integration, making it easy to build robust and scalable microservices. This README provides a comprehensive overview of **FastStream**'s features and benefits, as well as how to get started. Check out the [FastStream GitHub Repository](https://github.com/ag2ai/faststream/) for more details and to contribute!

---

## Key Features

*   **Multiple Broker Support:** Seamlessly work with various message brokers, including Kafka, RabbitMQ, NATS, and Redis, all through a unified API.
*   **Pydantic Validation:** Leverage Pydantic for robust data validation, ensuring data integrity and simplifying message parsing.
*   **Automatic Documentation:** Generate AsyncAPI documentation automatically, simplifying service integration and improving collaboration.
*   **Intuitive Development:** Enjoy full-typed editor support, catching errors early and streamlining your development workflow.
*   **Powerful Dependency Injection:** Manage service dependencies efficiently with FastStream's built-in dependency injection system.
*   **Testability:** Utilize in-memory tests for faster and more reliable CI/CD pipelines.
*   **Extensibility:** Customize your experience with extensions for lifespans, custom serialization, and middleware.
*   **Framework Integrations:** Easily integrate FastStream with any HTTP framework, including a dedicated FastAPI plugin.

---

## What is FastStream?

FastStream is designed to streamline the development of microservices that interact with event streams. It handles the complexities of message queue interaction, including parsing, networking, and documentation generation, allowing developers to focus on business logic. Built with junior developers in mind, FastStream simplifies your work while still accommodating advanced use cases.

---

## Getting Started

### Installation

Install FastStream using pip:

```bash
pip install 'faststream[kafka]'  # For Kafka
pip install 'faststream[rabbit]' # For RabbitMQ
pip install 'faststream[nats]'   # For NATS
pip install 'faststream[redis]'  # For Redis
```

FastStream uses PydanticV2 written in Rust by default, but will work with PydanticV1 if Rust support is unavailable.

### Writing App Code

Use function decorators to define message producers and consumers:

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

Leverage Pydantic models for structured data:

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

Use `TestBroker` context managers for in-memory testing:

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

Install the CLI:

```bash
pip install "faststream[cli]"
```

Run your application:

```bash
faststream run basic:app
```

Use the `--reload` flag for hot reloading, and `--workers` for multiprocessing.

---

## Project Documentation

FastStream generates [AsyncAPI](https://www.asyncapi.com/) documentation automatically, making it easy to understand how to integrate with your services.

![HTML-page](https://github.com/ag2ai/faststream/blob/main/docs/docs/assets/img/AsyncAPI-basic-html-short.png?raw=true)

---

## Dependencies

FastStream uses [FastDepends](https://lancetnik.github.io/FastDepends/) for dependency management.

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

---

## Framework Integrations

### Any Framework

You can use FastStream MQBrokers with any framework, just start and stop them with your application's lifespan.

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

---

## Stay in Touch

*   ⭐ Give us a star on [GitHub](https://github.com/ag2ai/faststream/)!
*   Join our [Discord server](https://discord.gg/qFm6aSqq59)!
*   Join our [Telegram group](https://t.me/python_faststream)!

---

## Contributors

Special thanks to the following people who have contributed to the project:

<a href="https://github.com/ag2ai/faststream/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ag2ai/faststream"/>
</a>