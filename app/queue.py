"""Job queue behind a small interface. The queue is only a trigger; the DB is the truth."""
from __future__ import annotations

import queue
from abc import ABC, abstractmethod


class QueueFull(Exception):
    """Backpressure signal: the caller should reject the upload with 429."""


class Queue(ABC):
    @abstractmethod
    def publish(self, job_id: str) -> None:
        """Enqueue a job id. Raises QueueFull when the broker will not accept more."""

    @abstractmethod
    def consume(self, timeout: float) -> str | None:
        """Lease the next job id, or None if nothing arrived within timeout seconds."""

    @abstractmethod
    def ack(self, job_id: str) -> None:
        """The message is fully handled and must not be redelivered."""

    @abstractmethod
    def nack(self, job_id: str) -> None:
        """Hand the message back so it is delivered again."""

    @abstractmethod
    def size(self) -> int:
        """Approximate number of messages waiting."""


class InMemoryQueue(Queue):
    """Bounded queue.Queue. Messages are only ids, so nack simply re-puts the id."""

    def __init__(self, maxsize: int) -> None:
        self._q: queue.Queue[str] = queue.Queue(maxsize=maxsize)

    def publish(self, job_id: str) -> None:
        try:
            self._q.put_nowait(job_id)
        except queue.Full as exc:
            raise QueueFull(f"queue holds at most {self._q.maxsize} jobs") from exc

    def consume(self, timeout: float) -> str | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def ack(self, job_id: str) -> None:
        self._q.task_done()

    def nack(self, job_id: str) -> None:
        self._q.task_done()
        # Best effort: if the queue is full the sweeper re-publishes the job from the DB.
        try:
            self._q.put_nowait(job_id)
        except queue.Full:
            pass

    def size(self) -> int:
        return self._q.qsize()


class PubSubQueue(Queue):
    """Google Pub/Sub mapping. Not implemented; documents the production swap.

    publish(job_id)  -> publisher.publish(topic, job_id.encode()). The message body is
                        only the id; the job row in the DB carries everything else.
    consume(timeout) -> subscriber.pull(subscription, max_messages=1) on a pull
                        subscription. Returns the id and remembers its ack_id. The
                        subscription's ack deadline is the lease: if the worker dies the
                        message is redelivered after the deadline. Long jobs call
                        modify_ack_deadline between chunks, in the same place as the
                        DB heartbeat.
    ack(job_id)      -> subscriber.acknowledge(ack_id).
    nack(job_id)     -> modify_ack_deadline(ack_id, 0), which makes Pub/Sub redeliver;
                        the subscription's retry policy supplies exponential backoff.
    size()           -> the num_undelivered_messages metric (or 0 if not needed).

    Pub/Sub is at-least-once, which is why workers claim jobs with one atomic UPDATE:
    a duplicate delivery finds the row already claimed and is dropped. A dead-letter
    topic on the subscription catches messages past max_delivery_attempts, while the
    job row keeps the 'failed' status so it stays inspectable.
    """

    def __init__(self, project: str, topic: str, subscription: str) -> None:
        raise NotImplementedError("PubSubQueue is a design stub; see the class docstring")

    def publish(self, job_id: str) -> None:
        raise NotImplementedError

    def consume(self, timeout: float) -> str | None:
        raise NotImplementedError

    def ack(self, job_id: str) -> None:
        raise NotImplementedError

    def nack(self, job_id: str) -> None:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError
