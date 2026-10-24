import asyncio
import logging

from dataclasses import dataclass

from fastapi import HTTPException

import config

from model import model_predict_batch
from schemas import FarmRequest


logger = logging.getLogger(__name__)


@dataclass
class InferenceTask:
    data: FarmRequest
    request_id: str
    future: asyncio.Future


class DynamicBatcher:
    def __init__(self):
        self.queue = asyncio.Queue(
            maxsize=config.QUEUE_MAX_SIZE,
        )
        self.worker_task = None

    async def start(self):
        self.worker_task = asyncio.create_task(self._worker())
        logger.info("Dynamic batch worker started")

    async def stop(self):
        if self.worker_task is not None:
            self.worker_task.cancel()

            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

        while not self.queue.empty():
            task = self.queue.get_nowait()

            if not task.future.done():
                task.future.cancel()

            self.queue.task_done()

        logger.info("Dynamic batch worker stopped")

    async def submit(
        self,
        data: FarmRequest,
        request_id: str,
    ) -> float:
        if self.worker_task is None or self.worker_task.done():
            raise HTTPException(
                status_code=503,
                detail="Inference worker is unavailable",
            )

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        task = InferenceTask(
            data=data,
            request_id=request_id,
            future=future,
        )

        try:
            self.queue.put_nowait(task)

        except asyncio.QueueFull as exc:
            raise HTTPException(
                status_code=429,
                detail="Inference queue is full. Try again later.",
                headers={"Retry-After": "1"},
            ) from exc

        logger.info(
            "Request queued | request_id=%s | queue_size=%s",
            request_id,
            self.queue.qsize(),
        )

        try:
            return await asyncio.wait_for(
                future,
                timeout=config.INFERENCE_TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail="Queued inference timeout",
            ) from exc

        finally:
            if not future.done():
                future.cancel()

    async def _worker(self):
        loop = asyncio.get_running_loop()

        while True:
            first = await self.queue.get()
            batch = [first]

            try:
                if first.future.done():
                    continue

                deadline = (
                    loop.time()
                    + config.MAX_BATCH_WAIT_MS / 1000
                )

                while len(batch) < config.MAX_BATCH_SIZE:
                    remaining = deadline - loop.time()

                    if remaining <= 0:
                        break

                    try:
                        task = await asyncio.wait_for(
                            self.queue.get(),
                            timeout=remaining,
                        )

                    except asyncio.TimeoutError:
                        break

                    batch.append(task)

                active = [
                    task
                    for task in batch
                    if not task.future.done()
                ]

                if not active:
                    continue

                logger.info(
                    "Dynamic batch formed | size=%s | "
                    "queue_size=%s | request_ids=%s",
                    len(active),
                    self.queue.qsize(),
                    [task.request_id for task in active],
                )

                scores = await asyncio.wait_for(
                    model_predict_batch(
                        [task.data for task in active]
                    ),
                    timeout=config.INFERENCE_TIMEOUT_SECONDS,
                )

                if len(scores) != len(active):
                    raise RuntimeError(
                        "Unexpected number of batch scores"
                    )

                for task, score in zip(active, scores):
                    if not task.future.done():
                        task.future.set_result(score)

            except asyncio.CancelledError:
                for task in batch:
                    if not task.future.done():
                        task.future.cancel()

                raise

            except asyncio.TimeoutError:
                logger.warning("Dynamic batch inference timed out")

                for task in batch:
                    if not task.future.done():
                        task.future.set_exception(
                            HTTPException(
                                status_code=504,
                                detail="Batch inference timeout",
                            )
                        )

            except Exception:
                logger.exception("Dynamic batch inference failed")

                for task in batch:
                    if not task.future.done():
                        task.future.set_exception(
                            HTTPException(
                                status_code=503,
                                detail="Inference model is unavailable",
                            )
                        )

            finally:
                for _ in batch:
                    self.queue.task_done()