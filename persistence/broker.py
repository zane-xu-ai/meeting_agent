"""RabbitMQ publisher with confirms and bounded, manually acknowledged consumers."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from contextlib import suppress
from pathlib import Path

import aio_pika
from loguru import logger

from config import settings
from persistence import dispatch
from persistence.repository import repository


ROUTES = {"io": 2, "cpu": 2, "cloud": 2, "gpu": 1}


async def topology(connection, prefetch=1):
    channel = await connection.channel(publisher_confirms=True, on_return_raises=True)
    await channel.set_qos(prefetch_count=prefetch)
    exchange = await channel.declare_exchange("meeting.tasks", durable=True)
    dlx = await channel.declare_exchange("meeting.dlx", durable=True)
    dead = await channel.declare_queue("meeting.dead", durable=True)
    await dead.bind(dlx, "dead")
    await dead.bind(exchange, "dead")
    queues = {}
    for route in ROUTES:
        queue = await channel.declare_queue(
            f"meeting.{route}", durable=True,
            arguments={"x-dead-letter-exchange": "meeting.dlx", "x-dead-letter-routing-key": "dead"},
        )
        await queue.bind(exchange, route)
        queues[route] = queue
    return exchange, queues


async def publish_pending(repo, exchange):
    for row in await asyncio.to_thread(dispatch.reserve_publications, repo):
        body = json.dumps({
            "dispatch_id": str(row["id"]), "job_id": row["job_id"], "schema_version": 1,
        }).encode()
        await exchange.publish(aio_pika.Message(
            body, message_id=str(row["id"]), content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ), routing_key=row["routing_key"], mandatory=True, timeout=10)
        await asyncio.to_thread(dispatch.published, repo, row["id"])


async def dispatch_loop(repo, exchange):
    while True:
        try:
            await publish_pending(repo, exchange)
        except Exception as error:
            logger.warning("派发暂不可用: {}", type(error).__name__)
        await asyncio.sleep(.5)


def initial_state(work):
    job = work["job"]
    task_id = job["id"]
    state = job["state"]
    return {
        "source": job["input_location"], "source_type": state["source_type"], "task_id": task_id,
        "asr_model": job["request_config"].get("asr_model"), "llm_model": job["request_config"].get("llm_model"),
        "audio_stem": task_id, "transcript": None, "transcript_text": "", "asr_duration": 0.0,
        "rec_type": "general", "output_text": "", "pipeline_start": time.time(),
    }


def execute_stage(repo, work):
    from agent.graph import pipeline_nodes, register_progress_cb, register_result_handler, unregister_task

    identifier = work["job_id"]
    state = dispatch.decode(work["checkpoint"]) if work["checkpoint"] else initial_state(work)
    state["work_path"] = str(settings.output_dir / "work" / identifier / work["lease_token"].hex)
    if dispatch.STEPS[work["step"]][0] == "agent_loop":
        state["audio_stem"] = f"{identifier}_{work['lease_token'].hex[:12]}"

    def progress(**values):
        dispatch.progress(repo, work, values)

    def save_report(task_id, stem, subdir, content, asr_model=None, llm_model=None):
        path = Path(state["work_path"]) / "report.md"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        progress(result_path=str(path))

    register_progress_cb(identifier, progress)
    register_result_handler(identifier, save_report)
    try:
        function = pipeline_nodes()[dispatch.STEPS[work["step"]][0]]
        state.update(asyncio.run(function(state)))
        dispatch.complete(repo, work, state)
    finally:
        unregister_task(identifier)


async def handle(repo, message, route=None):
    try:
        payload = json.loads(message.body)
        if payload["schema_version"] != 1:
            raise ValueError("version")
        identifier = uuid.UUID(payload["dispatch_id"])
    except (KeyError, TypeError, ValueError):
        await message.reject(requeue=False)
        return
    try:
        work = await asyncio.to_thread(dispatch.claim, repo, identifier, route)
        if work:
            renewal = asyncio.create_task(renew(repo, work))
            try:
                await asyncio.to_thread(execute_stage, repo, work)
            except Exception as error:
                logger.warning("任务 {} 阶段失败: {}", work["job_id"], type(error).__name__)
                await asyncio.to_thread(dispatch.failed, repo, work, error)
            finally:
                renewal.cancel()
                with suppress(asyncio.CancelledError):
                    await renewal
        await message.ack()
    except Exception as error:
        logger.warning("消费通道中断: {}", type(error).__name__)
        # SQL recovery will republish any uncommitted work after its reservation/lease expires.
        if not message.processed:
            with suppress(Exception):
                await message.ack()


async def renew(repo, work):
    while True:
        await asyncio.sleep(20)
        try:
            if not await asyncio.to_thread(dispatch.heartbeat, repo, work):
                return
        except Exception as error:
            logger.warning("阶段租约续期失败: {}", type(error).__name__)
            return


async def run(mode):
    repo = repository()
    url = settings.amqp_url if mode == "dispatcher" else settings.amqp_worker_url or settings.amqp_url
    if not repo or not url:
        raise SystemExit("DATABASE_URL 与 AMQP_URL 必须设置")
    async with await aio_pika.connect_robust(url, fail_fast=False) as connection:
        exchange, queues = await topology(connection, ROUTES.get(mode, 1))
        if mode == "dispatcher":
            await dispatch_loop(repo, exchange)
        else:
            await queues[mode].consume(lambda message: handle(repo, message, mode), no_ack=False)
            await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["dispatcher", *ROUTES])
    asyncio.run(run(parser.parse_args().mode))
