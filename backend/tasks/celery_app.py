"""
Celery 应用实例。
Broker / Backend 均使用 Redis。
"""

from celery import Celery

REDIS_URL = "redis://localhost:6379/0"

celery_app = Celery(
    "nocode_platform",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks.train_task"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # 任务超时：单次训练最长 24 小时
    task_soft_time_limit=86400,
    task_time_limit=90000,
    # 结果过期时间：7 天
    result_expires=604800,
    # Worker 并发：每个 GPU 任务独占一个 Worker
    worker_concurrency=1,
)