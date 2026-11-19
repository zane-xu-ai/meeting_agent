"""Idempotent knowledge reconciliation from durable completed analyses."""
import argparse
import time
from loguru import logger
from persistence.repository import repository
from knowledge.store import backfill

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    repo=repository()
    if not repo: raise SystemExit('DATABASE_URL is required')
    while True:
        try:
            logger.info('知识构建完成 {} 条',backfill(repo))
            from knowledge.retrieval import index
            index(repo)
        except Exception as error:
            if args.once: raise
            logger.error('知识构建暂不可用 {}',type(error).__name__)
        if args.once: break
        time.sleep(10)
