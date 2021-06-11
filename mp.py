import logging
import sys
import time
import shutil
import os.path
import tempfile

from multiprocessing import Process, SimpleQueue
from pathlib import Path

import pandas as pd


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def buffer(queue, done_queue, path):
    logger.info('[buffer] worker start.')

    while done_queue.empty():
        matched_name = None
        for item in sorted(Path(path).iterdir(), key=os.path.getmtime):
            if str(item.name).find('.lock') == -1:
                matched_name = str(item.name)
                break

        logger.info('[buffer] matched_name = {}'.format(matched_name))
        if matched_name is not None:       # single item queue
            raw_data = pd.read_csv(path + '/' + matched_name, sep='\t')
            queue.put(raw_data)

        time.sleep(0.5)

    logger.info('[buffer] worker end.')


def downloader(file_list, path, max_size=1048576):
    logger.info('[downloader] worker start.')

    for file in file_list:
        # download
        logger.info('[downloader] target file = {}'.format(file))
        while True:
            used_size = sum(os.path.getsize(f) for f in os.listdir(path) if os.path.isfile(f))
            logger.info('[downloader] disk used = {}'.format(used_size))

            # Note that the capacity of disk usage can larger than max_size.
            if used_size < max_size:
                # Downloading file from mounted path to temp path
                target_dir = './tcga/' + file
                filename = os.listdir(target_dir)[0]

                logger.info('[downloader] downloading start ({})'.format(file))
                shutil.copy2(target_dir + '/' + filename, path + '/' + filename + '.lock')
                logger.info('[downloader] downloading done ({})'.format(file))

                # done
                logger.info('[downloader] unlock the file ({})'.format(file))
                shutil.move(path + '/' + filename + '.lock', path + '/' + filename)
                break

            logger.info('[downloader] sleep for 0.5 sec..')
            time.sleep(0.5)

    logger.info('[downloader] worker end.')


class DatasetMPManager:

    def __init__(self, file_list):
        logger.info('[DatasetMPManager] init')

        # data
        self.file_list = file_list
        self.idx = 0

        # queue
        self.queue = SimpleQueue()
        self.done_queue = SimpleQueue()

        # temp directory
        self.tmpdir = tempfile.TemporaryDirectory()
        logger.info('[DatasetMPManager] tempdir = {}'.format(self.tmpdir.name))

        # workers
        self.buffer = Process(target=buffer, args=(self.queue, self.done_queue, self.tmpdir.name))
        self.downloader = Process(target=downloader, args=(file_list, self.tmpdir.name))
        self.buffer.start()
        self.downloader.start()

        logger.info('[DatasetMPManager] workers started.')

    def next(self):
        logger.info('[DatasetMPManager] next(): idx = {}'.format(self.idx))

        # get data
        ret = None
        while True:
            if self.queue.empty():
                logger.info('[DatasetMPManager] next(): queue empty. sleep 0.5 sec..')
                time.sleep(0.5)
                continue

            ret = self.queue.get()
            logger.info('[DatasetMPManager] next(): got value')
            break

        self.idx += 1
        if self.idx == len(self.file_list):         # end condition
            logger.info('[DatasetMPManager] next(): iter done. final idx = {}'.format(self.idx))
            # downloader will be done. Only stop buffer
            self.done_queue.put(0)      # arbitary data to stop buffer

        return ret

    def __del__(self):
        logger.info('[DatasetMPManager] done')
        self.tmpdir.cleanup()
