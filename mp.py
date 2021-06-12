import logging
import sys
import time
import shutil
import os.path
import tempfile
import pdb # TODO:
import datetime

from multiprocessing import Process, SimpleQueue
from pathlib import Path

import pandas as pd


logging.basicConfig(filename='gene_dataset.log', filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def diff_in_ms(start, end):
    return (end - start).seconds * 1000 + ((end - start).microseconds / 1000)


def buffer(queue, path):
    logger.info('[buffer] worker start.')

    while True:
        matched_name = None
        for item in sorted(Path(path).iterdir(), key=os.path.getmtime):
            if str(item.name).find('.lock') == -1:
                matched_name = str(item.name)
                break

        logger.info('[buffer] matched_name = {}'.format(matched_name))
        if matched_name is not None:       # single item queue
            raw_data = pd.read_csv(path + '/' + matched_name, sep='\t')
            queue.put(raw_data)
            os.remove(path + '/' + matched_name)

        time.sleep(0.5)

    logger.info('[buffer] worker end.')


def downloader(file_list, tcga_base, path, max_size=1073741824):
    logger.info('[downloader] worker start.')

    for file in file_list:
        # download
        logger.info('[downloader] target file = {}'.format(path + '/' + file))
        while True:
            logger.debug('[downloader] {}, {}'.format(os.listdir(path), path))
            logger.info('[downloader] path = {}'.format(path))
            used_size = sum(os.path.getsize(path + '/' + f) for f in os.listdir(path) if os.path.isfile(path + '/' + f))
            logger.info('[downloader] disk used = {}'.format(used_size))

            # Note that the capacity of disk usage can larger than max_size.
            if used_size < max_size or True: # TODO: "or True" is to test maximum disk size
                # Downloading file from mounted path to temp path
                target_dir = tcga_base + '/' + file
                filename = os.listdir(target_dir)[0]

                logger.info('[downloader] downloading start ({})'.format(target_dir + '/' + filename))
                start = datetime.datetime.now()
                shutil.copy2(target_dir + '/' + filename, path + '/' + filename + '.lock')
                end = datetime.datetime.now()
                filesize = os.path.getsize(path + '/' + filename + '.lock')
                diff = diff_in_ms(start, end)
                logger.info('[downloader] downloading done ({}). filesize={}, duration={} ms, throughput={} MB/s'.format(target_dir + '/' + filename,
                    filesize, diff, (filesize / 1024 / 1024) / (diff / 1000)))

                # done
                logger.info('[downloader] unlock the file ({})'.format(target_dir + '/' + filename))
                Path(path + '/' + filename + '.lock').touch()                   # touch needed due to preserve the download order
                shutil.move(path + '/' + filename + '.lock', path + '/' + filename)
                break

            logger.info('[downloader] sleep for 0.5 sec..')
            time.sleep(0.5)


    logger.info('[downloader] worker end. loop end.')


class DatasetMPManager:
    def __init__(self, file_list, tcga_base):
        logger.info('[DatasetMPManager] init(file_list={} of elems)'.format(len(file_list)))
        logger.debug('[DatasetMPManager] {}'.format(file_list))

        # data
        self.file_list = file_list
        self.idx = 0

        # queue
        self.queue = SimpleQueue()

        # temp directory
        self.tmpdir = tempfile.TemporaryDirectory()
        logger.info('[DatasetMPManager] tempdir = {}'.format(self.tmpdir.name))

        # workers
        self.buffer = Process(target=buffer, args=(self.queue, self.tmpdir.name))
        self.downloader = Process(target=downloader, args=(file_list, tcga_base, self.tmpdir.name))
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

        return ret

    def __del__(self):
        logger.info('[DatasetMPManager] cleanup')

        # wait
        self.buffer.terminate()
        self.downloader.terminate()

        self.buffer.join()
        self.downloader.join()

        self.buffer.close()
        self.downloader.close()

        # cleanup
        self.tmpdir.cleanup()

        logger.info('[DatasetMPManager] bye')

