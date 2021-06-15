import logging
import sys
import time
import shutil
import os.path
import tempfile
import pdb # TODO:
import datetime

from multiprocessing import Process, Queue
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
        logger.info('[buffer] queue.full() = {} (approximate)'.format(queue.full()))
        if matched_name is not None and queue.full() is False:       # single item queue
            logger.info('[buffer] consume')
            raw_data = pd.read_csv(path + '/' + matched_name, sep='\t')
            queue.put(raw_data)
            os.remove(path + '/' + matched_name)

        time.sleep(0.5)

    logger.info('[buffer] worker end.')


def downloader(file_list, tcga_base, path, max_size=1073741824, cache_dir=None, cache_size=0):
    logger.info('[downloader] worker start.')

    while True:
        for file in file_list:
            # download
            logger.info('[downloader] target file = {}'.format(path + '/' + file))
            while True:
                logger.debug('[downloader] {}, {}'.format(os.listdir(path), path))
                logger.info('[downloader] path = {}'.format(path))
                used_size = sum(os.path.getsize(path + '/' + f) for f in os.listdir(path) if os.path.isfile(path + '/' + f))
                logger.info('[downloader] disk used = {}'.format(used_size))

                # TODO: adogen if statements....
                # Note that the capacity of disk usage can larger than max_size.
                if used_size < max_size or True: # TODO: "or True" is to test maximum disk size
                    # Downloading file from mounted path to temp path
                    target_dir = tcga_base + '/' + file
                    filename = os.listdir(target_dir)[0]

                    path + '/' + filename + '.lock'

                    if os.path.isfile(path + '/' + filename):  # already file exists
                        logger.info('[downloader] downloader already done one cycle. sleep 0.5 sec.')
                        time.sleep(0.5)
                        continue

                    logger.info('[downloader] downloading start ({})'.format(target_dir + '/' + filename))
                    start = datetime.datetime.now()

                    if cache_size > 0:      # Use cache
                        try:    # Cache PATH: cache_dir + '/' + file + '/' + filename
                            logger.info('[downloader] trying to use cache')
                            shutil.copy2(cache_dir + '/' + file + '/' + filename, path + '/' + filename + '.lock')
                            logger.info('[downloader] using cache success')
                        except Exception as e:
                            logger.info('[downloader] trying to use cache failed. use mounted one instead. exception={}'.format(str(e)))
                            shutil.copy2(target_dir + '/' + filename, path + '/' + filename + '.lock')

                    else:       # Use remote
                        shutil.copy2(target_dir + '/' + filename, path + '/' + filename + '.lock')

                    end = datetime.datetime.now()
                    filesize = os.path.getsize(path + '/' + filename + '.lock')
                    diff = diff_in_ms(start, end)
                    logger.info('[downloader] downloading done ({}). filesize={}, duration={} ms, throughput={} MB/s'.format(target_dir + '/' + filename,
                        filesize, diff, (filesize / 1024 / 1024) / (diff / 1000)))

                    def put_to_cache(src, dst, fn):
                        logger.info('[downloader] save to cache dir: cp {} to {}. (replacement)'.format(src, dst + fn))
                        os.mkdir(dst)
                        shutil.copy2(src, dst + fn)
                        # shutil.move(cache_dir + '/' + target_dir + '/' + filename + '.lock', cache_dir + '/' + target_dir + '/' + filename)  # unlock

                    if cache_size > 0:      # try cache replacement
                        if os.path.exists(cache_dir + '/' + file + '/' + filename) is False: # not exists
                            # TODO: If the policy (using first item in the directory) is changed, we should also change this.
                            try:
                                # Collecting all files
                                cache_item_size = []
                                for dir_item in os.listdir(cache_dir):
                                    fn = os.listdir(cache_dir + '/' + dir_item)[0]
                                    fp = cache_dir + '/' + dir_item + '/' + fn
                                    cache_item_size.append((fp, os.path.getsize(fp)))

                                total = sum(map(lambda x: x[1], cache_item_size))
                                logger.info('[downloader] sum of cached items = {}'.format(total))

                                logger.info('[downloader] total({}) + filesize({}) > cache_size({})'.format(total, filesize, cache_size))
                                if (total + filesize) > cache_size:
                                    # eviction call
                                    s = list(sorted(cache_item_size, key=lambda x: x[1]))  # s[-1] is biggest. if no element in s, exception will be called.

                                    logger.info('[downloader] s[-1][1]({}) > filesize({})'.format(s[-1][1], filesize))
                                    if s[-1][1] > filesize: # eviction and replace
                                        logger.info('[downloader] evict {}.'.format(s[-1][0]))
                                        shutil.rmtree(os.path.dirname(s[-1][0]))
                                        put_to_cache(path + '/' + filename + '.lock', cache_dir + '/' + file + '/', filename) # replace

                                    # Note that if filesize is the greatest one, it will be remained as non-cached item.
                                else:
                                    put_to_cache(path + '/' + filename + '.lock', cache_dir + '/' + file + '/', filename) # replace

                                        
                            except Exception as e:
                                logger.warn('[downloader] cache replacement failed : {}'.format(str(e)))

                    # done
                    logger.info('[downloader] unlock the file ({})'.format(target_dir + '/' + filename))
                    Path(path + '/' + filename + '.lock').touch()                   # touch needed due to preserve the download order
                    shutil.move(path + '/' + filename + '.lock', path + '/' + filename)
                    break

                logger.info('[downloader] sleep for 0.5 sec..')
                time.sleep(0.5)


    logger.info('[downloader] worker end. loop end.')


class DatasetMPManager:
    def __init__(self, file_list, tcga_base, opt=True):
        logger.info('[DatasetMPManager] init(file_list={} of elems)'.format(len(file_list)))
        logger.debug('[DatasetMPManager] {}'.format(file_list))

        # data
        self.file_list = file_list
        self.idx = 0

        # queue. 3 is for max buf size 
        self.queue = Queue(3)

        # temp directory
        self.tmpdir = tempfile.TemporaryDirectory()
        logger.info('[DatasetMPManager] tempdir = {}'.format(self.tmpdir.name))

        # caching
        CACHE_DIR = './.cache/'
        CACHE_SIZE = 134217728  # 128MB
        if opt:
            try:
                logger.info('[DatasetMPManager] mkdir for caches.')
                os.mkdir(CACHE_DIR)
            except Exception as e:
                logger.info('[DatasetMPManager] mkdir failed.')

        # workers
        DISK_MAX = 1073741824   # 1GB
        if opt:
            self.downloader = Process(target=downloader, args=(file_list, tcga_base, self.tmpdir.name, DISK_MAX, CACHE_DIR, CACHE_SIZE))
        else:
            self.downloader = Process(target=downloader, args=(file_list, tcga_base, self.tmpdir.name, DISK_MAX, None, 0))

        self.buffer = Process(target=buffer, args=(self.queue, self.tmpdir.name))

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

