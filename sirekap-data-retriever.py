# Description: A simple script to retrieve all polling station voting data from SIREKAP API.
#              WARNING: This script will take a while to run and if retrieving images, 
#                       will consume a lot of bandwidth and disk space (~1TB).
# 
#              Please use responsibly, I urge users to run this script only outside of Indonesia's waking hours.
# Author: Raka Gunarto (https://github.com/raka-gunarto)

import logging
import json
import argparse
import sys
import requests
import aiohttp
import asyncio
import os

ROOT_STATION_URL = "https://sirekap-obj-data.kpu.go.id/wilayah/pemilu/ppwp/0.json"
EXPECTED_STATIONS = 823236 

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    LOG_LEVELS = {
        logging.DEBUG: (grey, ""),
        logging.INFO: (blue, ""),
        logging.WARNING: (yellow, ""),
        logging.ERROR: (red, ""),
        logging.CRITICAL: (bold_red, ""),
    }

    def format(self, record):
        log_fmt = self.LOG_LEVELS.get(record.levelno, (self.grey, ""))
        color, emoji = log_fmt
        formatter = logging.Formatter(f"%(asctime)s - [{color}%(levelname)s{self.reset}{emoji}]: %(message)s")
        return formatter.format(record)

def setup_logger(log_level, filename):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(CustomFormatter())
    
    fh = logging.FileHandler(filename, mode="w")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s]: %(message)s"))
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger

def parse_args():
    args = argparse.ArgumentParser(description="Sirekap Data Retriever")
    args.add_argument("-l", "--log-level", default="INFO", help="Log level")
    args.add_argument("-f", "--log-file", default="sirekap-data-retriever.log", help="Log file")
    args.add_argument("-c", "--max-concurrent-requests", default=500, type=int, help="Maximum concurrent requests")
    args.add_argument("-u", "--url-input-file", default=None, help="Override polling station root url with list of urls from JSON file")
    args.add_argument("-o", "--cache-file", default="polling_stations.json", help="Output cache file for polling stations")
    args.add_argument("-i", "--get-images", action="store_true", help="Retrieve images for polling stations")
    args.add_argument("--image-cache-dir", default="/mnt/data-sirekap", help="Directory to store polling station images")
    args.add_argument("--refresh-images", action="store_true", help="Force refresh images for polling stations")
    output_group = args.add_mutually_exclusive_group()
    output_group.add_argument("--force-refresh", action="store_true", help="Force refresh polling stations")
    output_group.add_argument("--append-to-cache", action="store_true", help="Append to cache instead of overwriting")
    output_group.add_argument("--use-cached-urls", action="store_true", help="Use cached data URLs instead of full recursive search")

    return args.parse_args()

async def fetch_station(session, url_and_path, urls, stations):
    current_path, url = url_and_path 
    logging.debug(f"Fetching {url}")
    async with session.get(url) as response:
        if response.status != 200:
            logging.error(f"Error fetching {url}: {response.status}, will retry")
            urls.append(url_and_path) 
            return
        data = await response.json()
        if isinstance(data, list):  # Not at a polling station (leaf) yet
            for next in data:
                next_path = f"{current_path}/{next['nama']}"
                next_url = url.removesuffix("/0.json").removesuffix(".json") + f"/{next['kode']}.json"
                if next['tingkat'] == 5:
                    next_url = next_url.replace("wilayah/pemilu", "pemilu/hhcw")
                logging.debug(f"Adding {(next_path, next_url)} to urls")
                urls.append((next_path, next_url))
        elif isinstance(data, dict):  # Polling station
            stations[current_path] = {
                "data_url": url,
                **data
            }
        else:
            logging.error(f"Unexpected data type: {type(data)} from {url}")

async def get_polling_stations(force_refresh: bool, max_concurrent_requests: int = 8, urls_override: list | None = None, append: bool = False, filename: str = "polling_stations", use_cached_urls: bool = False):
    stations = {}

    # check if we have a cached version of the polling stations
    if not force_refresh:
        logging.info("Checking for cached polling stations")
        try:
            with open(filename, "r") as f:
                logging.info("Loading cached polling stations")
                stations = json.load(f)
                if not append and not use_cached_urls: # if we're not appending, return the cached polling stations
                    return stations
        except FileNotFoundError:
            pass
        except Exception as e:
            logging.warning(f"Error loading cached polling stations: {e}")
    
    # force refresh or no cached polling stations
    logging.info("No cached polling stations found, retrieving from Sirekap API")

    # recursively retrieve polling stations from API
    urls = []
    if urls_override:
        urls = urls_override
    else:
        urls = [("", ROOT_STATION_URL)]

    if use_cached_urls:
        if len(stations) == 0:
            logging.error("No cached polling stations found, cannot use cached URLs")
            exit(1)
        urls = [(k, v['data_url']) for k, v in stations.items()]


    async with aiohttp.ClientSession() as session:
        tasks = []
        while urls or tasks:
            print(f"{len(urls)} left to process (number may increase as we process the current url). {len(stations)} retrieved so far             ", end="\r")
            if not urls:
                await asyncio.wait(tasks)
                tasks = [t for t in tasks if not t.done()]
            else:
                url_and_path = urls.pop(0)
                task = asyncio.create_task(fetch_station(session, url_and_path, urls, stations))
                tasks.append(task)
            
            if len(tasks) >= max_concurrent_requests:  # limit concurrent requests
                await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
                tasks = [t for t in tasks if not t.done()]

        # Wait for any remaining tasks
        if tasks:
            await asyncio.wait(tasks)
        
    # save polling stations to file
    with open(filename, "w") as f:
        json.dump(stations, f)
    
    # return polling stations
    return stations

async def fetch_summary_sheet(session, url, image_cache_dir):
    logging.debug(f"Fetching {url}")
    async with session.get(url) as response:
        if response.status != 200:
            logging.error(f"Error fetching {url}: {response.status}")
            return
        data = await response.read()
        with open(f"{image_cache_dir}/{url.split('/')[-1]}", "wb") as f:
            f.write(data)

async def get_polling_stations_summary_sheets(stations: dict, force_refresh: bool, max_concurrent_requests: int = 8, image_cache_dir: str = "/mnt/data-sirekap"):
    image_urls = []
    for polling_station, details in stations.items():
        url = details['images'][1]
        if url and (force_refresh or not os.path.exists(f"{image_cache_dir}/{url.split('/')[-1]}")):
            image_urls.append(url)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        while image_urls or tasks:
            print(f"{len(image_urls)} left.             ", end="\r")
            if not image_urls:
                await asyncio.wait(tasks)
                tasks = [t for t in tasks if not t.done()]
            else:
                url = image_urls.pop(0)
                task = asyncio.create_task(fetch_summary_sheet(session, url, image_cache_dir))
                tasks.append(task)
            
            if len(tasks) >= max_concurrent_requests:
                await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
                tasks = [t for t in tasks if not t.done()]

async def main():
    args = parse_args()
    logging = setup_logger(args.log_level, args.log_file)

    logging.info("Starting Sirekap Data Retriever")

    logging.info("Getting polling stations")

    urls = None
    if args.url_input_file:
        logging.info(f"Using input file: {args.url_input_file}")
        urls = json.load(open(args.url_input_file, "r"))
    
    logging.info(f"Using cache file: {args.cache_file}")

    polling_stations = await get_polling_stations(args.force_refresh, args.max_concurrent_requests, urls, args.append_to_cache, args.cache_file, args.use_cached_urls)
    if len(polling_stations) != EXPECTED_STATIONS:
        logging.warning(f"Expected {EXPECTED_STATIONS} polling stations, got {len(polling_stations)}")

    logging.info("Polling stations retrieved")

    if not args.get_images:
        logging.info("Images of summary sheets not requested, exiting")
        return
    
    logging.info("Getting images for polling stations summary sheets")

    await get_polling_stations_summary_sheets(polling_stations, args.refresh_images, args.max_concurrent_requests, args.image_cache_dir)

    logging.info("Images for polling stations summary sheets retrieved")
    





if __name__ == "__main__":
	asyncio.run(main())