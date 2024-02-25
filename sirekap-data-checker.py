# Description: A simple script to check discrepancies in polling station data from SIREKAP API
# Author: Raka Gunarto (https://github.com/raka-gunarto)

import logging
import json
import argparse
import sys
import pprint

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
    args = argparse.ArgumentParser(description="Sirekap Data Checker")
    args.add_argument("-l", "--log-level", default="INFO", help="Log level")
    args.add_argument("-f", "--log-file", default="sirekap-data-checker.log", help="Log file")
    args.add_argument("-c", "--cache-file", default="polling_stations.json", help="Cache file for polling stations")

    return args.parse_args()

def main():
    args = parse_args()
    setup_logger(args.log_level, args.log_file)

    logging.info("Starting Sirekap Data Checker")

    logging.info(f"Loading polling stations from {args.cache_file}...")
    data = dict(json.load(open(args.cache_file, "r")))
    logging.info(f"Loaded {args.cache_file} with {len(data)} polling stations")

    logging.info("Checking polling station data...")
    discrepancies = []
    candidate1 = 0
    candidate2 = 0
    candidate3 = 0
    suara_sah_reported = 0
    total_votes_reported = 0
    total_vote_delta = 0
    skipped = 0
    for polling_station, details in data.items():
        try:
            if type(details['chart']) != dict or type(details['administrasi']) != dict or details['administrasi']['suara_sah'] is None or len(details['chart'].values()) < 4:
                skipped += 1
                continue
            total_suara_sah = details['administrasi']['suara_sah']
            total_votes = sum([int(x) for x in details['chart'].values() if type(x) == int])

            suara_sah_reported += total_suara_sah
            total_votes_reported += total_votes

            if total_suara_sah != total_votes:
                logging.debug(f"Discrepancy found in {polling_station}: total suara sah {total_suara_sah} != total votes {total_votes}")
                discrepancies.append(polling_station)
                total_vote_delta += abs(total_suara_sah - total_votes)
                continue
                
            # count is valid, add counts
            candidate1 += int(details['chart']['100025'])
            candidate2 += int(details['chart']['100026'])
            candidate3 += int(details['chart']['100027'])

        except:
            logging.error(f"Error checking {polling_station}")
            logging.error(f"Details: {details}")
            continue
    
    logging.info(f"Skipped {skipped} polling stations with incomplete data")
    logging.info(f"Discrepancies found in {len(discrepancies)} polling stations")
    logging.info(f"Total suara sah: {suara_sah_reported} vs total votes: {total_votes_reported}")
    logging.info(f"Total vote delta: {total_vote_delta}")
    logging.info("Vote recount with only valid polling stations (without image checking)")
    logging.info(f"Candidate 1: {candidate1} ({candidate1/(candidate1+candidate2+candidate3)*100:.2f}%)")
    logging.info(f"Candidate 2: {candidate2} ({candidate2/(candidate1+candidate2+candidate3)*100:.2f}%)")
    logging.info(f"Candidate 3: {candidate3} ({candidate3/(candidate1+candidate2+candidate3)*100:.2f}%)")
    logging.info(f"Inconsistent polling stations that favours candidate 1: {len([x for x in discrepancies if data[x]['chart']['100025'] > data[x]['chart']['100026'] and data[x]['chart']['100025'] > data[x]['chart']['100027']])}")
    logging.info(f"Total votes for candidate 1 in inconsistent stations that favours candidate 1: {sum([data[x]['chart']['100025'] for x in discrepancies if data[x]['chart']['100025'] > data[x]['chart']['100026'] and data[x]['chart']['100025'] > data[x]['chart']['100027']])}")
    logging.info(f"Inconsistent polling stations that favours candidate 2: {len([x for x in discrepancies if data[x]['chart']['100026'] > data[x]['chart']['100025'] and data[x]['chart']['100026'] > data[x]['chart']['100027']])}")
    logging.info(f"Total votes for candidate 2 in inconsistent stations that favours candidate 2: {sum([data[x]['chart']['100026'] for x in discrepancies if data[x]['chart']['100026'] > data[x]['chart']['100025'] and data[x]['chart']['100026'] > data[x]['chart']['100027']])}")
    logging.info(f"Inconsistent polling stations that favours candidate 3: {len([x for x in discrepancies if data[x]['chart']['100027'] > data[x]['chart']['100025'] and data[x]['chart']['100027'] > data[x]['chart']['100026']])}")
    logging.info(f"Total votes for candidate 3 in inconsistent stations that favours candidate 3: {sum([data[x]['chart']['100027'] for x in discrepancies if data[x]['chart']['100027'] > data[x]['chart']['100025'] and data[x]['chart']['100027'] > data[x]['chart']['100026']])}")
    candidate_2_favorites_big = [x for x in discrepancies if data[x]['chart']['100026'] > data[x]['chart']['100025'] and data[x]['chart']['100026'] > data[x]['chart']['100027'] and str(data[x]['chart']['100026']).startswith('9')]
    logging.info(f"Inconsistent polling stations that favours candidate 2 and starts with 9: {len(candidate_2_favorites_big)}")
    logging.info(f"Random 10 samples: {pprint.pformat([(x,data[x]['images'][1]) for x in candidate_2_favorites_big])}")
    

if __name__ == "__main__":
    main()