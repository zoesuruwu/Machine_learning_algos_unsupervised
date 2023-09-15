import logging

from run_sample_data import seeds_data


def main():
    # Configure the logging settings
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    seeds_data(logger)


if __name__ == "__main__":
    main()
