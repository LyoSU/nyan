import argparse
import logging
import os

from nyan.daemon import Daemon


def setup_logging() -> None:
    # Unbuffered stdout so `docker logs -f` shows progress as it happens
    # instead of in blocks whenever the pipe buffer fills.
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL") or "INFO",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(
    input_path: str | None,
    posted_clusters_path: str | None,
    client_config_path: str,
    annotator_config_path: str,
    clusterer_config_path: str,
    ranker_config_path: str,
    channels_info_path: str,
    renderer_config_path: str,
    mongo_config_path: str | None,
    daemon_config_path: str,
) -> None:
    setup_logging()
    daemon = Daemon(
        client_config_path=client_config_path,
        annotator_config_path=annotator_config_path,
        clusterer_config_path=clusterer_config_path,
        ranker_config_path=ranker_config_path,
        channels_info_path=channels_info_path,
        renderer_config_path=renderer_config_path,
        daemon_config_path=daemon_config_path,
    )
    daemon.run(input_path, mongo_config_path, posted_clusters_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--mongo-config-path", type=str, default=None)
    parser.add_argument("--posted-clusters-path", type=str, default=None)
    parser.add_argument("--channels-info-path", type=str, default="channels.json")
    parser.add_argument(
        "--client-config-path", type=str, default="configs/client_config.json"
    )
    parser.add_argument(
        "--annotator-config-path", type=str, default="configs/annotator_config.json"
    )
    parser.add_argument(
        "--clusterer-config-path", type=str, default="configs/clusterer_config.json"
    )
    parser.add_argument(
        "--renderer-config-path", type=str, default="configs/renderer_config.json"
    )
    parser.add_argument(
        "--ranker-config-path", type=str, default="configs/ranker_config.json"
    )
    parser.add_argument(
        "--daemon-config-path", type=str, default="configs/daemon_config.json"
    )
    args = parser.parse_args()
    main(**vars(args))
