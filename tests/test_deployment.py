"""What the deployment may and may not mount over the image.

Written after a bug that cost a day of unfiltered feed. Coolify rewrites every
relative bind mount in docker-compose.yml into a persistent host directory,
seeded on first deploy and never touched again. `./configs:/app/configs` was in
that list, so production went on reading an annotator_config.json from
2025-08-23 — with no `rubric_detector` section at all — while the image kept
delivering newer code that expects one. The minute of silence, the sirens and
the digests were published for as long as that lasted, and the only trace was a
one-line warning at startup that log rotation had long since dropped.

The lesson is not about `configs/`: it is that a volume mounted over a
directory the image ships silently freezes it. So this checks the rule rather
than the instance — any future directory that ships in the image is protected
by the same test.
"""

import json
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
COMPOSE_PATH = ROOT / "docker-compose.yml"

#: Mounting over these is what froze production. Everything the image carries
#: belongs here; `data` and `models` are deliberately absent, because they are
#: state rather than code and have to survive a redeploy.
SHIPPED_IN_IMAGE = ("/app/configs", "/app/nyan", "/app/crawler", "/app")


@pytest.fixture(scope="module")
def compose() -> dict:
    with open(COMPOSE_PATH) as f:
        return yaml.safe_load(f)


def mounts(service: dict) -> list[str]:
    """Destination paths of a service's bind mounts, without trailing slashes."""
    destinations = []
    for volume in service.get("volumes") or []:
        # Short syntax only, which is what this file uses: "source:target[:mode]".
        target = volume.split(":")[1] if isinstance(volume, str) else volume["target"]
        destinations.append(target.rstrip("/") or "/")
    return destinations


def test_nothing_is_mounted_over_what_the_image_ships(compose: dict) -> None:
    for name, service in compose["services"].items():
        for destination in mounts(service):
            assert destination not in SHIPPED_IN_IMAGE, (
                f"{name} монтує том поверх {destination}, який їде в образі. "
                "Coolify зробить із нього постійний каталог хоста, і цей шлях "
                "перестане оновлюватися деплоями — саме так помер фільтр рубрик."
            )


def test_every_service_only_mounts_state(compose: dict) -> None:
    """The whitelist, stated positively: a new mount has to be a deliberate act."""
    allowed = {"/app/data", "/app/models"}
    for name, service in compose["services"].items():
        unexpected = set(mounts(service)) - allowed
        assert not unexpected, (
            f"{name} монтує {sorted(unexpected)}. Томи — лише для стану; усе "
            "інше має приїхати в образі. Якщо це справді стан, додайте його тут."
        )


def test_the_configs_the_code_reads_are_committed() -> None:
    """The other half of the same guarantee.

    Shipping configs in the image only helps if they are in the repository.
    These are the files the daemon is started with in send.sh; the two with
    secrets are absent by design and written by the entrypoint instead.
    """
    for name in (
        "annotator_config",
        "clusterer_config",
        "daemon_config",
        "ranker_config",
        "renderer_config",
    ):
        path = ROOT / "configs" / f"{name}.json"
        assert path.exists(), f"configs/{name}.json немає в репозиторії"
        with open(path) as f:
            json.load(f)


def test_the_secret_configs_are_never_committed() -> None:
    """Committing one would defeat the entrypoint and leak a bot token.

    Asked of git rather than of the filesystem: locally both files legitimately
    exist, because that is how the daemon is run from a developer machine. What
    must not happen is one of them entering a commit — and from there the image,
    where it would shadow the environment for every deployment at once.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "configs/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    for name in ("client_config", "mongo_config"):
        assert f"configs/{name}.json" not in tracked, (
            f"configs/{name}.json не має бути в git: його створює "
            "docker-entrypoint.sh зі змінних середовища"
        )


def test_the_entrypoint_writes_both_secret_configs() -> None:
    """A generator nobody calls is the same silence in a different place."""
    entrypoint = (ROOT / "docker-entrypoint.sh").read_text()

    assert "create_mongo_config.py" in entrypoint
    assert "create_client_config.py" in entrypoint


def test_every_service_runs_a_script_that_exists(compose: dict) -> None:
    """A typo in a `command:` is a service that restarts forever, silently.

    Nothing else catches it: the image builds, the deploy succeeds, and the
    container enters a restart loop whose only symptom is that whatever that
    service was supposed to do stops happening. The digest going missing is
    noticed within hours; the channel graph going missing is noticed by nobody,
    because the site degrades to a smaller number rather than to an error.
    """
    for name, service in compose["services"].items():
        command = service.get("command") or []
        # Only the ./script.sh form is checked, so a service given a bare command
        # is skipped rather than failed.
        script = command[0] if command else ""
        if not script.startswith("./"):
            continue
        assert (ROOT / script[2:]).exists(), f"{name} запускає {script}, якого немає"


def test_the_number_of_built_services_has_not_grown(compose: dict) -> None:
    """Adding a fifth one broke the deploy before anything was built.

    Coolify does not hand the Dockerfile to the builder as a file. It generates
    one — rewriting every `RUN` with a `--mount=type=secret` per environment
    variable, and prefixing each stage with an `ARG` per variable — base64s it into
    a shell command, and repeats that once per service. With four services and
    about thirty variables that command already sat near the kernel's limit; the
    fifth pushed it over, and the deploy died on `posix_spawn(): Argument list too
    long` with no image and no hint that a service count was the cause.

    So the count is pinned. If this test fails, that is the tradeoff being
    requested, not a mistake: either fold the new work into an existing service —
    which is what the channel graph does, riding along in nyan-app — or first buy
    headroom by merging `RUN` layers in the Dockerfile and dropping environment
    variables that have serviceable defaults.
    """
    built = [name for name, service in compose["services"].items() if service.get("build")]

    assert len(built) <= 4, (
        f"{len(built)} сервісів зі `build:`: {sorted(built)}. Coolify вшиває копію "
        "згенерованого Dockerfile у команду на кожен сервіс, і пʼята копія вже "
        "перевищила ARG_MAX. Читайте докстрінг цього тесту перед тим, як підіймати межу."
    )


def test_every_service_is_given_the_publishing_secret(compose: dict) -> None:
    """The entrypoint refuses to start without it, so a service that never
    receives it would fail on deploy rather than at first publication."""
    for name, service in compose["services"].items():
        environment = service.get("environment") or []
        keys = [entry.split("=")[0] for entry in environment]
        assert "CLIENT_CONFIG_JSON" in keys, f"{name} не отримує CLIENT_CONFIG_JSON"
