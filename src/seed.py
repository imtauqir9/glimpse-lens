"""One-shot sample seeder — the startup gate.

    python -m src.seed     # exits 0 when all four samples are indexed, else 1

docker-compose runs this as a service the api + worker depend on via
`service_completed_successfully`, so the UI only becomes reachable once the
sample corpus is fully indexed. On Fly it's the [deploy] release_command.
Durable in Qdrant Cloud, so after the first successful run this exits in
seconds. Set SEED_SAMPLE_VIDEOS=false to skip the gate entirely.
"""
import sys

from .seeding import seed_to_completion


def main() -> int:
    ok = seed_to_completion()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
