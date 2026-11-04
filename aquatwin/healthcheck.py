"""Container health check: healthy if the API or the dashboard in this container answers."""
import sys
import urllib.request

URLS = ("http://localhost:8000/health", "http://localhost:8501/_stcore/health")


def main() -> int:
    for url in URLS:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return 0
        except Exception:
            continue
    return 1


if __name__ == "__main__":
    sys.exit(main())
