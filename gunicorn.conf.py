import multiprocessing
import os

bind = os.getenv("PORT", "5000")
bind = f"0.0.0.0:{bind}"

# Use 1-2 workers for small free/basic instances; adjust by CPU count
workers = int(os.getenv("WEB_CONCURRENCY", max(1, multiprocessing.cpu_count() // 2)))
threads = int(os.getenv("GUNICORN_THREADS", 2))
timeout = int(os.getenv("GUNICORN_TIMEOUT", 120))
loglevel = os.getenv("GUNICORN_LOGLEVEL", "info")

# Forwarded allow for proxies
forwarded_allow_ips = "*"
proxy_allow_ips = "*"

# Keepalive
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", 5))
