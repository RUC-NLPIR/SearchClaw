"""
URL validation for SSRF prevention.

Blocks requests to internal/private IP addresses, cloud metadata
endpoints, and loopback addresses. This prevents the LLM from being
tricked into fetching sensitive internal resources.

Checks performed:
1. URL scheme must be http or https
2. Hostname must not be an IP in a private/reserved range
3. DNS resolution must not point to a private/reserved IP
4. Known cloud metadata endpoints are blocked
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Cloud metadata IP addresses that must always be blocked
_CLOUD_METADATA_IPS = {
    "169.254.169.254",  # AWS, GCP, Azure instance metadata
    "metadata.google.internal",  # GCP alias
    "100.100.100.200",  # Alibaba Cloud metadata
    "fd00:ec2::254",    # AWS IPv6 metadata
}

# Hostnames that must always be blocked
_BLOCKED_HOSTNAMES = {
    "metadata.google.internal",
    "metadata.goog",
    "kubernetes.default.svc",
}

# IP ranges that represent a real SSRF risk — internal networks,
# loopback, link-local, CGNAT, and cloud metadata ranges.
#
# We intentionally do NOT block all of Python's ``ip.is_private`` because
# it includes ranges like 198.18.0.0/15 (benchmark testing), 198.51.100.0/24,
# 203.0.113.0/24, and 192.0.2.0/24 (documentation).  These are commonly
# used by DNS-based content filters (NextDNS, Pi-hole, corporate firewalls)
# as sinkhole addresses.  Legitimate public websites may resolve to these
# IPs when a DNS filter is in the path.
_SSRF_BLOCKED_NETWORKS_V4 = [
    ipaddress.IPv4Network("0.0.0.0/8"),       # "This host on this network" (RFC 1122)
    ipaddress.IPv4Network("10.0.0.0/8"),      # Private (RFC 1918)
    ipaddress.IPv4Network("100.64.0.0/10"),   # CGNAT shared address (RFC 6598)
    ipaddress.IPv4Network("127.0.0.0/8"),     # Loopback (RFC 1122)
    ipaddress.IPv4Network("169.254.0.0/16"),  # Link-local (RFC 3927)
    ipaddress.IPv4Network("172.16.0.0/12"),   # Private (RFC 1918)
    ipaddress.IPv4Network("192.168.0.0/16"),  # Private (RFC 1918)
]

_SSRF_BLOCKED_NETWORKS_V6 = [
    ipaddress.IPv6Network("::1/128"),          # Loopback
    ipaddress.IPv6Network("fc00::/7"),         # Unique local (RFC 4193)
    ipaddress.IPv6Network("fe80::/10"),        # Link-local (RFC 4291)
    ipaddress.IPv6Network("::ffff:0:0/96"),    # IPv4-mapped — check the v4 part too
]


def _is_ssrf_ip(ip_str: str) -> bool:
    """
    Check if an IP address is in a range that poses an SSRF risk.

    Only blocks ranges that could reach internal services — private
    networks (RFC 1918), loopback, link-local, CGNAT.  Does NOT block
    IANA "reserved" ranges like 198.18.0.0/15 that DNS sinkholers use.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        # Not a valid IP — that's fine, it's a hostname
        return False

    if ip.is_multicast or ip.is_unspecified:
        return True

    if isinstance(ip, ipaddress.IPv4Address):
        return any(ip in net for net in _SSRF_BLOCKED_NETWORKS_V4)

    if isinstance(ip, ipaddress.IPv6Address):
        if any(ip in net for net in _SSRF_BLOCKED_NETWORKS_V6):
            return True
        # For IPv4-mapped IPv6 addresses (::ffff:x.x.x.x), also check
        # the embedded IPv4 address
        mapped_v4 = ip.ipv4_mapped
        if mapped_v4 is not None:
            return any(mapped_v4 in net for net in _SSRF_BLOCKED_NETWORKS_V4)

    return False


def validate_url_for_ssrf(url: str) -> tuple[bool, str]:
    """
    Validate a URL to prevent SSRF attacks.

    Returns (is_safe, reason). If is_safe is False, the reason
    explains why the URL was blocked.

    Checks:
    1. Scheme must be http/https
    2. Hostname must not be a known metadata endpoint
    3. If hostname is an IP literal, it must not be private/reserved
    4. DNS resolution of hostname must not point to private/reserved IPs
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL format"

    # Check scheme
    if parsed.scheme not in ("http", "https"):
        return False, f"Blocked scheme: {parsed.scheme}. Only http/https allowed."

    hostname = parsed.hostname or ""
    if not hostname:
        return False, "URL has no hostname"

    hostname_lower = hostname.lower().strip(".")

    # Check blocked hostnames
    if hostname_lower in _BLOCKED_HOSTNAMES:
        return False, f"Blocked hostname: {hostname}"

    # Check if hostname is a cloud metadata IP
    if hostname_lower in _CLOUD_METADATA_IPS:
        return False, f"Blocked cloud metadata endpoint: {hostname}"

    # Check if hostname is an IP literal in a dangerous range
    if _is_ssrf_ip(hostname):
        return False, f"Blocked private/internal IP: {hostname}"

    # DNS resolution check — resolve hostname and verify the IP
    try:
        # Use getaddrinfo for both IPv4 and IPv6
        results = socket.getaddrinfo(
            hostname, parsed.port or (443 if parsed.scheme == "https" else 80),
            proto=socket.IPPROTO_TCP,
        )
        for family, _type, _proto, _canonname, sockaddr in results:
            ip_str = sockaddr[0]
            if _is_ssrf_ip(ip_str):
                return False, (
                    f"Blocked: hostname '{hostname}' resolves to private/internal "
                    f"IP {ip_str}"
                )
    except socket.gaierror:
        # DNS resolution failed — allow the request (httpx will handle
        # the connection error). We don't want to block URLs that happen
        # to have temporary DNS issues.
        pass
    except Exception as e:
        logger.warning(f"URL validation DNS check failed for {hostname}: {e}")
        # On unexpected errors, allow the request (fail-open for DNS
        # check only — the other checks are strict)
        pass

    return True, ""
