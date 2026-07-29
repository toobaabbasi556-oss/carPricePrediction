"""robots.txt fetch + parse (SPEC §2.1 rule 1).

HARD RULE: this must run before any scraping. We print the disallowed paths and
any ``Crawl-delay`` for our user-agent; if ``/used-cars/`` is disallowed we STOP
and escalate rather than scrape it.

We parse robots.txt ourselves (instead of only using ``urllib.robotparser``) so we
can surface the exact matched group, its disallow list, and the crawl-delay for the
report the phase requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import requests


@dataclass
class RobotsRules:
    """Directives from the robots.txt group that applies to our user-agent."""

    user_agent_pattern: str  # the robots.txt User-agent token that matched (e.g. "*")
    disallow: list[str] = field(default_factory=list)
    allow: list[str] = field(default_factory=list)
    crawl_delay: float | None = None
    raw: str = ""

    def can_fetch(self, path: str) -> bool:
        """Standard longest-match rule: most specific Allow/Disallow wins; ties -> Allow."""
        best_len = -1
        decision = True  # default allow when nothing matches
        rules = [(p, False) for p in self.disallow] + [(p, True) for p in self.allow]
        for rule_path, allowed in rules:
            if rule_path == "":
                continue  # empty Disallow means "allow everything" for that group
            if path.startswith(rule_path):
                # Longest match wins; on an equal-length tie, Allow wins.
                if len(rule_path) > best_len or (len(rule_path) == best_len and allowed):
                    best_len = len(rule_path)
                    decision = allowed
        return decision


def fetch_robots_text(robots_url: str, user_agent: str, timeout: int = 30) -> str:
    """Fetch the raw robots.txt body with our descriptive User-Agent."""
    resp = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _ua_matches(token: str, user_agent: str) -> bool:
    """A robots.txt User-agent token applies if it is ``*`` or a substring of our UA."""
    token = token.strip().lower()
    if token == "*":
        return True
    return token in user_agent.lower()


def parse_robots(text: str, user_agent: str) -> RobotsRules:
    """Parse robots.txt and return the rules for the group matching our user-agent.

    Prefers the most specific matching group (a named token) over the wildcard ``*``.
    """
    # Split into groups keyed by their user-agent tokens.
    groups: list[tuple[list[str], list[tuple[str, str]]]] = []
    current_agents: list[str] = []
    current_rules: list[tuple[str, str]] = []
    expecting_agent = False

    def flush() -> None:
        if current_agents:
            groups.append((current_agents.copy(), current_rules.copy()))

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        key_l = key.lower()
        if key_l == "user-agent":
            if not expecting_agent and current_agents:
                flush()
                current_agents = []
                current_rules = []
            current_agents.append(value)
            expecting_agent = True
        else:
            expecting_agent = False
            if key_l in {"disallow", "allow", "crawl-delay"}:
                current_rules.append((key_l, value))
    flush()

    # Choose the applicable group: most specific named match, else "*".
    specific: RobotsRules | None = None
    wildcard: RobotsRules | None = None
    for agents, rules in groups:
        for agent in agents:
            if not _ua_matches(agent, user_agent):
                continue
            rr = RobotsRules(user_agent_pattern=agent, raw=text)
            for key_l, value in rules:
                if key_l == "disallow":
                    rr.disallow.append(value)
                elif key_l == "allow":
                    rr.allow.append(value)
                elif key_l == "crawl-delay":
                    try:
                        rr.crawl_delay = float(value)
                    except ValueError:
                        pass
            if agent.strip() == "*":
                wildcard = rr
            else:
                specific = rr
    return specific or wildcard or RobotsRules(user_agent_pattern="(none)", raw=text)


def load_rules(robots_url: str, user_agent: str, timeout: int = 30) -> RobotsRules:
    """Fetch + parse robots.txt for our user-agent in one call."""
    return parse_robots(fetch_robots_text(robots_url, user_agent, timeout), user_agent)
