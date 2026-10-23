"""Turns the picks payload into the dashboard page."""
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("template.html")


def fragment(payload):
    data = json.dumps(payload, default=str).replace("</", "<\\/")
    return TEMPLATE.read_text().replace("__DATA__", data)


def page(payload):
    """Full standalone document (for GitHub Pages or opening locally)."""
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '</head><body>\n' + fragment(payload) + "\n</body></html>\n")
