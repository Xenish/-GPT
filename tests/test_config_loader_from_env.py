from __future__ import annotations

import os

import pytest

from finantradealgo.system.config_loader import load_config_from_env


def test_load_config_from_env_defaults_to_research(monkeypatch):
    monkeypatch.delenv("FINANTRADE_PROFILE", raising=False)
    cfg = load_config_from_env()
    assert cfg["profile"] == "research"


def test_load_config_from_env_live(monkeypatch):
    monkeypatch.setenv("FINANTRADE_PROFILE", "live")
    cfg = load_config_from_env()
    assert cfg["profile"] == "live"


def test_load_config_from_env_invalid(monkeypatch):
    monkeypatch.setenv("FINANTRADE_PROFILE", "foobar")
    with pytest.raises(ValueError):
        load_config_from_env()
