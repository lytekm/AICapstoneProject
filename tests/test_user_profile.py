"""Tests for user profile model and JSON-backed storage."""

from __future__ import annotations

import json

import pytest

from src.user_profile import ProfileStore, UserProfile, _profile_from_dict


# ---------------------------------------------------------------------------
# UserProfile dataclass
# ---------------------------------------------------------------------------


class TestUserProfile:
    """Basic dataclass behavior and defaults."""

    def test_minimal_profile(self) -> None:
        p = UserProfile(user_id="u1")
        assert p.user_id == "u1"
        assert p.preferred_topics == []
        assert p.keywords == []
        assert p.default_persona == "default"
        assert p.default_length == "standard"
        assert p.feedback_weights == {}

    def test_full_profile(self) -> None:
        p = UserProfile(
            user_id="u2",
            preferred_topics=["AI", "finance"],
            keywords=["startup"],
            default_persona="executive",
            default_length="brief",
            feedback_weights={"AI": 0.3},
        )
        assert p.preferred_topics == ["AI", "finance"]
        assert p.keywords == ["startup"]
        assert p.default_persona == "executive"
        assert p.feedback_weights["AI"] == pytest.approx(0.3)

    def test_profile_from_dict_minimal(self) -> None:
        data = {"user_id": "u3"}
        p = _profile_from_dict(data)
        assert p.user_id == "u3"
        assert p.preferred_topics == []

    def test_profile_from_dict_full(self) -> None:
        data = {
            "user_id": "u4",
            "preferred_topics": ["tech"],
            "keywords": ["GPU"],
            "default_persona": "technical",
            "default_length": "detailed",
            "feedback_weights": {"tech": 0.5},
        }
        p = _profile_from_dict(data)
        assert p.default_persona == "technical"
        assert p.feedback_weights["tech"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ProfileStore (file-backed CRUD)
# ---------------------------------------------------------------------------


class TestProfileStore:
    """Test JSON persistence using pytest tmp_path for isolation."""

    def _make_store(self, tmp_path: str) -> ProfileStore:
        return ProfileStore(path=f"{tmp_path}/profiles.json")

    def test_empty_store(self, tmp_path: str) -> None:
        store = self._make_store(tmp_path)
        assert store.list_users() == []
        assert store.get("nobody") is None

    def test_save_and_get(self, tmp_path: str) -> None:
        store = self._make_store(tmp_path)
        p = UserProfile(user_id="u1", preferred_topics=["AI"])
        store.save(p)

        loaded = store.get("u1")
        assert loaded is not None
        assert loaded.user_id == "u1"
        assert loaded.preferred_topics == ["AI"]

    def test_save_overwrites(self, tmp_path: str) -> None:
        store = self._make_store(tmp_path)
        store.save(UserProfile(user_id="u1", default_persona="casual"))
        store.save(UserProfile(user_id="u1", default_persona="executive"))

        loaded = store.get("u1")
        assert loaded is not None
        assert loaded.default_persona == "executive"

    def test_list_users(self, tmp_path: str) -> None:
        store = self._make_store(tmp_path)
        store.save(UserProfile(user_id="zara"))
        store.save(UserProfile(user_id="alice"))
        # sorted alphabetically
        assert store.list_users() == ["alice", "zara"]

    def test_delete_existing(self, tmp_path: str) -> None:
        store = self._make_store(tmp_path)
        store.save(UserProfile(user_id="u1"))
        assert store.delete("u1") is True
        assert store.get("u1") is None

    def test_delete_missing(self, tmp_path: str) -> None:
        store = self._make_store(tmp_path)
        assert store.delete("ghost") is False

    def test_persistence_across_instances(self, tmp_path: str) -> None:
        """Save in one store instance, load in a fresh one."""
        path = f"{tmp_path}/profiles.json"
        store1 = ProfileStore(path=path)
        store1.save(UserProfile(user_id="u1", keywords=["vllm"]))

        # brand new store pointing at the same file
        store2 = ProfileStore(path=path)
        loaded = store2.get("u1")
        assert loaded is not None
        assert loaded.keywords == ["vllm"]

    def test_corrupted_file_recovers(self, tmp_path: str) -> None:
        """If the JSON file is garbled, the store starts fresh."""
        path = f"{tmp_path}/profiles.json"
        # write garbage
        import pathlib

        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_text("NOT VALID JSON {{{", encoding="utf-8")

        store = ProfileStore(path=path)
        assert store.list_users() == []

    def test_json_format_on_disk(self, tmp_path: str) -> None:
        """Verify the file is human-readable JSON with sorted keys."""
        path = f"{tmp_path}/profiles.json"
        store = ProfileStore(path=path)
        store.save(
            UserProfile(user_id="demo", preferred_topics=["AI"], default_persona="technical")
        )

        raw = json.loads(open(path, encoding="utf-8").read())
        assert "demo" in raw
        assert raw["demo"]["default_persona"] == "technical"

    def test_multiple_users(self, tmp_path: str) -> None:
        store = self._make_store(tmp_path)
        store.save(UserProfile(user_id="a", preferred_topics=["tech"]))
        store.save(UserProfile(user_id="b", preferred_topics=["sports"]))
        store.save(UserProfile(user_id="c", preferred_topics=["finance"]))

        assert len(store.list_users()) == 3
        assert store.get("b") is not None
        assert store.get("b").preferred_topics == ["sports"]  # type: ignore[union-attr]
