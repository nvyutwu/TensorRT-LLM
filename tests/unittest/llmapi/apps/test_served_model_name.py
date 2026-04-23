# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for multi-name / alias handling in OpenAIServer."""

import json
from http import HTTPStatus
from types import SimpleNamespace

import pytest

from tensorrt_llm.commands.serve import _resolve_served_model_names
from tensorrt_llm.serve.openai_server import OpenAIServer


def test_normalize_single_name():
    assert OpenAIServer._normalize_model_names(["m"]) == ("m", ["m"])


def test_normalize_dedup_preserves_order():
    primary, aliases = OpenAIServer._normalize_model_names(
        ("primary", "a1", "primary", "a2", "a1"))
    assert primary == "primary"
    assert aliases == ["primary", "a1", "a2"]


def test_normalize_directory_path_uses_basename(tmp_path):
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    primary, aliases = OpenAIServer._normalize_model_names(
        [str(model_dir), "alias"])
    assert primary == "ckpt"
    assert aliases == ["ckpt", "alias"]


def _make_server(primary, aliases):
    server = OpenAIServer.__new__(OpenAIServer)
    server.model = primary
    server.served_model_names = aliases
    return server


@pytest.mark.parametrize("name", ["primary", "alias1", "alias2"])
def test_is_model_supported_known(name):
    server = _make_server("primary", ["primary", "alias1", "alias2"])
    assert server._is_model_supported(name) is True


@pytest.mark.parametrize("name", [None, ""])
def test_is_model_supported_empty_is_ok(name):
    """vLLM-parity: empty/None client-supplied model is treated as valid."""
    server = _make_server("primary", ["primary", "alias1"])
    assert server._is_model_supported(name) is True


def test_is_model_supported_unknown():
    server = _make_server("primary", ["primary", "alias1"])
    assert server._is_model_supported("not-an-alias") is False


def test_check_model_accepts_known_alias():
    server = _make_server("primary", ["primary", "alias1"])
    request = SimpleNamespace(model="alias1")
    assert server._check_model(request) is None


def test_check_model_rejects_unknown_with_404():
    server = _make_server("primary", ["primary", "alias1"])
    request = SimpleNamespace(model="not-an-alias")
    response = server._check_model(request)
    assert response is not None
    assert response.status_code == HTTPStatus.NOT_FOUND
    body = json.loads(response.body)
    assert body["type"] == "NotFoundError"
    assert "not-an-alias" in body["message"]


@pytest.mark.parametrize("flag,expected", [
    (None, ["/p/m"]),
    ((), ["/p/m"]),
    (("foo", ), ["foo"]),
    (("foo", "bar"), ["foo", "bar"]),
    (("foo", "", "bar"), ["foo", "bar"]),
    (("foo", "bar", "foo"), ["foo", "bar", "foo"]),
])
def test_resolve_served_model_names_cli(flag, expected):
    assert _resolve_served_model_names(flag, {"model": "/p/m"}) == expected
