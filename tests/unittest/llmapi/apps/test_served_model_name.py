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

import pytest

from tensorrt_llm.serve.openai_server import OpenAIServer


def test_normalize_single_string():
    assert OpenAIServer._normalize_model_names("m") == ("m", ["m"])


def test_normalize_dedup_preserves_order():
    primary, aliases = OpenAIServer._normalize_model_names(("primary", "a1", "primary", "a2", "a1"))
    assert primary == "primary"
    assert aliases == ["primary", "a1", "a2"]


def test_normalize_directory_path_uses_basename(tmp_path):
    model_dir = tmp_path / "ckpt"
    model_dir.mkdir()
    primary, aliases = OpenAIServer._normalize_model_names([str(model_dir), "alias"])
    assert primary == "ckpt"
    assert aliases == ["ckpt", "alias"]


def _make_server(primary, aliases):
    server = OpenAIServer.__new__(OpenAIServer)
    server.model = primary
    server.served_model_names = aliases
    return server


def test_resolve_known_alias_is_echoed_back():
    server = _make_server("primary", ["primary", "alias1", "alias2"])
    assert server._resolve_model_name("alias1") == "alias1"
    assert server._resolve_model_name("primary") == "primary"


@pytest.mark.parametrize("requested", [None, "", "not-an-alias"])
def test_resolve_unknown_falls_back_to_primary(requested):
    server = _make_server("primary", ["primary", "alias1"])
    assert server._resolve_model_name(requested) == "primary"
