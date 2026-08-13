# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import pickle
import tarfile
import zipfile
from pathlib import Path

import pytest
import torch

import physicsnemo.core
from physicsnemo.core import ModelRegistry


# Fixture to clear registry between tests to avoid naming conflicts
@pytest.fixture(autouse=True)
def clear_registry():
    """Clear and restore the model registry before and after each test"""
    registry = ModelRegistry()
    registry.__clear_registry__()
    yield
    registry.__restore_registry__()


class MockModel(physicsnemo.core.Module):
    """Fake model"""

    def __init__(self, layer_size=16):
        super().__init__()
        self.layer_size = layer_size
        self.layer = torch.nn.Linear(layer_size, layer_size)


class NewMockModel(physicsnemo.core.Module):
    """Fake model"""

    def __init__(self, layer_size=16):
        super().__init__()
        self.layer_size = layer_size
        self.layer = torch.nn.Linear(layer_size, layer_size)


class MockModelNoOverride(physicsnemo.core.Module):
    """Fake model"""

    def __init__(self, value1, value2, x):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(value1, dtype=torch.float32))
        self.w2 = torch.nn.Parameter(torch.tensor(value2, dtype=torch.float32))
        self.x = x


class MockModelWithOverride(physicsnemo.core.Module):
    """Fake model"""

    _overridable_args = {"value2", "x"}

    def __init__(self, value1, value2, x):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(value1, dtype=torch.float32))
        self.w2 = torch.nn.Parameter(torch.tensor(value2, dtype=torch.float32))
        self.x = x


class NestedMockModel(physicsnemo.core.Module):
    """Model used to exercise malformed nested checkpoint metadata."""

    def __init__(self, child):
        super().__init__()
        self.child = child


_unsafe_pickle_loaded = False


def _unsafe_pickle_marker():
    global _unsafe_pickle_loaded
    _unsafe_pickle_loaded = True
    return {}


class _UnsafePicklePayload:
    def __reduce__(self):
        return (_unsafe_pickle_marker, ())


def _write_checkpoint(path, args, metadata, model_payload=None):
    model_buffer = io.BytesIO()
    torch.save({} if model_payload is None else model_payload, model_buffer)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("args.json", json.dumps(args))
        archive.writestr("metadata.json", json.dumps(metadata))
        archive.writestr("model.pt", model_buffer.getvalue())


@pytest.mark.parametrize("LoadModel", [MockModel, NewMockModel])
def test_from_checkpoint_custom(device, LoadModel):
    """Test checkpointing custom physicsnemo module"""
    torch.manual_seed(0)

    # Construct Mock Model and save it
    mock_model = MockModel().to(device)
    mock_model.save("checkpoint.mdlus")

    # Load from checkpoint using class
    LoadModel.from_checkpoint(
        "checkpoint.mdlus",
        allow_unsafe_imports=LoadModel is not MockModel,
    )
    # Delete checkpoint file (it should exist!)
    Path("checkpoint.mdlus").unlink(missing_ok=False)


def test_from_checkpoint_override(device):
    """Test checkpointing custom physicsnemo module with override"""
    torch.manual_seed(0)

    # Model with no overrides, loading without overrides
    mock_model = MockModelNoOverride(1, 2, 3).to(device)
    mock_model.save("checkpoint.mdlus")
    mock_model = MockModelWithOverride.from_checkpoint(
        "checkpoint.mdlus", allow_unsafe_imports=True
    )

    # Model with no overrides, loading with overrides (should fail)
    with pytest.raises(ValueError):
        mock_model = MockModelWithOverride.from_checkpoint(
            "checkpoint.mdlus",
            override_args={"value2": 20},
            allow_unsafe_imports=True,
        )

    Path("checkpoint.mdlus").unlink(missing_ok=False)

    # Model with overrides, loading without overrides
    mock_model = MockModelWithOverride(1, 2, 3).to(device)
    mock_model.save("checkpoint.mdlus")
    mock_model = MockModelWithOverride.from_checkpoint("checkpoint.mdlus")

    # Model with overrides, loading with allowed overrides (``value2`` value
    # should be erased by the state-dict, ``x`` should be overriden and kept)
    mock_model = MockModelWithOverride.from_checkpoint(
        "checkpoint.mdlus", override_args={"value2": 20, "x": 30}
    )
    assert torch.equal(mock_model.w2, torch.tensor(2, dtype=torch.float32))
    assert mock_model.x == 30

    # Model with overrides, loading with disallowed overrides (should fail)
    with pytest.raises(ValueError):
        mock_model = MockModelWithOverride.from_checkpoint(
            "checkpoint.mdlus", override_args={"value1": 10, "value2": 20}
        )

    # Model with overrides, loading with unexpected overrides (should fail)
    with pytest.raises(ValueError):
        mock_model = MockModelWithOverride.from_checkpoint(
            "checkpoint.mdlus", override_args={"value3": 4}
        )

    Path("checkpoint.mdlus").unlink(missing_ok=False)


@pytest.mark.parametrize("invalid_version", [[1, 2, 3], {"major": 1}])
def test_from_checkpoint_rejects_non_string_version(tmp_path, invalid_version):
    checkpoint = tmp_path / "invalid-version.mdlus"
    args = {
        "__name__": "MockModel",
        "__module__": __name__,
        "__args__": {"layer_size": 16},
    }
    _write_checkpoint(
        checkpoint,
        args,
        {"mdlus_file_version": invalid_version},
    )

    with pytest.raises(IOError, match="Invalid checkpoint version type"):
        MockModel.from_checkpoint(checkpoint)


def test_from_checkpoint_rejects_cyclic_nested_modules(tmp_path):
    checkpoint = tmp_path / "cyclic-modules.mdlus"
    nested_prefix = "__physicsnemo.Module__.child"
    nested_args = {
        "__name__": "NestedMockModel",
        "__module__": __name__,
        "__args__": {"child": nested_prefix},
    }
    args = {
        "__name__": "NestedMockModel",
        "__module__": __name__,
        "__args__": {"child": nested_prefix},
        nested_prefix: nested_args,
    }
    ModelRegistry().register(NestedMockModel)
    _write_checkpoint(checkpoint, args, {})

    with pytest.raises(IOError, match="cyclic module reference"):
        NestedMockModel.from_checkpoint(checkpoint)


def test_instantiate_rejects_unregistered_import(monkeypatch):
    def unexpected_import(_module_name):
        pytest.fail("untrusted module was imported")

    monkeypatch.setattr(
        "physicsnemo.core.module.importlib.import_module", unexpected_import
    )
    args = {
        "__name__": "run",
        "__module__": "subprocess",
        "__args__": {"args": ["true"]},
    }

    with pytest.raises(ValueError, match="neither registered"):
        physicsnemo.core.Module.instantiate(args)


def test_module_load_uses_restricted_unpickler_by_default(tmp_path):
    global _unsafe_pickle_loaded
    _unsafe_pickle_loaded = False
    checkpoint = tmp_path / "unsafe-pickle.mdlus"
    args = {
        "__name__": "MockModel",
        "__module__": __name__,
        "__args__": {"layer_size": 16},
    }
    _write_checkpoint(
        checkpoint,
        args,
        {"mdlus_file_version": "0.1.0"},
        model_payload=_UnsafePicklePayload(),
    )

    model = MockModel()
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        model.load(checkpoint, strict=False)
    assert not _unsafe_pickle_loaded

    model.load(checkpoint, strict=False, allow_unsafe_pickle=True)
    assert _unsafe_pickle_loaded

    _unsafe_pickle_loaded = False
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        MockModel.from_checkpoint(checkpoint, strict=False)
    assert not _unsafe_pickle_loaded

    MockModel.from_checkpoint(
        checkpoint,
        strict=False,
        allow_unsafe_pickle=True,
    )
    assert _unsafe_pickle_loaded


def test_checkpoint_archive_members_stay_within_destination(tmp_path):
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        for name in ("model.pt", "../outside.pt", "/absolute.pt"):
            content = b"checkpoint"
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))

        link = tarfile.TarInfo("linked-model.pt")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside.pt"
        archive.addfile(link)

    archive_buffer.seek(0)
    with tarfile.open(fileobj=archive_buffer, mode="r") as archive:
        safe_names = [
            member.name
            for member in physicsnemo.core.Module._safe_members(archive, tmp_path)
        ]

    assert safe_names == ["model.pt"]
