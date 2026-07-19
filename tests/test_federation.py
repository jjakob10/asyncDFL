import pytest
import torch

from app.federation import deserialize_state_dict, fedavg, serialize_state_dict


def test_fedavg_uses_sample_weights() -> None:
    first = {
        "weight": torch.tensor([1.0, 3.0]),
        "counter": torch.tensor(1, dtype=torch.int64),
    }
    second = {
        "weight": torch.tensor([5.0, 7.0]),
        "counter": torch.tensor(5, dtype=torch.int64),
    }

    result = fedavg([(first, 1), (second, 3)])

    assert torch.allclose(result["weight"], torch.tensor([4.0, 6.0]))
    assert result["counter"].item() == 4


def test_serialization_round_trip() -> None:
    state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}

    restored = deserialize_state_dict(serialize_state_dict(state), state)

    assert torch.equal(restored["weight"], state["weight"])


def test_deserialization_rejects_incompatible_model() -> None:
    state = {"weight": torch.ones(2)}
    incompatible = {"weight": torch.ones(3)}

    with pytest.raises(ValueError, match="shape mismatch"):
        deserialize_state_dict(serialize_state_dict(incompatible), state)


def test_fedavg_rejects_missing_parameters() -> None:
    with pytest.raises(ValueError, match="parameter names"):
        fedavg([({"a": torch.ones(1)}, 1), ({"b": torch.ones(1)}, 1)])

