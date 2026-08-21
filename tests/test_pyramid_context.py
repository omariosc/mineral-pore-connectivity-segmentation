import json

import pytest


torch = pytest.importorskip("torch")

from src.models.pyramid_context import DEFAULT_POOL_GRIDS, PyramidContextBlock


EXPECTED_DEFAULT_PARAMETER_COUNT = 131_840


def test_default_configuration_and_parameter_count_are_locked_and_serializable():
    module = PyramidContextBlock()
    resolved = module.resolved_config()

    assert DEFAULT_POOL_GRIDS == (1, 2, 4, 8)
    assert resolved["schema_version"] == 1
    assert resolved["implementation_class"] == "PyramidContextBlock"
    assert resolved["in_channels"] == 256
    assert resolved["out_channels"] == 256
    assert resolved["pool_grids"] == [1, 2, 4, 8]
    assert resolved["branch_channels"] == 32
    assert resolved["number_of_branches"] == 4
    assert resolved["concat_channels"] == 384
    assert resolved["normalization"] == {"type": "GroupNorm", "groups": 8}
    assert resolved["activation"] == "ReLU"
    assert resolved["dropout"] == {
        "type": "Dropout2d",
        "probability": 0.1,
    }
    assert resolved["resize"] == {
        "mode": "bilinear",
        "align_corners": False,
        "target": "input_spatial_shape",
    }
    assert resolved["residual"] is True
    assert resolved["convolution_bias"] is False
    assert resolved["weight_initialization"] == "kaiming_normal_fan_out_relu"
    assert [branch.pool.output_size for branch in module.branches] == [
        (1, 1),
        (2, 2),
        (4, 4),
        (8, 8),
    ]
    group_norm_layers = [
        layer for layer in module.modules() if isinstance(layer, torch.nn.GroupNorm)
    ]
    assert len(group_norm_layers) == 5
    assert all(layer.num_groups == 8 for layer in group_norm_layers)
    assert module.parameter_count == EXPECTED_DEFAULT_PARAMETER_COUNT
    assert module.trainable_parameter_count == EXPECTED_DEFAULT_PARAMETER_COUNT
    assert resolved["parameter_count"] == EXPECTED_DEFAULT_PARAMETER_COUNT
    assert resolved["trainable_parameter_count"] == EXPECTED_DEFAULT_PARAMETER_COUNT
    json.dumps(resolved)


@pytest.mark.parametrize(
    ("image_side", "expected_bottleneck_side"),
    [(683, 42), (2048, 128)],
)
def test_preserves_bottleneck_shape_for_patch_and_full_tile_paths(
    image_side, expected_bottleneck_side
):
    bottleneck_side = image_side
    for _ in range(4):
        bottleneck_side //= 2
    assert bottleneck_side == expected_bottleneck_side

    torch.manual_seed(9)
    module = PyramidContextBlock().eval()
    x = torch.randn(1, 256, bottleneck_side, bottleneck_side)

    with torch.no_grad():
        output = module(x)

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert torch.isfinite(output).all()


def test_eval_is_deterministic_and_training_dropout_is_seed_reproducible():
    torch.manual_seed(17)
    module = PyramidContextBlock()
    x = torch.randn(1, 256, 11, 13)

    module.eval()
    with torch.no_grad():
        first_eval = module(x)
        second_eval = module(x)
    assert torch.equal(first_eval, second_eval)

    module.train()
    torch.manual_seed(23)
    first_train = module(x)
    torch.manual_seed(23)
    second_train = module(x)
    assert torch.equal(first_train, second_train)


def test_all_parameters_and_input_receive_finite_gradients():
    torch.manual_seed(31)
    module = PyramidContextBlock(dropout=0.0).eval()
    x = torch.randn(2, 256, 12, 15, requires_grad=True)

    loss = module(x).square().mean()
    loss.backward()

    assert loss.isfinite()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_group_norm_remains_valid_for_batch_one_and_global_grid():
    module = PyramidContextBlock().train()
    output = module(torch.randn(1, 256, 8, 8))

    assert output.shape == (1, 256, 8, 8)
    assert torch.isfinite(output).all()


def test_fused_context_is_added_through_an_exact_identity_residual():
    module = PyramidContextBlock(dropout=0.0).eval()
    with torch.no_grad():
        module.fuse[0].weight.zero_()
    x = torch.randn(1, 256, 9, 10)

    with torch.no_grad():
        output = module(x)

    assert torch.equal(output, x)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pool_grids": ()}, "pool_grids"),
        ({"pool_grids": (1, 2, 2, 8)}, "duplicates"),
        ({"pool_grids": (2, 1, 4, 8)}, "in increasing order"),
        ({"branch_channels": 30}, "divisible by norm_groups"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_invalid_configuration_fails_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PyramidContextBlock(**kwargs)


def test_invalid_runtime_input_fails_closed():
    module = PyramidContextBlock()
    with pytest.raises(ValueError, match="shape"):
        module(torch.randn(256, 8, 8))
    with pytest.raises(ValueError, match="expected 256 input channels"):
        module(torch.randn(1, 128, 8, 8))
