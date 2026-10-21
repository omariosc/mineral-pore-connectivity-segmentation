import numpy as np
import torch
from scripts.run_revision_validation import logits_for
from scripts.evaluate_confirmatory_checkpoint import prepare_locked_model_input


def test_cpu_reanalysis_uses_float32_and_recomputes_input_gate():
    torch.set_num_threads(2)
    model = torch.nn.Conv2d(2, 2, 1, bias=False).eval()
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 0, 0, 0] = 1
        model.weight[1, 1, 0, 0] = 1
    image = np.array([[0, 99, 100, 255]], dtype=np.uint8)
    logits = logits_for(
        torch, model, image, "C2-P", prepare_locked_model_input, 100, device="cpu"
    )
    assert logits.dtype == np.float32
    assert logits[0, 0, 0] == -1 and logits[0, 0, -1] == 1
    assert logits[1].tolist() == [[1.0, 1.0, 0.0, 0.0]]
    changed = logits_for(
        torch, model, image, "C2-P", prepare_locked_model_input, 90, device="cpu"
    )
    assert changed[1].tolist() == [[1.0, 0.0, 0.0, 0.0]]
