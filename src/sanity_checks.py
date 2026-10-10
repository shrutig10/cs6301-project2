import math

import torch
from torch import nn

from baseline import BaselineCNN


def main():
    torch.manual_seed(0)

    loss_function = nn.CrossEntropyLoss()

    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    uniform_logits = torch.zeros(4, 10)

    uniform_loss = loss_function(uniform_logits, labels)
    expected_loss = math.log(10)

    print("Uniform prediction loss:", uniform_loss.item())
    print("Expected loss:", expected_loss)

    assert math.isclose(
        uniform_loss.item(),
        expected_loss,
        abs_tol=1e-6,
    )

    model = BaselineCNN()
    images = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        initial_logits = model(images)
        initial_loss = loss_function(initial_logits, labels)

    print("Untrained model loss on artificial inputs:", initial_loss.item())

    assert initial_logits.shape == (4, 10)
    assert torch.isfinite(initial_loss).item()

    print("Initial-loss checks passed.")

    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    weights_before = model.features[0].weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    loss = loss_function(logits, labels)

    assert torch.isfinite(loss).item()

    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, (
                f"Missing gradient: {name}"
            )
            assert torch.isfinite(parameter.grad).all().item(), (
                f"Non-finite gradient: {name}"
            )

    print("All trainable parameters have finite gradients.")

    optimizer.step()

    weights_after = model.features[0].weight.detach()

    largest_change = (
        weights_after - weights_before
    ).abs().max().item()

    print("Loss before the update:", loss.item())
    print("Largest first-layer weight change:", largest_change)

    assert largest_change > 0

    print("One training-step check passed.")


if __name__ == "__main__":
    main()