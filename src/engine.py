import torch


def train_one_epoch(model, loader, loss_function, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous batch.
        optimizer.zero_grad(set_to_none=True)

        # Predict and calculate the mean loss for this batch.
        logits = model(images)
        loss = loss_function(logits, labels)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss.")

        # Calculate gradients.
        loss.backward()

        # Verify gradients before updating the weights.
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None:
                    raise RuntimeError(f"Missing gradient: {name}")

                if not torch.isfinite(parameter.grad).all().item():
                    raise RuntimeError(f"Non-finite gradient: {name}")

        optimizer.step()

        # Accumulate metrics from this batch's predictions.
        batch_size = labels.shape[0]

        total_loss += loss.item() * batch_size

        predictions = logits.detach().argmax(dim=1)
        total_correct += (predictions == labels).sum().item()

        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Training loader contained no examples.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def evaluate(model, loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_function(logits, labels)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite evaluation loss.")

            batch_size = labels.shape[0]

            total_loss += loss.item() * batch_size

            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()

            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Evaluation loader contained no examples.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }