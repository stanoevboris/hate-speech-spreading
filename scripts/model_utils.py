import numpy as np
import torch as th
import torch.nn as nn


def train_epoch(model,
                data_loader,
                loss_fn,
                optimizer,
                scheduler,
                n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"]
        attention_mask = d["attention_mask"]
        targets = d["targets"]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = th.max(th.round(outputs), dim=1)
        loss = loss_fn(outputs.flatten(), targets)
        correct_predictions += th.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with th.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            targets = d["targets"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = th.max(th.round(outputs), dim=1)
            loss = loss_fn(outputs.flatten(), targets)
            correct_predictions += th.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with th.no_grad():
        for d in data_loader:
            texts = d["user_tweets"]
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            targets = d["targets"]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = th.max(th.round(outputs), dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)

    predictions = th.stack(predictions).cpu()
    prediction_probs = th.stack(prediction_probs).cpu()
    real_values = th.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values
