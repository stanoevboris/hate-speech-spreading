import numpy as np
import torch as th
import torch.nn as nn

BEST_MODEL_EN = 'best_model_state_en.bin'
BEST_MODEL_ES = 'best_model_state_es.bin'


def train(language,
          df_train,
          df_val,
          train_data_loader,
          val_data_loader,
          model,
          epochs,
          loss_fn,
          optimizer,
          scheduler):

    if language == 'en':
        best_model_path = BEST_MODEL_EN
    else:
        best_model_path = BEST_MODEL_ES

    best_accuracy = 0
    history = dict()
    history['train_acc'] = list()
    history['train_loss'] = list()
    history['val_acc'] = list()
    history['val_loss'] = list()

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            scheduler,
            len(df_train)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            len(df_val)
        )
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
            th.save(model.state_dict(), best_model_path)
            best_accuracy = val_acc


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
