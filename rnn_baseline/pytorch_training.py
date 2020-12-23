import torch
import pandas as pd
import torch.nn as nn
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score

from rnn_baseline.data_generators import batches_generator


def train_epoch(model, optimizer, dataset_train, batch_size=64, shuffle=True,
                print_loss_every_n_batches=500, device=None):
    """
    делает одну эпоху обучения модели, логирует
    :param model: nn.Module модель
    :param optimizer: nn.optim оптимизатор
    :param dataset_train: путь до директории с последовательностями
    :param batch_size: размерм батча
    :param shuffle: флаг, если True, то перемешивает данные
    :param print_loss_every_n_batches: число батчей после которых логируется лосс на этих батчах
    :param device: device, на который будут положены данные внутри батча
    :return: None
    """
    train_generator = batches_generator(dataset_train, batch_size=batch_size, shuffle=shuffle,
                                        device=device, is_train=True, output_format='torch')
    loss_function = nn.BCEWithLogitsLoss()

    num_batches = 1
    running_loss = 0.0

    model.train()

    for batch in tqdm(train_generator, desc='Training'):

        output = torch.flatten(model(batch['transactions_features'], batch['product']))

        batch_loss = loss_function(output, batch['label'].float())

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += batch_loss

        if num_batches % print_loss_every_n_batches == 0:
            print(f'Training loss after {num_batches} batches: {running_loss / num_batches}', end='\r')
        
        num_batches += 1
    
    print(f'Training loss after epoch: {running_loss / num_batches}', end='\r')
    

def eval_model(model, dataset_val, batch_size=32, device=None) -> float:
    """
    функция для оценки качества модели на отложенной выборке, возвращает roc-auc на валидационной
    выборке
    :param model: nn.Module модель
    :param dataset_val: путь до директории с последовательностями
    :param batch_size: размер батча
    :param device: device, на который будут положены данные внутри батча
    :return: val roc-auc score
    """
    preds = []
    targets = []
    val_generator = batches_generator(dataset_val, batch_size=batch_size, shuffle=False,
                                      device=device, is_train=True, output_format='torch')
    model.eval()

    for batch in tqdm(val_generator, desc='Evaluating model'):
        targets.extend(batch['label'].detach().cpu().numpy().flatten())
        output = model(batch['transactions_features'], batch['product'])
        preds.extend(output.detach().cpu().numpy().flatten())

    return roc_auc_score(targets, preds)


def inference(model, dataset_test, batch_size=32, device=None) -> pd.DataFrame:
    """
    функция, которая делает предикты на новых данных, возвращает pd.DataFrame из двух колонок:
    (app_id, score)
    :param model: nn.Module модель
    :param dataset_test: путь до директории с последовательностями
    :param batch_size: размер батча
    :param device: device, на который будут положены данные внутри батча
    :return: pd.DataFrame из двух колонок: (app_id, score)
    """
    model.eval()
    preds = []
    app_ids = []
    test_generator = batches_generator(dataset_test, batch_size=batch_size, shuffle=False,
                                       verbose=False, device=device, is_train=False,
                                       output_format='torch')

    for batch in tqdm(test_generator, desc='Test time predictions'):
        app_ids.extend(batch['app_id'])
        output = model(batch['transactions_features'], batch['product'])
        preds.extend(output.detach().cpu().numpy().flatten())
        
    return pd.DataFrame({
        'app_id': app_ids,
        'score': preds
    })
