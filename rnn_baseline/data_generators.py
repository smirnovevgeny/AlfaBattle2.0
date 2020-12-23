import numpy as np
import pickle
import torch

transaction_features = ['currency', 'operation_kind', 'card_type', 'operation_type',
                        'operation_type_group', 'ecommerce_flag', 'payment_system',
                        'income_flag', 'mcc', 'country', 'city', 'mcc_category',
                        'day_of_week', 'hour', 'weekofyear', 'amnt', 'days_before', 'hour_diff']


def batches_generator(list_of_paths, batch_size=32, shuffle=False, is_infinite=False,
                      verbose=False, device=None, output_format='torch', is_train=True):
    """
    функция для создания батчей на вход для нейронной сети для моделей на keras и pytorch.
    так же может использоваться как функция на стадии инференса
    :param list_of_paths: путь до директории с предобработанными последовательностями
    :param batch_size: размер батча
    :param shuffle: флаг, если True, то перемешивает list_of_paths и так же
    перемешивает последовательности внутри файла
    :param is_infinite: флаг, если True,  то создает бесконечный генератор батчей
    :param verbose: флаг, если True, то печатает текущий обрабатываемый файл
    :param device: device на который положить данные, если работа на торче
    :param output_format: допустимые варианты ['tf', 'torch']. Если 'torch', то возвращает словарь,
    где ключи - батчи из признаков, таргетов и app_id. Если 'tf', то возвращает картеж: лист input-ов
    для модели, и список таргетов.
    :param is_train: флаг, Если True, то для кераса вернет (X, y), где X - input-ы в модель, а y - таргеты, 
    если False, то в y будут app_id; для torch вернет словарь с ключами на device.
    :return: бачт из последовательностей и таргетов (или app_id)
    """
    while True:
        if shuffle:
            np.random.shuffle(list_of_paths)

        for path in list_of_paths:
            if verbose:
                print(f'reading {path}')

            with open(path, 'rb') as f:
                data = pickle.load(f)
            padded_sequences, targets, products = data['padded_sequences'], data['targets'], data[
                'products']
            app_ids = data['app_id']
            indices = np.arange(len(products))

            if shuffle:
                np.random.shuffle(indices)
                padded_sequences = padded_sequences[indices]
                targets = targets[indices]
                products = products[indices]
                app_ids = app_ids[indices]

            for idx in range(len(products)):
                bucket, product = padded_sequences[idx], products[idx]
                app_id = app_ids[idx]
                
                if is_train:
                    target = targets[idx]
                
                for jdx in range(0, len(bucket), batch_size):
                    batch_sequences = bucket[jdx: jdx + batch_size]
                    if is_train:
                        batch_targets = target[jdx: jdx + batch_size]
                    
                    batch_products = product[jdx: jdx + batch_size]
                    batch_app_ids = app_id[jdx: jdx + batch_size]
                    
                    if output_format == 'tf':
                        batch_sequences = [batch_sequences[:, i] for i in
                                           range(len(transaction_features))]
                        
                        # append product as input to tf model
                        batch_sequences.append(batch_products)
                        if is_train:
                            yield batch_sequences, batch_targets
                        else:
                             yield batch_sequences, batch_app_ids
                    else:
                        batch_sequences = [torch.LongTensor(batch_sequences[:, i]).to(device)
                                           for i in range(len(transaction_features))]
                        if is_train:
                            yield dict(transactions_features=batch_sequences,
                                       product=torch.LongTensor(batch_products).to(device),
                                       label=torch.LongTensor(batch_targets).to(device),
                                       app_id=batch_app_ids)
                        else:
                            yield dict(transactions_features=batch_sequences,
                                       product=torch.LongTensor(batch_products).to(device),
                                       app_id=batch_app_ids)
        if not is_infinite:
            break
