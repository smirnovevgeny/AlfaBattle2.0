import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score

from rnn_baseline.data_generators import batches_generator


def train_epoch(model, dataset_train, batch_size=64, shuffle=True, cur_epoch=0,
                steps_per_epoch=5000, callbacks=None):
    """
    функция обучения модели одну эпоху
    :param model: tf.keras.Model
    :param dataset_train: путь до директории с последовательностями
    :param batch_size: размер батча
    :param shuffle: флаг, если True, то перемешивает данные
    :param cur_epoch:
    :param steps_per_epoch:
    :param callbacks: cписок из tf.keras.callbacks или None
    :return: None
    """
    train_generator = batches_generator(dataset_train, batch_size=batch_size, shuffle=shuffle,
                                        output_format='tf', is_train=True)
    model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=cur_epoch + 1,
              initial_epoch=cur_epoch, callbacks=callbacks)


def eval_model(model, dataset_val, batch_size=32) -> float:
    """
    функция для оценки качества модели на отложенной выборке, возвращает roc-auc на валидационной
    выборке
    :param model: tf.keras.Model
    :param dataset_val: путь до директории с последовательностями
    :param batch_size: размер батча
    :return: val roc-auc score
    """
    val_generator = batches_generator(dataset_val, batch_size=batch_size, shuffle=False,
                                      output_format='tf', is_train=True)
    preds = model.predict(val_generator).flatten()
    val_generator = batches_generator(dataset_val, batch_size=batch_size, shuffle=False,
                                      output_format='tf', is_train=True)
    targets = []
    for _, y in val_generator:
        targets.extend(y)

    return roc_auc_score(targets, preds)


def inference(model, dataset_test, batch_size=32) -> pd.DataFrame:
    """
    функция, которая делает предикты на новых данных, возвращает pd.DataFrame из двух колонок:
    (app_id, score)
    :param model: tf.keras.Model
    :param dataset_test: путь до директории с последовательностями
    :param batch_size: размер батча
    :return: pd.DataFrame из двух колонок: (app_id, score)
    """
    app_ids = []
    test_generator = batches_generator(dataset_test, batch_size=batch_size, shuffle=False,
                                       is_train=False, output_format='tf')

    preds = model.predict(test_generator).flatten()
    
    app_ids = []
    test_generator = batches_generator(dataset_test, batch_size=batch_size, shuffle=False, 
                                       is_train=False, output_format='tf')
    for _, y in test_generator:
        app_ids.extend(y)
        
    return pd.DataFrame({
        'app_id': app_ids,
        'score': preds
    })
