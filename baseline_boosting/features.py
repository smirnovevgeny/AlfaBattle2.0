import pandas as pd

# константны для признаков танзакции

CAT_COLUMNS = ['currency', 'operation_kind', 'card_type',
               'operation_type', 'operation_type_group', 'ecommerce_flag',
               'payment_system', 'income_flag', 'mcc', 'country', 'city',
               'mcc_category', 'day_of_week', 'hour','weekofyear']

NUMERIC_COLUMNS = ['days_before', 'hour_diff']

REAL_COLUMNS = ['amnt']


def __amnt_pivot_table_by_column_as_frame(frame, column, agg_funcs=None) -> pd.DataFrame:
    """
    Строит pivot table для между колонкой `amnt`  и column на основе переданных aggregations_on
    :param frame: pd.DataFrame транзакций
    :param column: название колонки, на основе `amnt`  и column будет построен pivot_table
    :param agg_funcs: список из функций, которые нужно применить, по умолчанию ['mean', 'count']
    :return: pd.DataFrame
    """
    if agg_funcs is None:
        agg_funcs = ['mean', 'count']
    aggs = pd.pivot_table(frame, values='amnt',
                          index=['app_id'], columns=[column],
                          aggfunc={'amnt': agg_funcs},
                          fill_value=0.0)
    aggs.columns = [f'{col[0]}_{column}_{col[1]}' for col in aggs.columns.values]
    return aggs


def extract_basic_aggregations(transactions_frame: pd.DataFrame, cat_columns=None, agg_funcs=None) -> pd.DataFrame:
    """
    :param transactions_frame: pd.DataFrame с транзакциями
    :param cat_columns: список категориальных переменных, для которых будут построены агрегаты по `amnt`
    :param agg_funcs: список функций, который нужно применить для подсчета агрегатов, по умолчанию
    ['sum', 'mean', 'count']
    :return: pd.DataFrame с извлеченными признаками
    """
    if not cat_columns:
        cat_columns = CAT_COLUMNS

    pivot_tables = []
    for col in cat_columns:
        pivot_tables.append(__amnt_pivot_table_by_column_as_frame(transactions_frame, column=col,
                                                                  agg_funcs=agg_funcs))
    pivot_tables = pd.concat(pivot_tables, axis=1)

    aggs = {
        # посчитаем статистики для транзакций
        'amnt': ['mean', 'median', 'sum', 'std'],
        # посчитаем разумные агрегаты для разницы в часах между транзакциями
        'hour_diff': ['max', 'mean', 'median', 'var', 'std'],
        # добавим самую раннюю/позднюю и среднюю дату транзакции до подачи заявки на кредит
        'days_before': ['min', 'max', 'median']}

    numeric_stats = transactions_frame.groupby(['app_id']).agg(aggs)

    # дадим разумные имена новым колонкам; может не работать в python 3.5, так как порядок ключей в словаре не
    # гарантирован
    numeric_stats.columns = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]

    return pd.concat([pivot_tables, numeric_stats], axis=1).reset_index()
