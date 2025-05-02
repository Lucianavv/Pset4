import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_metrics(df, column):
    promedio = df[column].mean()
    maximo = df[column].max()
    minimum = df[column].min()
    std = df[column].std()

    print(f"Promedio: {promedio}")
    print(f"Maximo: {maximo}")
    print(f"Minimo: {minimum}")
    print(f"Std: {std}")


def cohort_analysis(visits: pd.DataFrame) -> pd.DataFrame:

    first_visit = visits.groupby('uid')['start_ts'].min().dt.to_period('M')
    visits = visits.assign(cohort_month=visits['uid'].map(first_visit))

    #   Weeks since first visit
    visits['week_index'] = ((visits['start_ts'] -
                             visits.groupby('uid')['start_ts'].transform('min'))
                             .dt.days // 7)

    cohort_pivot = (visits
                    .groupby(['cohort_month', 'week_index'])['uid']
                    .nunique()
                    .unstack(fill_value=0))

    cohort_size = cohort_pivot.iloc[:, 0]
    retention_pct = cohort_pivot.divide(cohort_size, axis=0).round(3)

    cohort_pivot.retention_pct = retention_pct
    return cohort_pivot


def plot_retention(cohort_table: pd.DataFrame, pct: bool = True) -> None:
    data = (cohort_table.retention_pct
            if pct and hasattr(cohort_table, "retention_pct")
            else cohort_table)

    plt.imshow(data, cmap="Blues", aspect="auto")
    plt.colorbar(label="Retention (%)" if pct else "Users")
    plt.ylabel("Cohorte (mes de 1ª visita)")
    plt.xlabel("Semanas desde 1ª visita")
    plt.yticks(ticks=np.arange(len(data.index)),
               labels=data.index.astype(str))
    plt.xticks(ticks=np.arange(data.shape[1]),
               labels=data.columns)
    plt.tight_layout()

def days_to_first_purchase(visits: pd.DataFrame,
                           orders: pd.DataFrame) -> pd.Series:

    first_visit = visits.groupby('uid')['start_ts'].min()
    first_order = orders.groupby('uid')['buy_ts'].min()
    joined = (pd.concat([first_visit, first_order], axis=1, join='inner')
                .dropna())

    return (joined['buy_ts'] - joined['start_ts']).dt.days


def orders_over_time(orders: pd.DataFrame,
                     freq: str = 'D') -> pd.Series:

    return (orders
            .set_index('buy_ts')
            .resample(freq)['uid']
            .count())


def ltv_per_cohort(visits: pd.DataFrame,
                   orders: pd.DataFrame,
                   monetary_col: str = 'revenue') -> pd.Series:
    # Cohort id
    cohort_month = (visits.groupby('uid')['start_ts']
                          .min()
                          .dt.to_period('M'))
    orders = orders.assign(cohort_month=orders['uid'].map(cohort_month))

    revenue_per_cohort = orders.groupby('cohort_month')[monetary_col].sum()
    cohort_sizes = cohort_month.value_counts().sort_index()
    return (revenue_per_cohort / cohort_sizes).round(2)


def calculate_cac(visits: pd.DataFrame,
                  costs: pd.DataFrame,
                  uid_col: str = 'uid',
                  source_col: str = 'source_id',
                  cost_col: str = 'costs') -> pd.Series:

    customers_per_source = visits.groupby(source_col)[uid_col].nunique()
    total_costs = costs.groupby(source_col)[cost_col].sum()
    return (total_costs / customers_per_source).round(2)


def calculate_romi(visits: pd.DataFrame,
                   orders: pd.DataFrame,
                   costs: pd.DataFrame,
                   source_col: str = 'source_id',
                   revenue_col: str = 'revenue',
                   cost_col: str = 'costs') -> pd.Series:


    revenue_by_source = (visits[[source_col, 'uid']]
                         .merge(orders[['uid', revenue_col]], on='uid')
                         .groupby(source_col)[revenue_col]
                         .sum())

    cost_by_source = costs.groupby(source_col)[cost_col].sum()
    romi = (revenue_by_source - cost_by_source) / cost_by_source
    return romi.round(3)
