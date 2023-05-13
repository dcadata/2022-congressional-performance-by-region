import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols

from task import read_and_merge_all


def view_differential_turnout_summary() -> None:
    results = read_and_merge_all()
    df = results[(results.party == 'R') & ~results.isUncontested]

    plot = sns.lmplot(data=df, x='votePctPres', y='turnoutRel', hue='isCompetitive', palette=(
        'purple', 'green'), hue_order=(False, True), markers='.')
    plot.fig.set_size_inches(8, 6)
    plot.fig.tight_layout()
    plot.fig.suptitle('2022 Midterms - Turnout by Congressional District')
    plot.set_xlabels("Trump '20 Vote Share")
    plot.set_ylabels("'22 Turnout as % of '20 Turnout")

    formula = 'turnoutRel ~ votePctPres + isCompetitive'
    model = ols(formula, data=df)
    fitted = model.fit()
    summary = fitted.summary()
    print(summary.as_text())


def _filter_by_location(results: pd.DataFrame, states: iter) -> pd.DataFrame:
    df = results[~results.isUncontested].copy()
    df['Location'] = df.state.apply(lambda x: ('In_' + '_'.join(states)) if x in states else 'Elsewhere')
    return df


def view_differential_turnout_summary_by_location(*args) -> None:
    results = _filter_by_location(read_and_merge_all(), *args)

    hue_order = list(results.Location.unique())
    hue_order.remove('Elsewhere')
    hue_order.append('Elsewhere')

    plot = sns.lmplot(
        data=results[results.party == 'R'], x='votePctPres', y='turnoutRel', hue='Location',
        palette=('orange', 'grey'), hue_order=hue_order, markers='.')
    plot.fig.set_size_inches(8, 6)
    plot.fig.tight_layout()
    plot.fig.suptitle('2022 Midterms - Turnout by Congressional District')
    plot.set_xlabels("Trump '20 Vote Share")
    plot.set_ylabels("'22 Turnout as % of '20 Turnout")
