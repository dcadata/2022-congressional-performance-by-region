import json
import re

import pandas as pd
import requests
import seaborn as sns
from statsmodels.formula.api import ols


def _read_midterm_election_results(office: str, refresh: bool = False) -> pd.DataFrame:
    """
    Unofficial election results via Politico
    """
    office_suffix, cols_to_drop = dict(
        congress=('cd', ['votePct', 'winner', 'calledAt', 'originalVoteCount', 'originalVotePct', 'runoff']),
        governor=('gov', ['votePct', 'winner', 'calledAt', 'runoff']),
    )[office]
    data_fp = f'data/2022-ge__collection__{office_suffix}.json'

    if refresh:
        response = requests.get(
            f'https://www.politico.com/election-data/2022-ge__collection__{office_suffix}/data.json')
        with open(data_fp, 'wb') as file:
            file.write(response.content)

    results = []
    for contest in json.load(open(data_fp))['contests']:
        if contest['progress']['pct'] >= 0.95:
            results.append(pd.DataFrame(contest['results']).assign(pecan=contest['id']))
    election_results = pd.concat(results).drop(columns=cols_to_drop)
    return election_results


def _read_candidates() -> pd.DataFrame:
    """
    Via Politico's flatpack:
    https://www.politico.com/election-data/2022-ge__flatpack-configs__overall/house.flatpack.json
    Candidate data:
    https://www.politico.com/election-data/2022-ge__metadata/all-candidates.meta.csv
    """
    candidates = pd.read_csv('data/all-candidates.meta.csv', usecols=[
        'pecan', 'id', 'fullName', 'party', 'isIncumbent'], dtype=str)
    return candidates


def _read_metadata() -> pd.DataFrame:
    """
    All races metadata via Politico:
    https://www.politico.com/election-data/2022-ge__metadata/all-races.meta.csv
    """
    metadata = pd.read_csv('data/all-races.meta.csv', usecols=['id', 'isUncontested', 'rating'])
    metadata.isUncontested = metadata.isUncontested.fillna(0).apply(bool)
    metadata['isCompetitive'] = metadata.rating.fillna('').str.startswith(('toss-up', 'lean', 'likely'))
    metadata = metadata.drop(columns='rating')
    return metadata


def _read_state_fips() -> pd.DataFrame:
    fips = pd.read_csv('G:/election_data/reference/state_fips_codes.csv', dtype=str).rename(columns=dict(
        stateName='name'))
    states = pd.read_csv('G:/election_data/reference/states_table.csv', usecols=[
        'name', 'iso'], dtype=str).drop_duplicates()
    states.name = states.name.str.upper()
    fips = fips.merge(states, on='name').drop(columns='name').rename(columns=dict(iso='state'))
    return fips


def _read_presidential_results_by_congressional_district() -> pd.DataFrame:
    """
    Daily Kos Elections' 2020 presidential results by congressional district:
    https://www.dailykos.com/stories/2021/9/29/2055001/-Daily-Kos-Elections-2020-presidential-results-by-congressional-district-for-new-and-old-districts
    """
    presidential_by_cd = pd.read_csv('data/DK_presidential_results_by_congressional_district.csv', usecols=[
        'District', 'Biden', 'Trump'], dtype=str).rename(columns=dict(District='district'))

    for col in ('Biden', 'Trump'):
        presidential_by_cd[col] = presidential_by_cd[col].apply(lambda x: x.replace(',', '')).apply(int)

    presidential_by_cd[['state', 'district']] = presidential_by_cd.district.str.split('-', expand=True)
    presidential_by_cd.loc[presidential_by_cd.district == 'AL', 'district'] = '00'
    return presidential_by_cd


def _build_presidential_results_by_congressional_district() -> pd.DataFrame:
    pres = _read_presidential_results_by_congressional_district()

    _separate_candidates = lambda candidate, p: pres[['state', 'district', candidate]].rename(columns={
        candidate: 'votePctPres'}).assign(party=p)
    pres = pd.concat((_separate_candidates('Biden', 'D'), _separate_candidates('Trump', 'R')))

    pres = pres.merge(pres.groupby(['state', 'district'], as_index=False).votePctPres.sum(), on=[
        'state', 'district'], suffixes=('', 'Total'))
    pres.votePctPres = pres.votePctPres / pres.votePctPresTotal
    pres = pres.rename(columns=dict(votePctPresTotal='turnoutPres'))
    return pres


def _build_presidential_results_by_state() -> pd.DataFrame:
    pres = _read_presidential_results_by_congressional_district()
    pres = pres.groupby('state', as_index=False)[['Biden', 'Trump']].sum()

    _separate_candidates = lambda candidate, p: pres[['state', candidate]].rename(columns={
        candidate: 'votePctPres'}).assign(party=p)
    pres = pd.concat((_separate_candidates('Biden', 'D'), _separate_candidates('Trump', 'R')))

    pres = pres.merge(pres.groupby('state', as_index=False).votePctPres.sum(), on='state', suffixes=('', 'Total'))
    pres.votePctPres = pres.votePctPres / pres.votePctPresTotal
    pres = pres.rename(columns=dict(votePctPresTotal='turnoutPres'))
    return pres


def _build_midterm_election_results(office: str) -> pd.DataFrame:
    results = _read_midterm_election_results(office=office)
    turnout = results.groupby('pecan', as_index=False).voteCount.sum().rename(columns=dict(voteCount='turnout22'))
    results = results.merge(_read_metadata(), left_on='pecan', right_on='id', suffixes=('', 'Pecan')).drop(
        columns='idPecan').merge(_read_candidates(), on=['pecan', 'id'])

    results.isIncumbent = results.isIncumbent.fillna(0).apply(bool)
    results.party = results.party.apply(dict(dem='D', gop='R').get)
    results = results.dropna(subset=['party'])

    results = results.merge(results.groupby('pecan', as_index=False).voteCount.sum(), on='pecan', suffixes=(
        '', 'Total'))
    results['votePct'] = results.voteCount / results.voteCountTotal
    results = results.drop(columns=['voteCount', 'voteCountTotal']).merge(turnout, on='pecan')
    return results


def build_congressional() -> pd.DataFrame:
    results = _build_midterm_election_results('congress')

    stateFips_district = results.pecan.apply(lambda x: re.search(
        '2022-11-08/([0-9]{2})/cd([0-9]{2})(Special)?/general', x).groups()[:2])
    results['stateFips'] = stateFips_district.apply(lambda x: x[0])
    results['district'] = stateFips_district.apply(lambda x: x[1])
    results = results.merge(_read_state_fips(), on='stateFips').merge(
        _build_presidential_results_by_congressional_district(), on=['state', 'district', 'party'])
    return results


def build_gubernatorial() -> pd.DataFrame:
    results = _build_midterm_election_results('governor')

    stateFips = results.pecan.apply(lambda x: re.search('2022-11-08/([0-9]{2})/gov/general', x).groups())
    results['stateFips'] = stateFips.apply(lambda x: x[0])
    results = results.merge(_read_state_fips(), on='stateFips').merge(
        _build_presidential_results_by_state(), on=['state', 'party'])
    return results


def read_and_merge_all() -> pd.DataFrame:
    """
    Still used in notebook...
    """
    results = build_congressional()
    results['turnoutRel'] = results.turnout22 / results.turnoutPres
    return results


def filter_by_region(results: pd.DataFrame, regions: iter) -> pd.DataFrame:
    df = results[~results.isUncontested & (results.party == 'D')].copy()
    df['Location'] = df.state.apply(lambda x: ('In_' + '_'.join(regions)) if x in regions else 'Elsewhere')
    return df


def make_plot_and_fit_model(results: pd.DataFrame, formula: str, competitive_only: bool = False) -> str:
    df = (results[results.isCompetitive] if competitive_only else results).copy()

    hue_order = list(df.Location.unique())
    hue_order.remove('Elsewhere')
    hue_order.append('Elsewhere')

    title = f'Democratic Performance by Region{" - in Competitive CDs" if competitive_only else ""}'
    plt = sns.lmplot(data=df, x='votePctPres', y='votePct', hue='Location', palette=(
        'purple', 'grey'), hue_order=hue_order, markers=('o', '.'), facet_kws=dict(legend_out=False))
    plt.set_xlabels(label='2020 Biden Share')
    plt.set_ylabels(label='2022 Congressional (D) Share')
    plt.fig.suptitle(title)
    plt.fig.set_size_inches(8, 6)

    model = ols(formula, data=df)
    fitted = model.fit()
    summary = fitted.summary()
    return '\n\n'.join((title, summary.as_text()))
