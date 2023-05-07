import datetime
import json
import re

import numpy as np
import pandas as pd
import requests
import seaborn as sns
from statsmodels.formula.api import ols


def _read_congressional_election_results(refresh: bool = False) -> pd.DataFrame:
    """
    Unofficial Congressional election results via Politico:
    https://www.politico.com/election-data/2022-ge__collection__cd/data.json
    """
    data_fp = 'data/congressional_election_results.json'

    if refresh:
        response = requests.get('https://www.politico.com/election-data/2022-ge__collection__cd/data.json')
        with open(data_fp, 'wb') as file:
            file.write(response.content)

    results = []
    for contest in json.load(open(data_fp))['contests']:
        if contest['progress']['pct'] >= 0.95:
            results.append(pd.DataFrame(contest['results']).assign(pecan=contest['id']))
    er = pd.concat(results).drop(columns=[
        'votePct', 'winner', 'calledAt', 'originalVoteCount', 'originalVotePct', 'runoff'])
    return er


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
    pres = pd.read_csv('data/DK_presidential_results_by_congressional_district.csv', usecols=[
        'District', 'Biden', 'Trump'], dtype=str).rename(columns={'District': 'district'})

    _separate_candidates = lambda candidate, p: pres[['district', f'{candidate}']].rename(columns={
        f'{candidate}': 'votePctPres'}).assign(party=p)
    pres = pd.concat((_separate_candidates('Biden', 'D'), _separate_candidates('Trump', 'R')))
    pres.votePctPres = pres.votePctPres.apply(lambda x: x.replace(',', '')).apply(int)
    pres = pres.merge(pres.groupby('district', as_index=False).votePctPres.sum(), on='district', suffixes=('', 'Total'))
    pres.votePctPres = pres.votePctPres / pres.votePctPresTotal
    pres = pres.rename(columns=dict(votePctPresTotal='turnoutPres'))

    pres[['state', 'district']] = pres.district.str.split('-', expand=True)
    pres.loc[pres.district == 'AL', 'district'] = '00'
    return pres


def read_and_merge_all() -> pd.DataFrame:
    results = _read_congressional_election_results()
    turnout = results.groupby('pecan', as_index=False).voteCount.sum().rename(columns=dict(voteCount='turnout22'))
    results = results.merge(_read_metadata(), left_on='pecan', right_on='id', suffixes=('', 'Pecan')).drop(
        columns='idPecan').merge(_read_candidates(), on=['pecan', 'id'])

    results.party = results.party.apply(dict(dem='D', gop='R').get)
    results = results.dropna(subset=['party'])

    results = results.merge(results.groupby('pecan', as_index=False).voteCount.sum(), on='pecan', suffixes=(
        '', 'Total'))
    results['votePct'] = results.voteCount / results.voteCountTotal
    results = results.drop(columns=['voteCount', 'voteCountTotal']).merge(turnout, on='pecan')

    results.isIncumbent = results.isIncumbent.fillna(0).apply(bool)

    stateFips_district = results.pecan.apply(lambda x: re.search(
        '2022-11-08/([0-9]{2})/cd([0-9]{2})(Special)?/general', x).groups()[:2])
    results['stateFips'] = stateFips_district.apply(lambda x: x[0])
    results['district'] = stateFips_district.apply(lambda x: x[1])

    results = results.merge(_read_state_fips(), on='stateFips').merge(
        _read_presidential_results_by_congressional_district(), on=['state', 'district', 'party'])

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
        'purple', 'grey'), hue_order=hue_order, markers=['o', '.'], facet_kws=dict(legend_out=False))
    plt.set_xlabels(label='2020 Biden Share')
    plt.set_ylabels(label='2022 Congressional (D) Share')
    plt.fig.suptitle(title)
    plt.fig.set_size_inches(8, 6)

    model = ols(formula, data=df)
    fitted = model.fit()
    summary = fitted.summary()
    return '\n\n'.join((title, summary.as_text()))


def fit_model_for_turnout() -> None:
    results = read_and_merge_all()
    df = results[(results.party == 'D') & ~results.isUncontested]

    plot = sns.lmplot(df, x='votePctPres', y='turnoutRel', hue='isCompetitive')
    plot.fig.set_size_inches(8, 6)
    plot.fig.suptitle("Biden '20 Vote Share vs. '22 Turnout, Among Contested CDs")
    plot.set_xlabels("Biden '20 Vote Share")
    plot.set_ylabels("'22 Turnout as % of '20 Turnout")

    formula = 'turnoutRel ~ votePctPres + isCompetitive'
    model = ols(formula, data=df)
    fitted = model.fit()
    summary = fitted.summary()
    print(summary.as_text())


def _get_fec_metadata() -> None:
    """
    FEC data description:
    https://www.fec.gov/campaign-finance-data/current-campaigns-house-and-senate-file-description/
    """
    metadata = pd.read_html(
        'https://www.fec.gov/campaign-finance-data/current-campaigns-house-and-senate-file-description/')[0]
    columns = list(metadata.iloc[0])
    metadata = metadata.iloc[1:]
    metadata.columns = columns
    with open('data/fec-data-columns.json', 'w') as file:
        json.dump(metadata['Column name'].to_list(), file, indent=2)


def _read_fec_data() -> pd.DataFrame:
    """
    FEC bulk data:
    https://www.fec.gov/data/browse-data/?tab=bulk-data
    Under section House/Senate current campaigns
    """
    # read data
    fec_data = pd.read_csv('data/fec-data.zip', sep='|', names=json.load(open('data/fec-data-columns.json')))[[
        'CAND_ID', 'CAND_OFFICE_ST', 'CAND_OFFICE_DISTRICT', 'CAND_PTY_AFFILIATION', 'TTL_RECEIPTS', 'TTL_DISB',
        'CVG_END_DT',
    ]]
    # filter on House
    fec_data = fec_data[fec_data.CAND_ID.str.startswith('H')].copy()
    # keep only latest filing per candidate
    fec_data.CVG_END_DT = fec_data.CVG_END_DT.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))
    fec_data = fec_data.sort_values('CVG_END_DT').drop_duplicates(subset=['CAND_ID'], keep='last')
    # convert district to match election results format
    fec_data.CAND_OFFICE_DISTRICT = fec_data.CAND_OFFICE_DISTRICT.apply(lambda x: str(int(x)).zfill(2))
    # drop minor candidates
    fec_data = fec_data.sort_values('TTL_RECEIPTS').drop_duplicates(subset=[
        'CAND_OFFICE_ST', 'CAND_OFFICE_DISTRICT', 'CAND_PTY_AFFILIATION'], keep='last')
    fec_data['party'] = fec_data.CAND_PTY_AFFILIATION.apply(dict(DEM='D', DFL='D', REP='R').get)
    fec_data = fec_data.dropna(subset=['party'])
    # drop unnecessary columns and rename remaining columns
    fec_data = fec_data.drop(columns=['CAND_ID', 'CVG_END_DT', 'CAND_PTY_AFFILIATION']).rename(columns=dict(
        CAND_OFFICE_ST='state', CAND_OFFICE_DISTRICT='district', TTL_RECEIPTS='receipts', TTL_DISB='disb'))
    return fec_data


def _add_gender_predictions_based_on_first_name(results: pd.DataFrame) -> pd.DataFrame:
    results['firstName'] = results.fullName.apply(lambda x: ''.join(re.findall('[A-Za-z]', x.split(None, 1)[0])))
    reference = pd.read_csv('G:/GitHub/name-finder/gender_prediction_reference.csv', dtype=str)
    results = results.merge(reference, left_on='firstName', right_on='name').drop(columns='name')
    results = results[results.gender_prediction.isin(('f', 'm'))].drop(columns='firstName')
    return results


def preprocess_data_with_gender() -> pd.DataFrame:
    # combine election results and FEC data
    results = read_and_merge_all().merge(_read_fec_data(), on=['state', 'district', 'party'])
    # add gender predictions
    results = _add_gender_predictions_based_on_first_name(results)

    # optionally, include opponent gender prediction
    # gender_predictions = results[['state', 'district', 'party', 'gender_prediction']].copy()
    # gender_predictions.party = gender_predictions.party.apply(dict(D='R', R='D').get)
    # results = results.merge(gender_predictions.drop(columns='party'), on=['state', 'district'], suffixes=(
    #     '', 'Opponent'))

    # filter results
    preprocessed = results[~results.isUncontested & (results.party == 'D')].copy()
    # take log of receipts and disbursements
    preprocessed.receipts = preprocessed.receipts.apply(np.log)
    preprocessed.disb = preprocessed.disb.apply(np.log)
    preprocessed = preprocessed.dropna()
    return preprocessed


def fit_model_with_gender(preprocessed: pd.DataFrame) -> str:
    formula = 'votePct ~ votePctPres + disb * gender_prediction'
    model = ols(formula, data=preprocessed)
    fitted = model.fit()
    summary = fitted.summary()
    return summary.as_text()
