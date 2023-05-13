import datetime
import json
import re

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols


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


def _add_fec_data(results: pd.DataFrame) -> pd.DataFrame:
    results = results.merge(_read_fec_data(), on=['state', 'district', 'party'])
    results.receipts = results.receipts.apply(np.log)
    results.disb = results.disb.apply(np.log)
    results = results[results.disb > 0].dropna()
    return results


def _read_gender_predictions() -> pd.DataFrame:
    reference = pd.read_csv('G:/GitHub/name-finder/gender_prediction_reference.csv', dtype=str)
    reference = reference[reference.gender_prediction.isin(('f', 'm'))].copy()
    reference['isF'] = reference.gender_prediction == 'f'
    reference = reference.drop(columns='gender_prediction')
    additional_predictions = pd.DataFrame([
        dict(name='Tudor', isF=True),
        dict(name='JB', isF=False),
    ])
    reference = pd.concat((reference, additional_predictions)).drop_duplicates(keep='last')
    return reference


def _add_gender_predictions(results: pd.DataFrame) -> pd.DataFrame:
    results['firstName'] = results.fullName.apply(lambda x: ''.join(re.findall('[A-Za-z]', x.split(None, 1)[0])))
    results = results.merge(_read_gender_predictions(), left_on='firstName', right_on='name', how='left').drop(
        columns=['fullName', 'name', 'firstName'])
    return results


def preprocess_data(results: pd.DataFrame, include_fec: bool = False) -> pd.DataFrame:
    # read election results
    results = results[~results.isUncontested].copy()

    # drop seats with more than two D+R candidates
    more_than_two_candidates = results.groupby('pecan', as_index=False).id.count()
    more_than_two_candidates = more_than_two_candidates[more_than_two_candidates.id > 2]
    results = results[~results.pecan.isin(more_than_two_candidates.pecan)].copy()

    # add gender predictions
    gender_predictions = _add_gender_predictions(results[['pecan', 'party', 'fullName']].copy())
    opponent = gender_predictions.copy()
    opponent.party = opponent.party.apply(dict(D='R', R='D').get)
    gender_predictions = gender_predictions.merge(opponent, on=['pecan', 'party'], suffixes=('', 'Opponent'))
    results = results.merge(gender_predictions, on=['pecan', 'party'])

    if include_fec:
        results = _add_fec_data(results)

    results['votePctDiff'] = results.votePct - results.votePctPres
    return results


def fit_model(results: pd.DataFrame, formula: str) -> None:
    model = ols(formula, data=results)
    fitted = model.fit()
    summary = fitted.summary()
    print(summary.as_text())
