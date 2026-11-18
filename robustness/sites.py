"""Patient-level collection site for the revision analyses (2026-11-17).

Study-level collection-site map from laboratory records, cross-checked against the per-slide `sites` column of the October laboratory
metadata export. BP007 is split per patient from that column (Walter Reed / Anne Arundel). Read-only inputs; no model access.
"""
import os
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / 'analyses/clinical_cohort_annotations_20261110/patient_clinical_ledger.csv'
SLIDES = REPO / 'analyses/cohort_revision_20261109/input/spectra_study_slides_annotated.csv'
DATA = Path(os.environ.get('SPECTRA_DATA_ROOT', 'data'))
LABMETA = DATA / 'manifests/groups_20261005_no_covid_clhs_v1/laboratory_metadata.csv'

# study-level site map, cross-checked against the laboratory `sites` column
STUDY_SITE = {
    'BC-OPT': 'Billings Clinic', 'D-OPT': 'Duke', 'CH-OPT': 'City of Hope', 'M-OPT-P': 'MD Anderson',
    'U-OPT': 'USC', 'U-OPT-P': 'USC', 'INTC17': 'Cedars-Sinai', 'INTC17-C': 'Cedars-Sinai',
    'PANC': 'Cedars-Sinai', 'PANCEUS': 'Cedars-Sinai', 'PANC-Ctrl': 'Cedars-Sinai', 'PANC-M': 'Mayo Clinic',
    'S1416': 'SWOG S1416 sites', 'S1222': 'Unknown (S1222)', 'NRG-BR002': 'NRG-BR002 sites',
    'TLP': 'University of Colorado', 'ENZA1': 'University of Colorado',
    'MDAnd': 'MD Anderson', 'MDA-AVPC': 'MD Anderson', 'MM02': 'MD Anderson',
    'MSM-GHS': 'Morehouse/Grady', 'MSM-UAB': 'UAB', 'CART-P': 'City of Hope', 'CART-O': 'City of Hope',
    'CART-UP': 'UPenn', 'JHHBa': 'Johns Hopkins', 'VA': 'UCLA', 'KETA': 'MSKCC',
    'SU': 'Stanford', 'UM': 'Manchester', '0S-16-8': 'USC', 'JN': 'USC (JN)', 'JN2': 'USC (JN)', 'JN2-R': 'USC (JN)',
    'ATEZ': 'USC', 'ATEZ-R': 'USC', '2N-19-3': 'USC', 'PDMS_P': 'USC', 'USC-BCa': 'USC', 'USC-UTUC': 'USC',
    'USC-UM': 'USC', 'CHLA-OS': 'CHLA', 'NBD': 'Scripps', 'EPIC': 'Epic Sciences', 'INTC09': 'UChicago',
    'GS': 'Gilead sites', 'MCCT-039': 'UCSD', 'LMD': 'City of Hope',
}


def patient_sites():
    led = pd.read_csv(LEDGER, dtype={'donor_id': str})
    led['site'] = led.study.map(STUDY_SITE).fillna('Unmapped')
    # BP007: per-patient site from the laboratory column
    sl = pd.read_csv(SLIDES, usecols=['patient_id', 'slide_id', 'study'])
    lm = pd.read_csv(LABMETA, usecols=['slide_id', 'sites'])
    b = sl[sl.study == 'BP007'].merge(lm, on='slide_id', how='left')
    bp = b.groupby('patient_id').sites.agg(lambda v: '|'.join(sorted(set(v.dropna().str.strip()))))
    bp = bp.map({'Walter Reed National Military Medical Center': 'Walter Reed', 'Anne Arundel Medical Center': 'Anne Arundel'})
    m = led.study == 'BP007'
    led.loc[m, 'site'] = led.loc[m, 'patient_id'].map(bp).fillna('BP007 unknown')
    return led


if __name__ == '__main__':
    led = patient_sites()
    print(led.site.value_counts().to_string())
    print('Unmapped studies:', sorted(led.loc[led.site == 'Unmapped', 'study'].unique()))
