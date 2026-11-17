"""Regenerate manuscript Tables1/2 from frozen cohort and matched CV sources."""
from pathlib import Path
import sys,html,zipfile,datetime,json,numbers
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape
import pandas as pd
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import read,write,sha,utc
OUT=Path(__file__).resolve().parent
POP=REPO/'cohort/reference_policy/population_patient_ledger.csv'
CV=REPO/'analyses/reference_mil_cv_20261110/model_comparison.csv'
CLIN=REPO/'analyses/clinical_cohort_annotations_20261110/patient_clinical_ledger.csv'
DISPLAY={'Multiple Myeloma':'Multiple myeloma and related states','Bone':'Bone tumors','HeadNeck':'Head and neck','Upper Tract Urothelial Carcinoma':'Upper tract urothelial carcinoma','Uveal Melanoma':'Uveal melanoma','Leptomeningeal Disease (LMD)':'Leptomeningeal disease'}
T1_TITLE='Table 1. Patient distribution by cancer cohort and control population.'
T2_TITLE='Table 2. Comparison of SPECTRA with alternative pooling strategies in matched cross-validation.'
T1_NOTE='Counts represent unique patients, not slides or cell groups. The primary cancer-versus-reference analysis includes 2,550 patients: 1,689 cancer-cohort patients and 104 reference controls for training, and 713 cancer-cohort patients and 44 reference controls for testing. Reference controls comprise NBD, EPIC, INTC17-C and PANC-Ctrl; PANC-Ctrl represents non-cancer pancreatic workup. Lung-workup controls (SU, JN2 and JN2-R) and BRCA-risk controls (INTC09) are retained for separate cancer-specific analyses. The multiple-myeloma cohort includes precursor and related disease states; these are detailed in the companion state table. Counts follow the frozen cohort labels; no labels were changed for these tables.'
T2_NOTE='Values are means across three patient-exclusive validation folds. Within each fold, predictions from seeds 17 and 29 were averaged before scoring. All methods used the same 1,793 training-cohort patients, fixed 15 epochs, input embeddings, three cell filters, fitting-only normalizers, sampling schedules and optimizer budget. The primary endpoint uses one predefined group per patient. AUROC pools cancer types within each fold; macro AUROC averages nine prespecified cancer categories. Parameter counts exclude the shared frozen image encoder. Architectures and their native losses/dropout settings were fixed, rather than separately optimized. These are training-CV results, not final-test results; no AUROC confidence intervals are reported.'


def md_table(df,fmt=False):
 values=[]
 for row in df.itertuples(index=False,name=None):
  values.append([f'{v:.3f}' if fmt and isinstance(v,float) else f'{v:,}' if isinstance(v,numbers.Integral) else str(v) for v in row])
 return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']+['---:']*(len(df.columns)-1))+' |',*['| '+' | '.join(row)+' |' for row in values]])


def docx_table(df,widths,decimals=False):
 grid=''.join(f'<w:gridCol w:w="{v}"/>' for v in widths)
 rows=[]
 for i,row in enumerate([list(df.columns),*df.itertuples(index=False,name=None)]):
  cells=[]
  for j,value in enumerate(row):
   s=(f'{value:.3f}' if i and decimals and isinstance(value,float) else f'{value:,}' if i and isinstance(value,numbers.Integral) else str(value))
   bold='<w:b/>' if i==0 or str(row[0]).startswith('Total') else ''
   align='left' if j==0 else 'right'
   cells.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/></w:tcPr><w:p><w:pPr><w:jc w:val="{align}"/><w:spacing w:after="35" w:before="35"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="18"/>{bold}</w:rPr><w:t xml:space="preserve">{escape(s)}</w:t></w:r></w:p></w:tc>')
  rows.append('<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if i==0 else '')+'</w:trPr>'+''.join(cells)+'</w:tr>')
 return '<w:tbl><w:tblPr><w:tblW w:w="10080" w:type="dxa"/><w:tblLayout w:type="fixed"/><w:tblBorders><w:top w:val="single" w:sz="6"/><w:bottom w:val="single" w:sz="6"/></w:tblBorders><w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:bottom w:w="60" w:type="dxa"/><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>'+grid+'</w:tblGrid>'+''.join(rows)+'</w:tbl>'


def write_docx(table1,table2):
 def paragraph(text,bold=False,size=20):return f'<w:p><w:pPr><w:spacing w:after="120"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="{size}"/>'+('<w:b/>' if bold else '')+f'</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
 body=paragraph(T1_TITLE,True,22)+docx_table(table1,[5800,1426,1427,1427])+paragraph(T1_NOTE,size=17)+'<w:p><w:r><w:br w:type="page"/></w:r></w:p>'+paragraph(T2_TITLE,True,22)+docx_table(table2,[2500,1700,2940,2940],True)+paragraph(T2_NOTE,size=17)
 document='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+body+'<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/></w:sectPr></w:body></w:document>'
 ct='<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
 rel='<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
 with zipfile.ZipFile(OUT/'SPECTRA_Tables_1_and_2.docx','w',zipfile.ZIP_DEFLATED) as z:
  z.writestr('[Content_Types].xml',ct);z.writestr('_rels/.rels',rel);z.writestr('word/document.xml',document)


def write_xlsx(sheets):
 ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
 content=['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>','<Default Extension="xml" ContentType="application/xml"/>','<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>','<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
 workbook=[];relationships=[]
 def column(n):
  s=''
  while n:n,r=divmod(n-1,26);s=chr(65+r)+s
  return s
 with zipfile.ZipFile(OUT/'SPECTRA_Tables_1_and_2.xlsx','w',zipfile.ZIP_DEFLATED) as z:
  for k,(name,df) in enumerate(sheets.items(),1):
   assert len(name)<=31
   workbook.append(f'<sheet name="{escape(name)}" sheetId="{k}" r:id="rId{k}"/>');relationships.append(f'<Relationship Id="rId{k}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{k}.xml"/>');content.append(f'<Override PartName="/xl/worksheets/sheet{k}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
   rows=[]
   for i,row in enumerate([list(df.columns),*df.itertuples(index=False,name=None)],1):
    cells=[]
    for j,v in enumerate(row,1):
     address=f'{column(j)}{i}'
     if i>1 and isinstance(v,numbers.Real) and not isinstance(v,bool) and pd.notna(v):cells.append(f'<c r="{address}" s="{2 if isinstance(v,float) else 0}"><v>{v}</v></c>')
     else:cells.append(f'<c r="{address}" t="inlineStr" s="{1 if i==1 else 0}"><is><t xml:space="preserve">{escape(str(v))}</t></is></c>')
    rows.append(f'<row r="{i}">'+''.join(cells)+'</row>')
   cols='<cols><col min="1" max="1" width="42" customWidth="1"/><col min="2" max="30" width="22" customWidth="1"/></cols>'
   z.writestr(f'xl/worksheets/sheet{k}.xml',f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>{cols}<sheetData>'+''.join(rows)+'</sheetData></worksheet>')
  relationships.append(f'<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
  z.writestr('xl/workbook.xml',f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+''.join(workbook)+'</sheets></workbook>')
  z.writestr('xl/_rels/workbook.xml.rels','<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(relationships)+'</Relationships>')
  z.writestr('xl/styles.xml',f'<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="{ns}"><numFmts count="1"><numFmt numFmtId="164" formatCode="0.000"/></numFmts><fonts count="2"><font><sz val="11"/><name val="Arial"/></font><font><b/><sz val="11"/><name val="Arial"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs></styleSheet>')
  z.writestr('_rels/.rels','<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
  z.writestr('[Content_Types].xml','<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'+''.join(content)+'</Types>')


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 delivery=read(REPO/'analyses/reference_mil_cv_20261110/delivery_receipt.json')
 assert sha(CV)==delivery['outputs_sha256']['model_comparison.csv']
 ledger=pd.read_csv(POP);assert len(ledger)==2653 and ledger.donor_id.is_unique
 cases=ledger[ledger.label.eq(1)];assert len(cases)==2402
 def count_row(name,d):return {'Cancer cohort or control population':name,'Training':int(d.split.eq('train').sum()),'Test':int(d.split.eq('test').sum()),'Total':len(d)}
 rows=[]
 for cancer,d in sorted(cases.groupby('cancer_types'),key=lambda v:(-len(v[1]),v[0])):rows.append(count_row(DISPLAY.get(cancer,cancer),d))
 rows.append(count_row('Total cancer-cohort patients',cases))
 reference=ledger[ledger.primary_reference_control];lung=ledger[ledger.label.eq(0)&ledger.study.isin(['SU','JN2','JN2-R'])];brca=ledger[ledger.label.eq(0)&ledger.study.eq('INTC09')]
 assert len(reference)==148 and len(lung)==81 and len(brca)==22
 rows += [count_row('Reference controls',reference),count_row('Lung-workup controls',lung),count_row('BRCA-risk controls',brca),count_row('Total study patients',ledger)]
 table1=pd.DataFrame(rows);table1.to_csv(OUT/'Table_1_patient_counts.csv',index=False)
 primary=ledger[ledger.primary_cancer_vs_reference_eligible]
 assert len(primary)==2550 and primary.split.eq('train').sum()==1793 and primary.split.eq('test').sum()==757
 count_row_primary=count_row('Primary cancer-versus-reference analysis',primary)
 pd.DataFrame([count_row('Cancer-cohort patients',cases),count_row('Reference controls',reference),count_row_primary]).to_csv(OUT/'Table_1_primary_analysis_totals.csv',index=False)
 ctrl=[]
 for study,d in ledger[ledger.label.eq(0)].groupby('study'):
  row=count_row(study,d);row['Role']='Primary reference' if d.primary_reference_control.all() else 'Lung-specific comparator' if study in ['SU','JN2','JN2-R'] else 'Breast-specific comparator';ctrl.append(row)
 controls=pd.DataFrame(ctrl).sort_values(['Role','Cancer cohort or control population']);controls.to_csv(OUT/'Table_1_control_studies.csv',index=False)
 clinical=pd.read_csv(CLIN);myeloma=clinical[clinical.label.eq(1)&clinical.cancer_types.eq('Multiple Myeloma')];states=pd.DataFrame([count_row(state,d) for state,d in myeloma.groupby('mm_state')]);states=states.rename(columns={'Cancer cohort or control population':'Myeloma-cohort disease state'});states.to_csv(OUT/'Table_1_myeloma_states.csv',index=False);assert states.Total.sum()==71
 cv=pd.read_csv(CV);selected=cv[cv.endpoint.eq('primary')].copy();assert len(selected)==6 and selected.family.is_unique
 selected=selected.sort_values('mean_fold_pooled_AUROC',ascending=False)
 table2=selected[['model','parameters','mean_fold_pooled_AUROC','mean_fold_fixed_nine_macro_AUROC']].rename(columns={'model':'Model','parameters':'Parameters','mean_fold_pooled_AUROC':'Mean CV AUROC','mean_fold_fixed_nine_macro_AUROC':'Mean CV macro AUROC'})
 table2.loc[table2.Model.eq('SPECTRA, six layers'),'Model']='SPECTRA (six layers)'
 table2.to_csv(OUT/'Table_2_pooling_comparison_CV.csv',index=False)
 selected.to_csv(OUT/'Table_2_full_precision_and_OOF.csv',index=False)
 cv[cv.endpoint.eq('strict_same_draw_tube_supplemental')].to_csv(OUT/'Table_2_supplemental_same_tube_CV.csv',index=False)
 markdown=f'# {T1_TITLE}\n\n{md_table(table1)}\n\n{T1_NOTE}\n\n# {T2_TITLE}\n\n{md_table(table2,True)}\n\n{T2_NOTE}\n'
 (OUT/'Tables_1_and_2.md').write_text(markdown)
 (OUT/'Tables_1_and_2.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>SPECTRA Tables 1 and 2</title><style>body{font:11pt Arial;max-width:7in;margin:24px auto}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{padding:5px 8px;text-align:right}th:first-child,td:first-child{text-align:left}thead{border-top:1px solid;border-bottom:1px solid}tbody{border-bottom:1px solid}.note{font-size:9pt}h2{font-size:12pt;margin-top:24px}</style></head><body><h2>'+html.escape(T1_TITLE)+'</h2>'+table1.to_html(index=False,border=0)+'<p class="note">'+html.escape(T1_NOTE)+'</p><h2>'+html.escape(T2_TITLE)+'</h2>'+table2.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')+'<p class="note">'+html.escape(T2_NOTE)+'</p></body></html>')
 write_docx(table1,table2)
 write_xlsx({'Table 1':table1,'Table 2 CV':table2,'Primary analysis counts':pd.DataFrame([count_row_primary]),'Control study breakdown':controls,'Myeloma cohort states':states,'Table 2 precision and OOF':selected})
 # Validate exported containers and table structure without claiming visual rendering QA.
 for name in ['SPECTRA_Tables_1_and_2.docx','SPECTRA_Tables_1_and_2.xlsx']:
  with zipfile.ZipFile(OUT/name) as z:
   assert z.testzip() is None
   for n in z.namelist():
    if n.endswith('.xml') or n.endswith('.rels'):ET.fromstring(z.read(n))
   if name.endswith('.docx'):
    doc=ET.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};tables=doc.findall('.//w:tbl',ns)
    assert len(tables)==2 and len(tables[0].findall('w:tr',ns))==len(table1)+1 and len(tables[1].findall('w:tr',ns))==7
 assert pd.read_csv(OUT/'Table_1_patient_counts.csv').Total.tolist()==table1.Total.tolist()
 (OUT/'README.md').write_text('Editable manuscript Tables1/2 regenerated from the study cohort and completed matched CV. Table1 includes all2653patients across cancer and main/specializedcontrol cohorts;primaryanalysis2550explicit. Counts unique canonicaldonors,all exclusions frozen. Myeloma/relatedstates andcontrolroles supplied as companion sheets. Table2primary meanfold two-seedAUROC/macro,not finaltest/no CIs. Fullprecision/pooledOOF andstrictsame-tube supplementary tables separate. No count-based trainedMIL baseline was fit in this CV campaign,so none fabricated. DOCXandXLSX structural ZIP/XML/rowcount validation passed;visual pagination/rendering not independently audited. Scientificfigures anddraftPDF not edited by this table build.\n')
 files=[p for p in OUT.iterdir() if p.suffix in ['.csv','.md','.html','.docx','.xlsx']]
 write(OUT/'provenance.json',dict(created_utc=utc(),inputs_sha256={str(p):sha(p) for p in [POP,CV,CLIN,Path(__file__)]},outputs_sha256={str(p):sha(p) for p in files},unique_study_patients=2653,primary_patients=2550,cancer_cohort_patients=2402,reference_controls=148,specialized_controls=103,CV_folds=3,CV_seeds=[17,29],CV_epochs=15,test_metrics_not_used_in_table2=True,validation='Cohortcounts/CVsourcehash/ZIPXML/tablecellcounts;visualrenderingnotaudited'))
 print(table1.to_string(index=False));print(table2.to_string(index=False))

if __name__=='__main__':main()
