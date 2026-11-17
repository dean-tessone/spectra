"""Render source-authenticated cell panels from the frozen v1b AttnLRP registry."""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import argparse
import hashlib
import html
import json
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
from scipy.special import expit

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = REPO / 'figures/attribution_cell_review'
RUN = REPO / 'runs/pancancer_cell_attribution_20261020_v1b'
DATA = Path(os.environ.get('SPECTRA_DATA_ROOT', 'data'))
COHORT = DATA / 'manifests/pancancer_combined_final_test_20261019_v1'
SKILL = Path(os.environ.get('NATURE_FIGURE_SKILL', 'nature-figure'))
DEST = Path(os.environ.get('SPECTRA_SHARED_OUT', 'shared_outputs')) / 'attribution_cell_review'
sys.path.insert(0, str(REPO / 'imaging/cell_images'))
sys.path.insert(0, str(SKILL / 'scripts'))
import render_pancreatic_model_comparison as extraction
from cell_ext_display import preprocess_frame, rgb, channel_u8
from spectra.review_intensity import acquisition_gains, apply_gains


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    complete = json.loads((RUN/'complete.json').read_text())
    assert complete['donors'] == 120 and complete['selected_method'] == 'attnlrp'
    for p, h in complete['outputs_sha256'].items():
        assert sha(p) == h, p
    registration = json.loads((RUN/'registration.json').read_text())
    donors_path = RUN/'donors.csv'
    assert sha(donors_path) == registration['files_sha256'][str(donors_path)]
    donors = pd.read_csv(donors_path, dtype={'donor_id': str})
    assert len(donors) == donors.donor_id.nunique() == 120
    manifests = pd.concat([pd.read_csv(COHORT/n, dtype=str) for n in
        ['train_manifest.csv','train_same_tube_endpoint_manifest.csv']])
    mapping = manifests.drop_duplicates('slide_id').set_index('slide_id')
    authenticated = json.loads((RUN/'cache_authentication.json').read_text())['files_sha256']
    selections, sources, receipts, shortages = [], {}, {}, []
    for person in donors.itertuples():
        folder = RUN/'donors'/person.donor_id
        receipt = json.loads((folder/'complete.json').read_text())
        receipts[str(folder/'complete.json')] = sha(folder/'complete.json')
        for name in ['cells.csv.gz','cell_scores.npz','attribution_diagnostics.json']:
            assert sha(folder/name) == receipt['outputs_sha256'][str(folder/name)]
        cells = pd.read_csv(folder/'cells.csv.gz', dtype={'slide_id': str,'donor_id':str,'group_id':str})
        primary = manifests[manifests.donor_id.eq(person.donor_id) & manifests.primary_evaluation.str.lower().eq('true')].group_id.unique()
        if cells.group_id.nunique() > 1:
            primary = sorted(set(primary) & set(cells.group_id))
            assert len(primary) == 1, 'Resolve primary bag without attribution-based selection'
        assert cells.donor_id.eq(person.donor_id).all()
        assert not cells.duplicated(['group_id','slide_id','source_row']).any()
        with np.load(folder/'cell_scores.npz') as z:
            cells['attribution'] = z['attnlrp']
            cells['seed_std'] = z['attnlrp__seed_std']
            assert np.isfinite(cells[['attribution','seed_std']]).all().all()
            for seed in [17,29,43]: cells[f'seed_{seed}'] = z[f'attnlrp__{seed}']
            np.testing.assert_allclose(cells.attribution, cells[['seed_17','seed_29','seed_43']].mean(axis=1))
        if cells.group_id.nunique() > 1:
            cells = cells[cells.group_id.eq(primary[0])].copy()
        diagnostics=json.loads((folder/'attribution_diagnostics.json').read_text())
        context=[r for r in diagnostics if r.get('method')=='attnlrp' and r['group_id']==cells.group_id.iloc[0]]
        assert len(context)==3 and {r['seed'] for r in context}=={17,29,43}
        cells['context_cancer_score']=float(np.mean([expit(r['output']) for r in context]))
        for side, count in [('support',6),('oppose',4)]:
            p = cells[cells.attribution.gt(0) if side == 'support' else cells.attribution.lt(0)]
            p = p.sort_values(['attribution','slide_id','source_row'], ascending=[side != 'support',True,True]).head(count).copy()
            p['side'] = side; p['rank'] = np.arange(1,len(p)+1)
            if len(p)<count: shortages.append(dict(donor_id=person.donor_id,side=side,requested=count,available=len(p)))
            p['phase'] = person.phase; p['fold'] = person.fold
            p['diagnosis'] = person.cancer_types if person.label else 'Control'
            p['study'] = person.study; p['label'] = person.label
            selections.append(p)
    selected = pd.concat(selections, ignore_index=True)
    tables = []
    for slide, part in selected.groupby('slide_id', sort=True):
        item = mapping.loc[slide]; source = Path(item.slide_path).resolve(strict=True)
        assert source.is_relative_to(DATA) 
        assert str(source) in authenticated and sha(source) == authenticated[str(source)], source
        sources[str(source)] = authenticated[str(source)]
        metadata = pd.read_parquet(source, columns=['frame_id','cell_id','x','y','area',*extraction.MEANS])
        rows = part.source_row.to_numpy(dtype=int)
        assert rows.min() >= 0 and rows.max() < len(metadata)
        p = part.reset_index(drop=True).join(metadata.iloc[rows].reset_index(drop=True))
        p['cell_key'] = [f'{slide}_r{int(i):05d}' for i in rows]
        p['source_path'] = str(source); tables.append(p)
    selected = pd.concat(tables,ignore_index=True).sort_values(['donor_id','side','rank'])
    selected.to_csv(OUT/'Displayed_cell_selections.csv',index=False)
    review = selected[['donor_id','diagnosis','study','phase','group_id','slide_id','source_row','frame_id','cell_id','side','rank','attribution','seed_std']].copy()
    for name in ['reviewer','technical_interpretability','observed_phenotype','molecular_followup_priority','notes']:
        review[name] = ''
    review.to_csv(OUT/'Cell_review_template.csv',index=False)
    write(OUT/'selection_receipt.json', dict(method='attnlrp', donors=120,
        selected_rows=len(selected), unique_cells=selected.cell_key.nunique(),
        policy='Frozen primary earliest same-tube bag, 6 highest positive and 4 lowest negative per donor; no visual selection or replacement',
        shortages=shortages,source_sha256=sources,donor_receipts_sha256=receipts,
        registration_sha256=sha(RUN/'registration.json'),
        selection_sha256=sha(OUT/'Displayed_cell_selections.csv'),build_sha256=sha(__file__)))
    print('Selected',len(selected),'cells from120donors,',selected.slide_id.nunique(),'slides',flush=True)


def extract_slide(args):
    slide, part = args
    from slideutils.utils.frame import Frame
    cache = HERE/'image_cache'; cache.mkdir(exist_ok=True,parents=True)
    record_path = cache/f'{slide}.json'; target = cache/f'{slide}.npz'
    part = part.sort_values(['frame_id','source_row']).reset_index(drop=True)
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert record['requested_keys'] == part.cell_key.tolist()
        assert sha(target) == record['cache_sha256']
        return record
    images, masks, keys, frame_records, missing = [], [], [], [], []
    try:
        raw_path = extraction.acquisition(slide, part.frame_id.iloc[0])
    except ValueError as e:
        if not str(e).startswith('Expected one read-only acquisition') or not str(e).endswith('{}'): raise
        raw_path = None
        missing = [dict(cell_key=r.cell_key, slide_id=slide, reason='Raw acquisition unavailable; no replacement') for r in part.itertuples()]
    if raw_path is not None:
        gains = None
        if (raw_path/'slideinfo.txt').is_file():
            try: gains = acquisition_gains((raw_path/'slideinfo.txt').read_text())
            except (ValueError,KeyError): pass
        for fid, events in part.groupby('frame_id',sort=True):
            events = events.reset_index(drop=True)
            try:
                with tempfile.TemporaryDirectory(prefix='review_tiles_',dir=HERE) as tmp:
                    paths, tiles = [], []
                    for offset in [0,2304,4608,9216]:
                        number = int(fid)+offset
                        source = next((raw_path/f'Tile{number:06d}.{ext}' for ext in ['tif','jpg'] if (raw_path/f'Tile{number:06d}.{ext}').is_file()),None)
                        if source is None: raise FileNotFoundError(f'Missing channel tile {number}')
                        dest = Path(tmp)/source.name
                        tiles.append(extraction.stage(source,dest));paths.append(str(dest))
                        if source.suffix == '.jpg': tiles.append(extraction.stage(source.with_suffix('.tags'),dest.with_suffix('.tags')))
                    frame = Frame(int(fid),extraction.CHANNELS,paths); frame.readImage()
                    native,_ = frame.extract_crops(events,75)
                    np.testing.assert_array_equal(native[:,37,37],frame.image[events.y.astype(int),events.x.astype(int)])
                    frame_gain = None; accepted = []; audits = []; mask_cache = {}
                    for i,row in enumerate(events.itertuples()):
                        binary, mask_path = extraction.mask_for_cell(slide,fid,row,mask_cache)
                        if binary is None:
                            missing.append(dict(cell_key=row.cell_key,slide_id=slide,reason='No segmentation matching cell ID, area and centroid'))
                            continue
                        expected = np.array([getattr(row,c) for c in extraction.MEANS],dtype=int)
                        pixels = frame.image[binary]
                        error = int(np.abs(pixels.mean(0).astype(np.uint16).astype(int)-expected).max())
                        gained_error = None if gains is None else int(np.abs(apply_gains(pixels,gains).mean(0).astype(np.uint16).astype(int)-expected).max())
                        if error<=1: effective = np.ones(4)
                        elif gained_error is not None and gained_error<=1: effective = gains
                        else:
                            missing.append(dict(cell_key=row.cell_key,slide_id=slide,reason=f'Intensity source verification failed: native error{error}, gained error{gained_error}'))
                            continue
                        if frame_gain is not None: np.testing.assert_array_equal(effective,frame_gain)
                        frame_gain = effective
                        cut = np.pad(binary,37)[int(row.y):int(row.y)+75,int(row.x):int(row.x)+75]
                        assert cut.shape == (75,75) and cut.any()
                        accepted.append((i,row.cell_key,cut))
                        audits.append(dict(cell_key=row.cell_key,mask_path=str(mask_path),mask_sha256=sha(mask_path),
                            raw_mean_error=error,gained_mean_error=gained_error,effective_gains=effective.tolist(),center_pixel_verified=True))
                    if accepted:
                        frame.image = preprocess_frame(frame.image,frame_gain)
                        processed,_ = frame.extract_crops(events,75)
                        for i,key,cut in accepted:
                            images.append(processed[i]); masks.append(cut); keys.append(key)
                    frame_records.append(dict(frame_id=int(fid),tiles=tiles,cells=audits,
                        full_frame_tophat75=True,gains_applied_once=True))
            except FileNotFoundError as e:
                missing.extend(dict(cell_key=r.cell_key,slide_id=slide,reason=str(e)) for r in events.itertuples())
    np.savez_compressed(target,images=np.stack(images) if images else np.zeros((0,75,75,4),dtype=np.uint16),
        masks=np.stack(masks) if masks else np.zeros((0,75,75),dtype=bool),cell_keys=np.asarray(keys,dtype=str))
    record = dict(slide_id=slide,requested_keys=part.cell_key.tolist(),available_keys=keys,
        missing=missing,frames=frame_records,cache_path=str(target),cache_sha256=sha(target),
        source_writes=False,temporary_tiles_removed=True)
    write(record_path,record)
    return record


def extract():
    cv2.setNumThreads(1)
    d = pd.read_csv(OUT/'Displayed_cell_selections.csv',dtype={'slide_id':str})
    records = []; tasks = list(d.groupby('slide_id',sort=True))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for record in pool.map(extract_slide,tasks):
            records.append(record)
            write(HERE/'status.json',dict(phase='extracting',slides_complete=len(records),slides_total=len(tasks),
                available_cells=sum(len(r['available_keys']) for r in records),updated_unix=time.time()))
            print('Extracted',len(records),'/',len(tasks),'slides;',record['slide_id'],len(record['available_keys']),'available',flush=True)
    write(OUT/'Image_provenance.json',dict(slides=records,source_writes=False,
        transform_sha256=sha(REPO/'imaging/cell_images/cell_ext_display.py'),channel_order=extraction.CHANNELS))
    pd.DataFrame([m for r in records for m in r['missing']],columns=['cell_key','slide_id','reason']).to_csv(OUT/'Missing_cell_images.csv',index=False)


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none'})
    d = pd.read_csv(OUT/'Displayed_cell_selections.csv',dtype={'slide_id':str})
    provenance = json.loads((OUT/'Image_provenance.json').read_text())
    images, masks = {}, {}
    for r in provenance['slides']:
        assert sha(r['cache_path']) == r['cache_sha256']
        with np.load(r['cache_path']) as z:
            images.update(zip(z['cell_keys'].tolist(),z['images']))
            masks.update(zip(z['cell_keys'].tolist(),z['masks']))
    folder = OUT/'patients';folder.mkdir(exist_ok=True)
    donors = pd.read_csv(RUN/'donors.csv')
    donors['order'] = donors.phase.map({'confirmation':0,'development':1})
    donors = donors.sort_values(['order','label','cancer_types','donor_id'])
    qa=[]
    with PdfPages(OUT/'All_patients_cell_panels.pdf') as pdf:
        for person in donors.itertuples():
            rows = d[d.donor_id.eq(person.donor_id)].copy()
            rows['order']=rows.side.map({'support':0,'oppose':1})
            rows=rows.sort_values(['order','rank']).reset_index(drop=True)
            fig, axes = plt.subplots(10,5,figsize=(7,11),squeeze=False)
            fig.subplots_adjust(left=.25,right=.985,top=.91,bottom=.055,wspace=.075,hspace=.13)
            for i, axrow in enumerate(axes):
                if i>=len(rows):
                    for ax in axrow: ax.set_axis_off()
                    continue
                row = rows.iloc[i]; key = row.cell_key
                for j,ax in enumerate(axrow):
                    if key in images:
                        im=rgb(images[key]) if j==0 else channel_u8(images[key][...,j-1])
                        if j==0: ax.imshow(im,interpolation='nearest')
                        else: ax.imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                        if masks[key].any() and not masks[key].all():
                            ax.contour(masks[key],levels=[.5],colors=['white'],linewidths=.4)
                    else:
                        ax.set_facecolor('#EEEEEE');ax.text(.5,.5,'Unavailable',ha='center',va='center',transform=ax.transAxes,fontsize=6)
                    ax.set(xticks=[],yticks=[],xlim=(-.5,74.5),ylim=(74.5,-.5));ax.set_box_aspect(1)
                    for spine in ax.spines.values(): spine.set_linewidth(.35);spine.set_color('#B8BEC3')
            fig.canvas.draw()
            for i,row in rows.iterrows():
                bbox=axes[i,0].get_position();y=(bbox.y0+bbox.y1)/2
                color='#A65D2D' if row.side=='support' else '#24647B'
                fig.text(.023,y,f'{row.side.title()} {int(row["rank"])}\nRelevance {row.attribution:+.4g}\nSeed SD {row.seed_std:.3g}\n{row.slide_id}:r{int(row.source_row)}\nFrame {int(row.frame_id)}, cell {int(row.cell_id)}',
                    va='center',fontsize=6.5,color=color,linespacing=1.25)
            headers=['Composite','DAPI','panCK','CD45/CD31','Vimentin']
            for j,label in enumerate(headers):
                bbox=axes[0,j].get_position();fig.text((bbox.x0+bbox.x1)/2,.925,label,ha='center',fontsize=7)
            diagnosis=person.cancer_types if person.label else 'Control'
            fig.text(.023,.977,f'{person.donor_id}, {diagnosis}; model cancer score {rows.context_cancer_score.iloc[0]:.3f}',fontsize=9,weight='bold')
            fig.text(.023,.953,f'{person.study}, {person.phase}; fold {person.fold}, seeds 17/29/43; donor excluded from these fits',fontsize=6.6)
            fig.text(.023,.032,'White outline: exact segmented target. Four-channel composite: panCK red, CY5 green, DAPI blue, vimentin white.',fontsize=6.1)
            fig.text(.023,.018,'75 x 75 pixels (44.25 um), fixed scale; gains once, full-frame top-hat. Attribution does not establish malignancy.',fontsize=6.1)
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            text_boxes=[t.get_window_extent(renderer) for t in fig.texts]
            page=fig.bbox
            for i,bbox in enumerate(text_boxes):
                assert bbox.x0>=page.x0 and bbox.y0>=page.y0 and bbox.x1<=page.x1 and bbox.y1<=page.y1, 'Clipped figure text'
                assert not any(bbox.overlaps(other) for other in text_boxes[:i]), 'Overlapping figure text'
                assert not any(bbox.overlaps(ax.get_window_extent(renderer)) for ax in axes.flat), 'Figure text intersects image area'
            prefix=folder/person.donor_id
            require_matplotlib_panel_alignment(fig,json_out=prefix.with_suffix('.alignment.json'),
                tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
            fig.savefig(prefix.with_suffix('.pdf')); fig.savefig(prefix.with_suffix('.svg'))
            fig.savefig(prefix.with_suffix('.png'),dpi=300)
            pdf.savefig(fig);plt.close(fig)
            qa.append(dict(donor_id=person.donor_id,selected=len(rows),available=sum(k in images for k in rows.cell_key),
                phase=person.phase,diagnosis=diagnosis,study=person.study))
            print('Rendered',person.donor_id,flush=True)
    pd.DataFrame(qa).to_csv(OUT/'panel_index.csv',index=False)
    body=['<!doctype html><html><head><meta charset="utf-8"><title>SPECTRA attributed cell review</title>',
        '<style>body{font-family:Arial;max-width:1100px;margin:30px auto}img{max-width:700px;width:100%}summary{cursor:pointer;padding:12px;background:#eee}details{margin:10px 0}a{color:#24647b}</style></head><body>',
        '<h1>SPECTRA attributed cell review</h1><p>AttnLRP, 120 donor-excluded examples, six supporting and four opposing events per donor. Exact channels and source IDs. Cells are not molecularly confirmed malignant.</p>',
        '<p><a href="All_patients_cell_panels.pdf">All patients PDF</a> | <a href="Displayed_cell_selections.csv">Cell metadata CSV</a> | <a href="Cell_review_template.csv">Review worksheet</a> | <a href="Missing_cell_images.csv">Unavailable images</a></p>',
        '<label>Filter by donor, cancer, study or phase: <input id="filter" oninput="filterRows()"></label>']
    for r in qa:
        key=html.escape(r['donor_id']);label=html.escape(f"{r['donor_id']}, {r['diagnosis']}, {r['study']}, {r['phase']}; {r['available']}/{r['selected']} images")
        body.append(f'<details data-search="{label.lower()}"><summary>{label}</summary><a href="patients/{key}.pdf">PDF</a><br><img loading="lazy" src="patients/{key}.png" alt="{label}"></details>')
    body.append('<script>function filterRows(){const q=document.getElementById("filter").value.toLowerCase();document.querySelectorAll("details").forEach(x=>x.hidden=!x.dataset.search.includes(q));}</script></body></html>')
    (OUT/'index.html').write_text('\n'.join(body))
    (OUT/'START_HERE.md').write_text('# Attributed cell review\n\nOpen `index.html` in a browser and filter by donor, diagnosis, study or phase. Expand a patient to see the composite and separate channels. `All_patients_cell_panels.pdf` contains all120patients; individual PDFs/PNGs are in `patients/`.\n\nSix strongest positive and four strongest negative signed AttnLRP cells per donor, no visual cherry-picking. Missing images remain placeholders, not replaced. Source coordinates, source rows, scores and seed dispersion are in `Displayed_cell_selections.csv`. Selection is descriptive, not molecular validation. All donors are original training donors, but were excluded from their corresponding CV model fits. Development/confirmation refer to attribution-method selection.\n\nWhite segmentation outlines identify the target; crop context includes neighbors. Fixed four-channel linear scale and audited once-only gains plus full-frame75px top-hat. Vimentin contributes white. Source marker means in the CSV precede display preprocessing.\n')
    write(HERE/'status.json',dict(phase='rendered',donors=len(qa),selected_cells=len(d),available_cells=len(images)))


def audit():
    reports=[]
    for name in ['validate_figure.py','audit_pdf_text.py','audit_figure_collisions.py']:
        target=Path(__file__) if name=='validate_figure.py' else OUT/'All_patients_cell_panels.pdf'
        command=[sys.executable,str(SKILL/'scripts'/name),str(target)]
        if name=='audit_pdf_text.py': command.extend(['--min-pt','5'])
        if name=='audit_figure_collisions.py': command.extend(['--json-out',str(OUT/'collision_audit.json')])
        result=subprocess.run(command,capture_output=True,text=True)
        reports.append(dict(tool=name,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode==1: raise RuntimeError(f'Figure audit failed: {name}: {result.stdout} {result.stderr}')
    write(OUT/'QA.json',dict(audits=reports,alignment='per-patient rendered alignment gate',
        visual_review='Case/control previews inspected; final gallery manual review pending',collision_auditable=reports[-1]['exit_code']==0,
        matplotlib_text_bounds='Every rendered page checks figure-text bounds, text-to-text overlap and text/image-area overlap before export; this does not replace the unavailable PDF collision audit',
        source_warnings='Review bundle uses PDF/SVG plus300dpi PNG previews, not TIFF submission rasters; seed SD is printed numerically beside descriptive images, not plotted as an interval'))


def publish():
    shutil.copy2(Path(__file__),OUT/'build.py')
    shutil.copy2(HERE/'verify_bundle.py',OUT/'verify_bundle.py')
    shutil.copy2(REPO/'imaging/cell_images/cell_ext_display.py',OUT/'display_transform.py')
    DEST.mkdir(parents=True,exist_ok=True)
    prior_path=OUT/'export_receipt.json'
    prior=json.loads(prior_path.read_text()) if prior_path.exists() else {'files_sha256':{}}
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p!=prior_path)
    for p in files:
        name=str(p.relative_to(OUT));q=DEST/name
        if q.exists() and sha(q)!=sha(p):
            assert sha(q)==prior['files_sha256'].get(name),f'Unexpected destination change: {q}'
    hashes={}
    for p in files:
        name=str(p.relative_to(OUT));q=DEST/name;q.parent.mkdir(parents=True,exist_ok=True)
        if not q.exists() or sha(q)!=sha(p): shutil.copy2(p,q)
        assert sha(p)==sha(q);hashes[name]=sha(q)
    write(prior_path,dict(destination=str(DEST),files_sha256=hashes,verified_files=len(hashes)))
    shutil.copy2(prior_path,DEST/'export_receipt.json')
    print('Published',len(files),'verified files to',DEST,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','extract','render','audit','publish','all']);a=parser.parse_args()
    funcs=dict(prepare=prepare,extract=extract,render=render,audit=audit,publish=publish)
    if a.stage=='all':
        for f in funcs.values(): f()
    else: funcs[a.stage]()
