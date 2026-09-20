"""Render frozen checkpoint diagnostics and recorded trajectories; no estimators."""
from pathlib import Path
import argparse
import hashlib
import json
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import yaml


def build(config_path):
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text())['report']
    source = (config_path.parent / config['source']).resolve()
    output = (config_path.parent / config['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    execution = json.loads((source / 'provenance/execution_manifest.original_paths.json').read_text())
    hashes = {}
    def verify(path, expected):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f'Checksum mismatch: {path}')
        hashes[str(path.relative_to(source))] = digest
    for item in execution['canonical_inputs'].values():
        verify(source / 'input' / Path(item['path']).name, item['sha256'])
    parts = []
    for completion_path in sorted((source / 'groups').glob('*.complete.json')):
        completion = json.loads(completion_path.read_text())
        stem = completion_path.name.removesuffix('.complete.json')
        for suffix, key in [('information', 'information_sha256'), ('support', 'support_sha256')]:
            verify(source / 'groups' / f'{stem}.{suffix}.parquet', completion[key])
        parts.append(pd.read_parquet(source / 'groups' / f'{stem}.information.parquet'))
    estimates = pd.concat(parts, ignore_index=True)
    cells = pd.read_parquet(source / 'input/cells.parquet')
    submissions = pd.read_csv(source / 'provenance/submission_manifest.csv')
    names = dict(zip(submissions.array_index, submissions.config_path.map(lambda x: Path(x).stem)))
    labels = {row.cell_id: names[row.source_config_index] for row in cells.itertuples()}
    columns = ['cell_id', 'episode_id', 'parent_id', 'copy_id', 'branch_policy', 'posting_budget', 'round_index', 'truth_vote_share', 'truth_vote_share_before', 'sensor_sample_size']
    rounds = pd.read_parquet(source / 'input/rounds.parquet', columns=columns)
    episodes = pd.read_parquet(source / 'input/episodes.parquet', columns=['cell_id', 'episode_id', 'status'])
    assert episodes.status.eq('completed').all()
    branches = rounds[~rounds.branch_policy.isin(['preparation', 'checkpoint'])].copy()
    branches['posting_budget'] = branches.posting_budget.fillna(0).astype(int)
    keys = ['cell_id', 'episode_id', 'parent_id', 'copy_id', 'branch_policy', 'posting_budget']
    assert not branches.duplicated(keys + ['round_index']).any()
    sizes = branches.groupby(keys, dropna=False).size()
    assert sizes.eq(10).all(), 'Only full ten-round paths may enter this descriptive report'
    assert len(sizes) == len(episodes) * 9
    assert branches[['truth_vote_share', 'truth_vote_share_before']].notna().all().all()
    branches['branch_label'] = branches.branch_policy + branches.posting_budget.map(lambda b: '' if b == 0 else f' b={b}')
    branches['config_label'] = branches.cell_id.map(labels)
    order = ['none'] + [f'{policy} b={b}' for policy in ['always_truth','always_false','sensing_truth','sensing_false'] for b in [3,12]]
    ordered_cells = sorted(labels, key=lambda key: labels[key])
    notes = [
        'PROVISIONAL — saved results only. No aggregation, causal estimator, classifier, bootstrap or permutation was run.',
        'The canonical tables contain 100 completed parent bundles out of 160 expected: 900 complete continuation paths, each with 10 rounds. Counts by configuration appear below.',
        'This report uses canonical completed parents only. The archive also contains interrupted prefixes; recovering additional complete paths is a separate analysis and is not performed here. The archived handoff describes greater recoverable coverage, which is not the plotted sample.',
        'Only sensor MAE, sensor MSE and controller-action entropy are saved in the estimator fragments. Checkpoint paired causal responses, assigned-policy information, classifier scores, null comparisons and propensity-weighted phase maps are unavailable in this snapshot.',
        'Bars are descriptive means, without newly estimated confidence intervals. Every parent contributes one path to each branch. Preparation/checkpoint records are excluded. Truth share is used across all policies so the vertical axis has the same meaning.',
        'Configuration labels use q for social group size (3 or 12) and rho for persistence (0.70 or 1.00). The controller sensor sample size is a separate quantity, fixed at 12 in every configuration.'
    ]
    estimates['config_label'] = estimates.cell_id.map(labels)
    estimates.to_parquet(output / 'saved_diagnostics.parquet', index=False)
    summaries = []
    examples = []
    pages = []
    with PdfPages(output / 'report.pdf') as pdf:
        def save(fig, name):
            fig.savefig(figures / f'{name}.png', dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
            pages.append(name)
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06,.94,config['title'],fontsize=18,weight='bold')
        y=.87
        for note in notes:
            lines=textwrap.wrap(note,145)
            fig.text(.06,y,'\n'.join(lines),fontsize=10,va='top')
            y-=len(lines)*.027+.025
        counts = cells.assign(config_label=cells.cell_id.map(labels))
        lines = [f'{row.config_label}: {row.completed_episodes}/40 completed parent bundles' for row in counts.itertuples()]
        fig.text(.06,y,'\n'.join(lines),fontsize=11,va='top')
        save(fig,'overview')
        fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off')
        fields=['config_label','metric','estimate','ci_low','ci_high','null_permutations']
        frame=estimates[fields].sort_values(['config_label','metric'])
        rows=[[str(v) if not isinstance(v,float) else ('N/A' if np.isnan(v) else f'{v:.5g}') for v in row] for row in frame.itertuples(index=False,name=None)]
        table=ax.table(cellText=rows,colLabels=['Config','Stored metric','Estimate','CI low','CI high','Null draws'],loc='center',colWidths=[.16,.35,.12,.12,.12,.1])
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
        ax.set_title('Saved diagnostic estimates — unchanged values\nNo saved null comparisons for these metrics',pad=20)
        save(fig,'saved_diagnostics')
        for mode,field in [('start_of_round','truth_vote_share_before'),('end_of_round','truth_vote_share'),('final_episode','truth_vote_share')]:
            selected = branches if mode!='final_episode' else branches.sort_values('round_index').groupby(keys,dropna=False).tail(1)
            summary=selected.groupby(['cell_id','config_label','branch_label'],as_index=False).agg(mean_truth_share=(field,'mean'),n_records=(field,'size'))
            summary['boundary']=mode;summaries.append(summary)
            fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
            for ax,cell in zip(axes.flat,ordered_cells):
                s=summary[summary.cell_id.eq(cell)].set_index('branch_label').reindex(order)
                bars=ax.bar(range(9),100*s.mean_truth_share)
                ax.bar_label(bars,fmt='%.1f',fontsize=7)
                ax.set_xticks(range(9),order,rotation=55,ha='right',fontsize=8)
                ax.set_ylim(0,110);ax.set_ylabel('Truth share (%)');ax.set_title(labels[cell]);ax.grid(axis='y',alpha=.2)
            fig.suptitle(mode.replace('_',' ').title()+' truth share — descriptive means, no new confidence intervals',fontsize=14)
            save(fig,mode)
        for cell in ordered_cells:
            parents=branches[branches.cell_id.eq(cell)][['episode_id','parent_id']].drop_duplicates().sort_values(['episode_id','parent_id'])
            identity=parents.iloc[0]
            selected=branches[branches.cell_id.eq(cell)&branches.episode_id.eq(identity.episode_id)&branches.parent_id.eq(identity.parent_id)]
            fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
            for ax,label in zip(axes.flat,order):
                r=selected[selected.branch_label.eq(label)].sort_values('round_index')
                assert len(r)==10
                ax.plot(r.round_index,r.truth_vote_share_before,':',label='before')
                ax.plot(r.round_index,r.truth_vote_share,'.-',label='after')
                ax.set(title=label,ylim=(0,1),xlabel='Recorded round',ylabel='Truth share');ax.grid(alpha=.2)
            axes.flat[0].legend(fontsize=8)
            fig.suptitle(f'{labels[cell]} — parent {identity.episode_id}\nFirst sorted completed parent; illustrative, not outcome-selected')
            save(fig,'example_'+labels[cell])
            examples.append({'cell_id':cell,'config_label':labels[cell],'episode_id':identity.episode_id,'parent_id':identity.parent_id})
    pd.concat(summaries).to_parquet(output/'descriptive_bars.parquet',index=False)
    (output/'report.md').write_text('# '+config['title']+'\n\n'+'\n\n'.join(notes)+'\n\n'+'\n\n'.join(f'![{p}](figures/{p}.png)' for p in pages))
    manifest={'source':str(source),'source_hashes_verified':hashes,'config':str(config_path),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'provisional_saved_results_only','completed_parents':len(episodes),'plotted_paths':len(sizes),'saved_estimate_rows':len(estimates),'new_estimator_calls':0,'new_resampling_calls':0,'descriptive_operations':['arithmetic means','last recorded round selection'],'pages':pages,'episode_examples':examples,'limitations':notes}
    (output/'report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Built {len(pages)} pages: {output / "report.pdf"}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    build(parser.parse_args().config)
