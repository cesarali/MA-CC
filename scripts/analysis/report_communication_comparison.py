"""Display saved communication-comparison results without scientific reanalysis."""
from pathlib import Path
import argparse,hashlib,json,textwrap
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def build(path):
    path=path.resolve();config=yaml.safe_load(path.read_text())['report']
    source_paths=config.get('source_analyses') or [config['source_analysis']]
    sources=[(path.parent/p).resolve() for p in source_paths]
    source=sources[0];out=(path.parent/config['output_dir']).resolve()
    def read_table(name):
        frames=[pd.read_parquet(root/'tables'/f'{name}.parquet') for root in sources]
        return pd.concat(frames,ignore_index=True) if len(frames)>1 else frames[0]
    figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
    cells=read_table('cells')
    cells['arm']=cells.target.map({'correct':'truth','ALLOCATION_2':'false'}).fillna('none')
    assert cells.loc[cells.arm.eq('none'),'target'].isna().all()
    cells['budget']=cells.intervention_budget.fillna(0)
    assert not cells.cell_id.duplicated().any(), 'Overlapping cell IDs across report sources'
    assert not cells.duplicated(['arm','budget','epistemic_persistence','communication_profile']).any(), 'Overlapping scientific plot coordinates'
    coords=['cell_id','arm','budget','epistemic_persistence','communication_profile']
    raw=read_table('rounds')
    episodes=read_table('episodes')
    episodes=episodes[episodes.status.eq('completed')]
    fields=['cell_id','episode_id','round_index','controller_target_share_before','controller_target_share','truth_vote_share_before','truth_vote_share']
    r=raw[fields].merge(episodes[['cell_id','episode_id']],on=['cell_id','episode_id'],validate='many_to_one').merge(cells[coords],on='cell_id',validate='many_to_one')
    assert not r.duplicated(['cell_id','episode_id','round_index']).any()
    assert r.groupby(['cell_id','episode_id']).size().eq(int(cells.population_rounds.iloc[0])).all()
    for boundary in ['before','after']:
        target='controller_target_share_before' if boundary=='before' else 'controller_target_share'
        truth='truth_vote_share_before' if boundary=='before' else 'truth_vote_share'
        r['share_'+boundary]=r[target].where(r.arm.ne('none'),r[truth])
        assert r['share_'+boundary].between(0,1).all()
    profiles=['report_only','full_communication'];colors=['#3678a8','#dc8740']
    budgets=sorted(cells.loc[cells.arm.ne('none'),'budget'].unique())
    budget_labels=[str(int(b))+('*' if b in config.get('extension_budgets',[]) else '') for b in budgets]
    panels=[('none',0)]+[(arm,b) for arm in ['truth','false'] for b in budgets]
    rhos=sorted(cells.epistemic_persistence.unique());pages=[];summaries=[];examples=[]
    validations=[json.loads((root/'validation.json').read_text()) for root in sources]
    v=validations[0] if len(sources)==1 else dict(complete=all(x['complete'] for x in validations),counts={k:sum(x['counts'].get(k,0) for x in validations) for k in validations[0]['counts']},source_validations=validations,report_only_collection=True)
    notes=[
      f"Status: {'COMPLETE' if v['complete'] else 'PROVISIONAL'}. {v['counts']['completed_episodes']}/{v['counts']['expected_episodes']} completed episodes; {v['counts']['found_cells']}/{v['counts']['expected_cells']} cells present. {int(cells.population_rounds.iloc[0])} rounds per episode. Interrupted prefixes are excluded. Missing comparisons are gaps, not zero.",
      'Bars compare report-only and full-communication profiles within the same arm, budget and persistence. No budgets or control arms are pooled. Target share means truth-target share under truth control and ALLOCATION_2 share under false control. No-control panels show truth share.',
      'Start/end-of-round bars are arithmetic means across retained rounds. Final bars use the last recorded round per completed episode. Error bars show +/-1 sample standard deviation across episode-level shares (episode round means for start/end bars, final values for final bars). Variance is saved in the CSV. These are descriptive spreads, not confidence intervals; endpoints can extend beyond 0–100%.',
      'All scientific estimates, confidence intervals and null summaries are copied from the archive. No aggregation, estimators, bootstrap, permutations or simulations are run. Null p-values are unadjusted for multiple comparisons.',
      'The archive contains empty causal-response effect tables. Its causal validation labels controlled episodes incomplete despite the canonical completed-episode count; no causal-response claim is made here. Occupancy bars use canonical completed episodes and recorded shares.',
      'Trajectory examples use the first sorted episode in each cell, independently of outcome. Examples across profiles are not assumed to be matched. Complete archive validation warnings are retained in report_manifest.json.'
    ]
    if config.get('comparison_note'):notes.insert(1,config['comparison_note'])
    with PdfPages(out/'report.pdf') as pdf:
        def save(fig,name):
            fig.savefig(figures/(name+'.png'),dpi=160);pdf.savefig(fig);plt.close(fig);pages.append(name)
        fig=plt.figure(figsize=(12,9));fig.text(.06,.94,config['title'],fontsize=20,weight='bold');y=.86
        for note in notes:
            lines=textwrap.wrap(note,130);fig.text(.06,y,'\n'.join(lines),fontsize=11,va='top');y-=.029*len(lines)+.025
        save(fig,'overview')
        for mode in ['before','after','final']:
            selected=r if mode!='final' else r.sort_values('round_index').groupby(['cell_id','episode_id']).tail(1)
            field='share_before' if mode=='before' else 'share_after'
            summary=selected.groupby(coords,as_index=False).agg(mean_share=(field,'mean'),n_observations=(field,'size'))
            episode_values=selected.groupby(coords+['episode_id'],as_index=False)[field].mean()
            spread=episode_values.groupby(coords,as_index=False).agg(episode_sd=(field,'std'),episode_variance=(field,'var'),n_episodes=(field,'size'))
            summary=summary.merge(spread,on=coords,validate='one_to_one')
            summary['episode_variance_percentage_points_squared']=summary.episode_variance*10000
            summary['boundary']=mode;summaries.append(summary)
            fig,axes=plt.subplots(int(np.ceil((len(panels)+1)/3)),3,figsize=(14,4.5*int(np.ceil((len(panels)+1)/3))),layout='constrained')
            for ax,(arm,budget) in zip(axes.flat,panels):
                s=summary[summary.arm.eq(arm)&summary.budget.eq(budget)]
                for j,(profile,color) in enumerate(zip(profiles,colors)):
                    data=s[s.communication_profile.eq(profile)].set_index('epistemic_persistence').reindex(rhos)
                    x=np.arange(len(rhos))+(j-.5)*.34
                    bars=ax.bar(x,data.mean_share*100,width=.32,color=color,label=profile.replace('_',' '))
                    ax.bar_label(bars,fmt='%.1f%%',fontsize=9,padding=3)
                    valid=data.mean_share.notna() & data.episode_sd.notna()
                    ax.errorbar(x[valid],data.mean_share[valid]*100,yerr=data.episode_sd[valid]*100,fmt='none',ecolor='black',capsize=4,elinewidth=1.2,zorder=3)
                ax.set(xticks=np.arange(len(rhos)),xticklabels=rhos,ylim=(min(0,float((summary.mean_share-summary.episode_sd).min()*100)-5),max(112,float((summary.mean_share+summary.episode_sd).max()*100)+5)),xlabel='Persistence rho',ylabel='Share (%)',title='No control: truth share' if arm=='none' else f'{arm.title()}-target share, b={budget}'+(' [extension]' if budget in config.get('extension_budgets',[]) else ''))
                ax.grid(axis='y',alpha=.15)
            axes.flat[0].legend(fontsize=8)
            for empty_ax in list(axes.flat)[len(panels):]:empty_ax.axis('off')
            axes.flat[-1].text(.02,.8,'Bars: descriptive means\nError bars: +/-1 episode SD\nSD = square root of sample variance\nNot confidence intervals\nEach comparison keeps budget and arm fixed.',va='top',fontsize=11)
            fig.suptitle({'before':'Mean start-of-round target share','after':'Mean end-of-round target share','final':'Final-episode target share'}[mode],fontsize=18)
            save(fig,'target_share_'+mode)
        # All saved whole-cell estimates remain separate by physical cell.
        estimates=read_table('primary_estimates')
        estimates=estimates[estimates.target_fraction_bin_index.isna()] if 'target_fraction_bin_index' in estimates else estimates
        estimates=estimates.drop(columns=[c for c in coords if c!='cell_id' and c in estimates]).merge(cells[coords],on='cell_id',validate='many_to_one')
        estimates.to_parquet(out/'saved_whole_cell_estimates.parquet',index=False)
        for arm in ['truth','false']:
            f=estimates[estimates.arm.eq(arm)&estimates.metric.eq('round_target_actuation_cmi')].sort_values(['budget','epistemic_persistence','communication_profile'])
            if f.empty:continue
            assert not f.cell_id.duplicated().any()
            rows=[]
            for row in f.itertuples():
                values=[row.budget,row.epistemic_persistence,row.communication_profile,row.estimate,row.null_mean,row.null_std,row.estimate-row.null_mean,row.p_value,row.null_permutations]
                rows.append([str(x) if isinstance(x,str) else ('N/A' if pd.isna(x) else f'{x:.4g}') for x in values])
            fig,ax=plt.subplots(figsize=(13,max(8,len(rows)*.34+2)));ax.axis('off')
            table=ax.table(cellText=rows,colLabels=['b','rho','Profile','T [bits]','Null mean','Null SD','T - null','p','Draws'],loc='center',colWidths=[.05,.06,.20,.10,.10,.10,.10,.10,.07])
            table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.3 if len(rows)>14 else 2)
            ax.set_title(f'{arm.title()} control: saved whole-cell information and null comparisons\nUnadjusted permutation p-values; null SD is not an estimate confidence interval',fontsize=14)
            save(fig,'null_'+arm)
        derived=read_table('derived_observables')
        whole=derived[derived.target_fraction_bin_index.isna()] if 'target_fraction_bin_index' in derived else derived
        whole=whole.drop(columns=[c for c in coords if c!='cell_id' and c in whole]).merge(cells[coords],on='cell_id',validate='many_to_one')
        whole.to_parquet(out/'saved_derived_estimates.parquet',index=False)
        metrics=[('T_pi (whole-cell)','round_target_actuation_cmi',estimates),('eta_IF (whole-cell)','round_target_information_fraction',estimates),('Susceptibility chi (whole-cell)','round_target_susceptibility',estimates),('chi (occupancy weighted over x)','susceptibility_occupancy_weighted',whole)]
        metrics += [(m,m,whole) for m in sorted(whole.metric.unique()) if m.startswith('eta_') and m!='eta_ir_state_local']
        availability=[]
        for label,metric,frame in metrics:
            frame=frame[frame.metric.eq(metric)&frame.arm.ne('none')].copy()
            assert not frame.cell_id.duplicated().any(),metric
            availability.append(dict(metric=metric,rows=len(frame),finite=int(np.isfinite(pd.to_numeric(frame.estimate,errors='coerce')).sum())))
            fig,axes=plt.subplots(2,len(rhos),figsize=(13,8),squeeze=False,layout='constrained')
            for i,arm in enumerate(['truth','false']):
                for j,rho in enumerate(rhos):
                    ax=axes[i,j];has_finite=False
                    for profile,color in zip(profiles,colors):
                        f=frame[frame.arm.eq(arm)&frame.epistemic_persistence.eq(rho)&frame.communication_profile.eq(profile)].sort_values('budget')
                        if 'support_status' in f:f=f[~f.support_status.eq('unsupported')]
                        f=f.set_index('budget').reindex(budgets)
                        y=pd.to_numeric(f.estimate,errors='coerce')
                        has_finite=has_finite or bool(np.isfinite(y).any())
                        ax.plot(budgets,y,'.-',label=profile.replace('_',' '),color=color)
                        if 'ci_low' in f:
                            lo=pd.to_numeric(f.ci_low,errors='coerce');hi=pd.to_numeric(f.ci_high,errors='coerce')
                            valid=np.isfinite(y)&np.isfinite(lo)&np.isfinite(hi)
                            ax.vlines(np.array(budgets)[valid],lo[valid],hi[valid],color=color,alpha=.6)
                    ax.set(title=f'{arm} control, rho={rho}',xlabel='Budget b',ylabel=label,xticks=budgets);ax.set_xticklabels(budget_labels);ax.grid(alpha=.2)
                    if not has_finite:ax.text(.5,.5,'No supported finite saved estimate',transform=ax.transAxes,ha='center')
            axes[0,0].legend(fontsize=8)
            fig.suptitle(label+' — aggregated over observed states, separate rho panels\nSaved estimates and intervals; no pooling across rho, arms or profiles',fontsize=13)
            save(fig,'metric_'+metric)
        aggregate_metrics=['round_target_actuation_cmi','round_target_information_fraction','susceptibility_occupancy_weighted','eta_ir','eta_th','eta_th_signed','eta_th_bounded']
        aggregate_table=pd.concat([estimates[estimates.metric.isin(aggregate_metrics)],whole[whole.metric.isin(aggregate_metrics)]],ignore_index=True)
        aggregate_table.to_csv(out/'aggregated_over_x_metrics.csv',index=False)
        (out/'metric_availability.json').write_text(json.dumps(availability,indent=2))
        # Reshape saved state-local estimates; no estimator or bin recomputation.
        primary_local=read_table('primary_estimates')
        for metric,label,slug,table,diverging in [
            ('round_target_actuation_cmi','T_pi [bits]','T_pi_state_local',primary_local,False),
            ('round_target_susceptibility','chi','chi_state_local',primary_local,True),
            ('eta_ir_state_local','eta_IR','eta_ir_state_local',derived,False),
        ]:
            local=table[table.metric.eq(metric)&table.target_fraction_bin_index.notna()].copy()
            if local.empty:continue
            local=local.drop(columns=[c for c in coords if c!='cell_id' and c in local]).merge(cells[coords],on='cell_id',validate='many_to_one')
            supported=local[~local.support_status.eq('unsupported')] if 'support_status' in local else local
            values=pd.to_numeric(supported.estimate,errors='coerce')
            vmax=float(values.abs().max() if diverging else values.max())
            if not np.isfinite(vmax) or vmax<=0:vmax=1.0
            vmin=-vmax if diverging else 0
            for arm in ['truth','false']:
                fig,axes=plt.subplots(len(rhos),2,figsize=(12,8),squeeze=False,layout='constrained')
                for i,rho in enumerate(rhos):
                    for j,profile in enumerate(profiles):
                        ax=axes[i,j];f=local[local.arm.eq(arm)&local.epistemic_persistence.eq(rho)&local.communication_profile.eq(profile)].copy()
                        if 'support_status' in f:f.loc[f.support_status.eq('unsupported'),'estimate']=np.nan
                        assert not f.duplicated(['target_fraction_bin_index','budget']).any()
                        matrix=f.pivot(index='target_fraction_bin_index',columns='budget',values='estimate').reindex(index=range(8),columns=budgets)
                        cmap=plt.get_cmap('RdBu_r' if diverging else 'viridis').copy();cmap.set_bad('#dddddd')
                        im=ax.imshow(matrix.to_numpy(dtype=float),origin='lower',aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap)
                        ax.set(xticks=range(len(budgets)),xticklabels=budget_labels,yticks=range(8),yticklabels=[f'{(k+.5)/8:.2f}' for k in range(8)],xlabel='Budget b',ylabel='Target-share bin center',title=f'{profile}, rho={rho}')
                fig.colorbar(im,ax=axes,label='Saved state-local '+label)
                fig.suptitle(f'{arm} control: state-local {label} — gray is missing or unsupported')
                save(fig,slug+'_'+arm)
        if config.get('rho_aggregate_maps', False):
            aggregate_rows=[]
            for metric,label,table,diverging in [
                ('round_target_actuation_cmi','T_pi [bits]',primary_local,False),
                ('round_target_susceptibility','chi',primary_local,True),
                ('round_target_information_fraction','eta_IF',primary_local,False),
                ('eta_ir_state_local','eta_IR',derived,False),
                ('eta_th_state_local','eta_th (unavailable)',derived,True),
            ]:
                local=table[table.metric.eq(metric)&table.target_fraction_bin_index.notna()].copy()
                local=local.drop(columns=[c for c in coords if c!='cell_id' and c in local]).merge(cells[coords],on='cell_id',validate='many_to_one')
                matrices={}
                for arm in ['truth','false']:
                    for profile in profiles:
                        f=local[local.arm.eq(arm)&local.communication_profile.eq(profile)].copy()
                        assert not f.duplicated(['budget','target_fraction_bin_index','epistemic_persistence']).any()
                        if 'support_status' in f:f.loc[f.support_status.eq('unsupported'),'estimate']=np.nan
                        matrix=np.full((8,len(budgets)),np.nan)
                        for k in range(8):
                            for j,budget in enumerate(budgets):
                                rows=f[f.budget.eq(budget)&f.target_fraction_bin_index.eq(k)].set_index('epistemic_persistence').reindex(rhos)
                                values=pd.to_numeric(rows.estimate,errors='coerce').to_numpy(dtype=float)
                                valid=bool(np.isfinite(values).all())
                                value=float(values.mean()) if valid else np.nan
                                matrix[k,j]=value
                                record=dict(metric=metric,arm=arm,communication_profile=profile,budget=budget,target_fraction_bin_index=k,estimate=value,required_rhos=len(rhos),available_rhos=int(np.isfinite(values).sum()),weighting='equal rho weight; all rho required')
                                for rho,component in zip(rhos,values):record[f'estimate_rho_{rho}']=component
                                aggregate_rows.append(record)
                        matrices[arm,profile]=matrix
                values=np.concatenate([m.ravel() for m in matrices.values()]);finite=values[np.isfinite(values)]
                vmax=float(np.max(np.abs(finite)) if diverging else np.max(finite)) if len(finite) else 1.0
                if vmax<=0:vmax=1.0
                fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
                for i,arm in enumerate(['truth','false']):
                    for j,profile in enumerate(profiles):
                        ax=axes[i,j];matrix=matrices[arm,profile]
                        cmap=plt.get_cmap('RdBu_r' if diverging else 'viridis').copy();cmap.set_bad('#dddddd')
                        im=ax.imshow(matrix,origin='lower',aspect='auto',vmin=-vmax if diverging else 0,vmax=vmax,cmap=cmap)
                        ax.set(xticks=range(len(budgets)),xticklabels=budget_labels,yticks=range(8),yticklabels=[f'{(k+.5)/8:.2f}' for k in range(8)],xlabel='Budget b',ylabel='Target-share bin center',title=f'{arm} control, {profile}')
                        if not np.isfinite(matrix).any():ax.text(.5,.5,'No complete supported rho pair',transform=ax.transAxes,ha='center',bbox=dict(facecolor='white',alpha=.8))
                fig.colorbar(im,ax=axes,label='Equal-rho descriptive mean: '+label)
                fig.suptitle('RHO-AGGREGATED PHASE DIAGRAM: '+label+'\nEqual weights for '+', '.join(map(str,rhos))+'; gray if either rho missing/unsupported\nDescriptive average of saved estimates, not a pooled estimator; no new CI'+('\n* Extension budgets: different mixed-message setting; see overview' if config.get('extension_budgets') else ''),fontsize=12)
                save(fig,'rho_aggregated_'+metric)
            pd.DataFrame(aggregate_rows).to_csv(out/'rho_aggregated_phase_maps.csv',index=False)
        for rho in rhos:
            fig,axes=plt.subplots(len(panels),2,figsize=(13,3*len(panels)),layout='constrained')
            for i,(arm,budget) in enumerate(panels):
                for j,profile in enumerate(profiles):
                    f=r[r.arm.eq(arm)&r.budget.eq(budget)&r.epistemic_persistence.eq(rho)&r.communication_profile.eq(profile)]
                    if f.empty:
                        axes[i,j].text(.5,.5,'No saved cell',ha='center');axes[i,j].axis('off');continue
                    identities=f[['cell_id','episode_id']].drop_duplicates().sort_values(['cell_id','episode_id']);identity=identities.iloc[0]
                    selected=f[f.cell_id.eq(identity.cell_id)&f.episode_id.eq(identity.episode_id)].sort_values('round_index');ax=axes[i,j]
                    ax.plot(selected.round_index,selected.share_after,'.-',label='Target after' if arm!='none' else 'Truth after')
                    if arm=='false':ax.plot(selected.round_index,selected.truth_vote_share,'.--',label='Truth after')
                    ax.set(ylim=(0,1),xlabel='Recorded round',ylabel='Share',title=f'{arm}, b={budget}, {profile}\n{identity.episode_id}');ax.legend(fontsize=8);ax.grid(alpha=.2)
                    examples.append(dict(cell_id=identity.cell_id,episode_id=identity.episode_id))
            fig.suptitle(f'Illustrative episodes, rho={rho} — first sorted episode in each cell',fontsize=15)
            save(fig,'episodes_rho_'+str(rho))
    pd.concat(summaries).to_csv(out/'target_share_bars.csv',index=False)
    (out/'report.md').write_text('# '+config['title']+'\n\n'+'\n\n'.join(notes)+'\n\n'+'\n\n'.join(f'![{p}](figures/{p}.png)' for p in pages))
    paths=['tables/cells.parquet','tables/episodes.parquet','tables/rounds.parquet','tables/primary_estimates.parquet','tables/derived_observables.parquet','validation.json']
    manifest=dict(source=str(source),source_sha256={str(root/p):hashlib.sha256((root/p).read_bytes()).hexdigest() for root in sources for p in paths},sources=[str(root) for root in sources],validation=v,pages=pages,examples=examples,estimator_calls=0,resampling_calls=0,renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out/'report_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Built {len(pages)} pages: {out / "report.pdf"}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);build(p.parse_args().config)
