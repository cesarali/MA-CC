"""Finite-horizon rho-by-budget maps for every sealed v4 physical cell."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from .config import load_config


def map_data(config):
    root=config.results_dir
    base=pd.read_csv(root/"summaries"/"cell_metric_summary.csv")
    base=base.rename(columns={"estimate":"value"})
    base["view"]="physical"
    path=root/"information"/"llm_parallel"/"round_information_estimates.parquet"
    if path.exists():
        info=pd.read_parquet(path)
        info=info.loc[info.scope=="pooled"]
        coords=base.drop_duplicates("cell_id")[["cell_id","controller_target","beta_regime",
            "rho","budget","q","q_c","independent_episodes"]]
        expanded=[]
        for field,label in (("estimate","raw"),("null_mean","null"),
                            ("estimate_minus_null","excess")):
            frame=info[["cell_id","statistic",field]].rename(columns={"statistic":"metric",field:"value"})
            frame=frame.merge(coords,on="cell_id",how="left")
            frame["view"]=label
            frame["window"]="all_60_transitions"
            expanded.append(frame)
        base=pd.concat([base,*expanded],ignore_index=True)
    return base


def plot_maps(config_path:str,output:str|None=None,metric:str|None=None)->dict:
    config=load_config(config_path)
    data=map_data(config)
    target=Path(output) if output else config.results_dir/"plots"/"live_board_maps.pdf"
    target.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(target.with_suffix(".csv"),index=False)
    regimes=list(dict.fromkeys(cell.beta_regime for cell in config.cells))
    targets=list(dict.fromkeys(cell.params.controller_target for cell in config.cells))
    rhos=sorted({cell.params.rho for cell in config.cells})
    budgets=sorted({cell.params.budget for cell in config.cells})
    q_values=sorted({cell.params.q for cell in config.cells})
    qc_values=sorted({round(cell.params.sensing_fraction*cell.params.N) for cell in config.cells})
    pages=0
    with PdfPages(target) as pdf:
        for (name,view),group in data.groupby(["metric","view"],sort=False):
            if metric and name!=metric:continue
            if group.value.notna().sum()==0:continue
            values=group.value.to_numpy(dtype=float)
            if view=="excess":
                limit=np.nanmax(np.abs(values));vmin,vmax=-limit,limit;cmap="RdBu_r"
            elif name.endswith("share") or name.endswith("fraction") or "majority" in name:
                vmin,vmax=0,1;cmap="viridis"
            else:
                vmin,vmax=np.nanmin(values),np.nanmax(values)
                if vmin==vmax:vmax=vmin+1e-9
                cmap="viridis"
            for regime in regimes:
                for controller_target in targets:
                    subset=group.loc[(group.beta_regime==regime)&
                                     (group.controller_target==controller_target)]
                    if subset.empty:continue
                    fig,axes=plt.subplots(len(q_values),len(qc_values),figsize=(11,8.5),
                                          squeeze=False,constrained_layout=True)
                    image=None
                    for i,q in enumerate(q_values):
                        for j,q_c in enumerate(qc_values):
                            ax=axes[i,j]
                            chosen=subset.loc[(subset.q==q)&(subset.q_c==q_c)]
                            matrix=np.full((len(rhos),len(budgets)),np.nan)
                            for row in chosen.itertuples():
                                matrix[rhos.index(row.rho),budgets.index(row.budget)]=row.value
                            image=ax.imshow(np.ma.masked_invalid(matrix),origin="lower",aspect="auto",
                                            vmin=vmin,vmax=vmax,cmap=cmap)
                            ax.set(title=f"q={q}, q_c={q_c}",xticks=range(len(budgets)),
                                   yticks=range(len(rhos)))
                            ax.set_xticklabels(budgets,fontsize=7)
                            ax.set_yticklabels([f"{rho:g}" for rho in rhos],fontsize=7)
                            if i==len(q_values)-1:ax.set_xlabel("posts per active round B")
                            if j==0:ax.set_ylabel("persistence rho")
                    fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.7)
                    n=subset.independent_episodes.dropna().unique()
                    n_label=",".join(str(int(x)) for x in n) if len(n) else "unknown"
                    fig.suptitle(f"{name} ({view}); {regime}; target={controller_target:+d}\n"
                                 f"finite horizon 60 rounds; independent episodes/cell={n_label}; "
                                 "blank=unavailable",fontsize=11)
                    pdf.savefig(fig)
                    plt.close(fig)
                    pages+=1
    return {"pdf":str(target),"plotted_data_csv":str(target.with_suffix(".csv")),
            "pages":pages}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--output")
    parser.add_argument("--metric")
    args=parser.parse_args()
    print(plot_maps(args.config,args.output,args.metric))


if __name__=="__main__":main()
