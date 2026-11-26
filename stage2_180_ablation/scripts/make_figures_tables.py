import os,sys
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'work/matplotlib_cache')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
# Plot dependencies are separate; the numerical experiment environment stays frozen.
if (ROOT/'work/plot_packages').exists():sys.path.append(str(ROOT/'work/plot_packages'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_model import GROUPS

BLUE='#0072B2'; ORANGE='#D55E00'; GREEN='#009E73'; PURPLE='#CC79A7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/'figures'/f'{name}.png',bbox_inches='tight',dpi=180)
    plt.close(fig)

def paired_dot(ax,data,order,labels):
    subset=data.set_index('experiment_config').loc[order]
    y=np.arange(len(subset))
    for offset,prefix,color,marker,label in [(-.11,'series',BLUE,'o','Series average'),(.11,'dataset',ORANGE,'s','Dataset average')]:
        val=subset[prefix+'_average'].to_numpy()
        ci='series_cluster_bootstrap_ci95_' if prefix=='series' else 'dataset_bootstrap_ci95_'
        lo=subset[ci+'low'].to_numpy();hi=subset[ci+'high'].to_numpy()
        ax.errorbar(val,y+offset,xerr=np.array([val-lo,hi-val]),fmt=marker,color=color,ms=3,lw=.8,capsize=2,label=label)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('VUS-PR');ax.set_xlim(0,1)

def main():
    (OUT/'figures').mkdir(exist_ok=True);(OUT/'tables').mkdir(exist_ok=True)
    a=pd.read_csv(OUT/'s2a_core_ablation_summary.csv');b=pd.read_csv(OUT/'s2b_summary.csv')
    av=a[a.metric=='VUS-PR'];order=GROUPS['s2a_core_ablation_180.csv']
    labels=['Branch A','Branch B','Raw fusion*','Std-scaled fusion*','ECDF, no smoothing*','ECDF, smoothing*','POT, no smoothing','Full DB-ESC']
    fig,ax=plt.subplots(figsize=(6.7,3.8));paired_dot(ax,av,order,labels)
    ax.legend(loc='lower right',frameon=False)
    ax.set_title('Core ablation: 180 series, 17 datasets',loc='left',fontsize=9)
    fig.text(.03,-.02,'95% dataset-cluster bootstrap intervals; *Branch-B Stage-1 POT retained.',fontsize=7)
    save(fig,'s2a_core_ablation')
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.8),layout='constrained')
    for i,((filename,order),ax) in enumerate(zip(list(GROUPS.items())[1:],axes.flat)):
        sub=b[(b.group==filename.replace('_180.csv',''))&(b.metric=='VUS-PR')]
        label=[x.replace('PCA Orthogonal Residual','PCA residual').replace('FFT Period-Adaptive Window','FFT adaptive').replace('Fixed Window = ','Fixed W=').replace('No Stage-1 and Stage-2 POT','No Stage-1 or Stage-2') for x in order]
        paired_dot(ax,sub,order,label)
        ax.set_title(chr(97+i)+'  '+['Covariance scales','Branch-B POT stages','Temporal window','Branch-A components'][i],loc='left',fontsize=9)
    fig.suptitle('Internal ablations: 180 series, 17 datasets\n95% dataset-cluster bootstrap intervals',fontsize=9)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    save(fig,'s2b_internal_ablation')
    v=pd.read_csv(OUT/'s2c_real_branch_variance_180.csv')
    masks=[('Clean-160',v.clean160,BLUE,'o'),('Other 20',~v.clean160,ORANGE,'^')]
    fig,axes=plt.subplots(1,3,figsize=(9,3.05),layout='constrained')
    for ax,col,title in zip(axes,['general_covariance_formula','equal_variance_approximation','shrunk_equal_variance_approximation'],
                            ['General covariance identity','Equal-variance approximation','Shrunk-correlation approximation']):
        finite=v[[col,'actual_fusion_variance']].dropna()
        for label,mask,color,marker in masks:
            g=v[mask].dropna(subset=[col,'actual_fusion_variance'])
            ax.scatter(g.actual_fusion_variance,g[col],s=10,alpha=.6,color=color,marker=marker,label=label)
        maximum=float(max(finite.actual_fusion_variance.max(),finite[col].max()))
        ax.plot([0,maximum],[0,maximum],color='black',ls='--',lw=.7)
        ax.set_xscale('symlog',linthresh=.1);ax.set_yscale('symlog',linthresh=.1)
        ax.set_xlabel('Measured fusion variance');ax.set_ylabel('Formula / approximation variance')
        ax.set_title(title+'\nn='+str(len(finite)),fontsize=8)
    axes[0].legend(frameon=False,loc='upper left')
    save(fig,'s2c_measured_vs_theoretical_variance')
    fig,ax=plt.subplots(figsize=(6.8,3.5),layout='constrained')
    datasets=sorted(v.dataset.unique()); vals=[];plotlabels=[]
    for ds in datasets:
        g=v[v.dataset==ds].var_ratio_a_over_b
        valid=g[(g>0)&np.isfinite(g)]
        vals.append(valid.to_numpy());plotlabels.append(f'{ds}\n{len(valid)}/{len(g)}')
    ax.boxplot(vals,tick_labels=plotlabels,widths=.55,showfliers=False,medianprops={'color':ORANGE})
    rng=np.random.default_rng(20261126)
    for i,values in enumerate(vals,1):ax.scatter(i+rng.uniform(-.15,.15,len(values)),values,s=7,color=BLUE,alpha=.4)
    ax.axhline(1,color='black',ls='--',lw=.7);ax.axhspan(.5,2,color=GREEN,alpha=.1)
    ax.set_yscale('log');ax.set_ylabel('Var(branch A) / Var(branch B)');ax.tick_params(axis='x',rotation=65)
    ax.set_title('Branch variance ratios; labels show valid / total series',loc='left',fontsize=9)
    save(fig,'s2c_branch_variance_ratio_boxplot')
    fig,axes=plt.subplots(1,2,figsize=(7,3.1),layout='constrained')
    for label,mask,color,marker in masks:
        g=v[mask].dropna(subset=['pearson_unshrunk','variance_reduction_percent'])
        axes[0].scatter(g.pearson_unshrunk,g.variance_reduction_percent,s=12,alpha=.6,color=color,marker=marker,label=label)
        axes[1].scatter(g.variance_reduction_percent,g['fusion_gain_over_mean_branch_VUS-PR'],s=12,alpha=.6,color=color,marker=marker)
    axes[0].axvline(0,color='gray',lw=.6);axes[0].axhline(0,color='gray',lw=.6)
    axes[0].set_xlabel('Unshrunk prefix Pearson correlation');axes[0].set_ylabel('Measured variance reduction (%)')
    axes[0].set_title('Correlation and variance reduction',fontsize=9);axes[0].legend(frameon=False)
    axes[1].axhline(0,color='gray',lw=.6);axes[1].set_xlabel('Measured variance reduction (%)')
    axes[1].set_ylabel('VUS-PR gain over mean branch')
    axes[1].set_title('Unsmoothed fusion vs. matched branches',fontsize=9)
    save(fig,'s2c_correlation_variance_performance')
    # Tables use standard AP throughout; frozen reference values are never copied.
    wide=a.pivot(index='experiment_config',columns='metric',values='series_average').loc[GROUPS['s2a_core_ablation_180.csv']]
    ds=av.set_index('experiment_config').dataset_average
    lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Method & Ser. VUS-PR & DS. VUS-PR & AP \\',r'\midrule']
    for config,row in wide.iterrows():
        label=config.replace('A+B ','').replace('without Smoothing','w/o smoothing').replace('with Smoothing','w/ smoothing')
        label=label.replace('&',r'\&')
        lines.append(f'{label} & {row["VUS-PR"]:.4f} & {ds[config]:.4f} & {row["AUC-PR"]:.4f} '+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'% AP: sklearn Average Precision with grouped ties. Branch-B Stage-1 POT remains active in all S2-A controls.'])
    (OUT/'tables/s2a_core_ablation.tex').write_text('\n'.join(lines),encoding='utf-8')
    for filename,configs in list(GROUPS.items())[1:]:
        group=filename.replace('_180.csv','');sub=b[b.group==group]
        pivot=sub.pivot(index='experiment_config',columns='metric',values='series_average').loc[configs]
        ds=sub[sub.metric=='VUS-PR'].set_index('experiment_config').dataset_average
        lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Configuration & Ser. VUS-PR & DS. VUS-PR & AP \\',r'\midrule']
        for config,row in pivot.iterrows():
            label=config.replace('{',r'\{').replace('}',r'\}').replace('&',r'\&')
            lines.append(f'{label} & {row["VUS-PR"]:.4f} & {ds[config]:.4f} & {row["AUC-PR"]:.4f} '+r'\\')
        lines.extend([r'\bottomrule',r'\end{tabular}'])
        (OUT/'tables'/f'{group}.tex').write_text('\n'.join(lines),encoding='utf-8')
    vs=pd.read_csv(OUT/'s2c_variance_summary.csv')
    lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Quantity (median [IQR]) & All 180 & Clean 160 \\',r'\midrule']
    for field,label in [('var_ratio_a_over_b',r'$\mathrm{Var}(z_A)/\mathrm{Var}(z_B)$'),
                        ('pearson_unshrunk',r'Pearson $r$'),('variance_reduction_percent',r'Variance reduction (\%)'),
                        ('equal_approximation_relative_error',r'Equal-variance relative error')]:
        values=[]
        for cohort in ['All-180','Clean-160']:
            row=vs[(vs.cohort==cohort)&(vs.quantity==field)].iloc[0]
            if field=='equal_approximation_relative_error':
                values.append(f'{row["median"]:.2e} [{row.q25:.2e}, {row.q75:.2e}]')
            else:
                values.append(f'{row["median"]:.3f} [{row.q25:.3f}, {row.q75:.3f}]')
        lines.append(label+' & '+' & '.join(values)+' '+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (OUT/'tables/s2c_variance_audit.tex').write_text('\n'.join(lines),encoding='utf-8')
    print('Saved vector PDFs, PNG previews and six LaTeX tables.',flush=True)

if __name__=='__main__':main()
