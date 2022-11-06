import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_measure_learning_relearning(df_info, ms, unit, dct_repl, savefig=False, **kwargs):
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 5), sharey=True)
    fig.suptitle(ms)
    plt.subplots_adjust(wspace=0.1, hspace=0)

    for i, fname_i in enumerate(['MWM', 'MWM-Rev']):
        df_i = df_info[df_info['fname']==fname_i]
        sns.lineplot(
            x='days_adjusted',
            y=ms,
            hue='treatment',
            data=df_i,
            ax=ax[i],
            **kwargs)
        ax[i].tick_params(axis='x', rotation=90)
        #print(df_i['day'].iloc[0])
        #ax[i].set_xticklabels('Day ' + str(df_i['day'].iloc[0]+3))

        ax[i].spines.right.set_visible(False)
        ax[i].spines.top.set_visible(False)
        ax[i].set_xlabel('')
        fig.subplots_adjust(bottom=0.3)

        if fname_i == 'MWM':
            ax[i].set_title('learning')
        elif fname_i == 'MWM-Rev':
            ax[i].set_title('reversal learning')
      #  ax[i].set_ylim(0, 0.7)

    ax[0].set_ylabel(unit)

    sns.move_legend(ax[0], "lower left")
    leg = ax[0].get_legend()
    leg.set_title('')
    ls_str_new = []
    for text_i in leg.texts:
        str_new = dct_repl[text_i.get_text()]
        text_i.set_text(str_new)
    plt.draw()
    
    ax[1].get_legend().remove()

    if savefig:
        fig.savefig('figures/measures/'+ms.replace(' ', '_') +'.pdf')
        
    return fig, ax

def plot_trajectory(
    xy,
    xy_pltfrm,
    r_pltfrm=0.05,
    r_pool=0.6,
    xy_pltfrm_old=None,
    r_pltfrm_old=0.05,
    fac_scale=1.1,
    kw_pltfrm=None,
    kw_pltfrm_old=None,
    kw_pool=None,
    kw_plot=None,
    kw_fig=None,
    fname=None,
    display=True,
    flip_y=False):

    if kw_fig is None:
        kw_fig = {
            'figsize': (4,4)
        }
    # create figure
    fig, ax = plt.subplots(1, **kw_fig)

    # scale size
    ax.set_xlim(-r_pool*fac_scale, r_pool*fac_scale)
    ax.set_ylim(-r_pool*fac_scale, r_pool*fac_scale)

    # turn axis off
    #ax.axis('off')

    # plot trajectory
    if kw_plot is None:
        kw_plot = {}
    ax.plot(xy[:,0], xy[:,1], **kw_plot)
    
    # create additional axis for plotting platform and enclosure
    ax_dupl = ax.twinx()
    #ax_dupl.axis('off')
    ax_dupl.set_aspect('equal')

    #ax_dupl.set_aspect('equal', adjustable='box')
    ax_dupl.set_xlim(-r_pool*fac_scale, r_pool*fac_scale)
    ax_dupl.set_ylim(-r_pool*fac_scale, r_pool*fac_scale)

    ls_patches = []
    # create platform patch
    if kw_pltfrm is None:
        kw_pltfrm = {
            'linewidth': 0,
            'facecolor': 'r'
        }
    
    ell_pltfrm = patches.Ellipse(
        (xy_pltfrm[0], xy_pltfrm[1]),
        r_pltfrm*2, r_pltfrm*2,
        **kw_pltfrm)
    ls_patches.append(ell_pltfrm)

    # create platform old patch if desired
    if xy_pltfrm_old is not None:
        if kw_pltfrm_old is None:
             kw_pltfrm_old = {
                'linewidth': 1,
                'linestyle': 'dotted',
                'edgecolor': 'r',
                'facecolor': (0,0,0,0)
            }
            
        ell_pltfrm_old = patches.Ellipse(
            (xy_pltfrm_old[0], xy_pltfrm_old[1]),
            r_pltfrm_old*2., r_pltfrm_old*2.,
            **kw_pltfrm_old)
        ls_patches.append(ell_pltfrm_old)


    # create ring around enclosure
    if kw_pool is None:
        kw_pool = {
            'linewidth': 4,
            'edgecolor': 'k',
            'facecolor': 'none'
        }
    ell_pool = patches.Ellipse(
        (0, 0),
        r_pool*2, r_pool*2,
        **kw_pool
        )
    ls_patches.append(ell_pool)
    
    # add patches to plot
    for patch_i in ls_patches:
        ax_dupl.add_patch(patch_i)

    if flip_y:
        ax_dupl.invert_yaxis()
        ax.invert_yaxis()
    
    if fname is not None:
        fig.savefig(fname)
    
    if display:
        plt.show()
    else:
        plt.close()
    return fig, ax, ax_dupl

