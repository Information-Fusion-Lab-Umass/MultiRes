import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


# *Code for creating attention heat-maps*
def make_heatmaps(model, tokens, attention_values, cmap_val='OrRd', words_or_sent=True, fig_size_v=10,
                  model_name='Model'):
    # model is just the name of the model for which you are creating the figure
    # cmap example value: GnBu or OrRd
    # tokens: For your use-cases it would just be the time step values or indices
    models = [model]
    attention_values = np.array([attention_values])
    fig, ax = plt.subplots()
    im = ax.imshow(attention_values, cmap=cmap_val)

    # to show all ticks...
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(models)))
    # label them with the respective list entries
    ax.set_xticklabels(tokens, fontsize=14)
    ax.set_yticklabels(models)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')

    ax.set_title(model_name)
    if (words_or_sent):
        fig.set_size_inches(fig_size_v, 2)
    else:
        fig.set_size_inches(7, 2)
    plt.savefig('attn')


if __name__ == '__main__':
    n = 10
    make_heatmaps(model='Physio', tokens=np.arange(n), attention_values=np.random.rand(n))

