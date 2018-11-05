import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import json

def attention_plot(x, attention_value, step):
    axlist = []
    for i in range(np.shape(x)[0]):
        fig = plt.figure(figsize=(8,12))
        axlist.append(fig.add_subplot(np.shape(x)[0],1,i+1))
        df = pd.DataFrame({'token' : devectorize([l.argmax() for l in x[i]]),
                      'line' : i,
                      'attention' : attention_value[i]
                      })
        #print(df)
        df_attention_pivot = pd.pivot_table(data=df,
                                       values='attention',
                                       columns='token',
                                       index='line',
                                       aggfunc=np.mean)
        #sprint(df_attention_pivot)
        heatmap = sns.heatmap(
                df_attention_pivot,
                cbar=False,
                annot=True,
                cmap='Reds',
                ax = axlist[i],
                linewidths=.5,
                square=False,
                fmt="1.1f",
            )
    pdf = PdfPages('./yelp/img/img_{}.pdf'.format(step))
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf.savefig()
    pdf.close()

def devectorize(program):
    g = open('./trans_data/src/data.json', 'r')
    token_vector = json.load(g)
    program_lines = []
    for token in program:
        program_lines.append([key for key, value in token_vector.items() if value == token][0])
    return program_lines