import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.ticker as mticker


def is_ndarray(data):
    return isinstance(data, np.ndarray)


def multi_class_data_distplot(data: list,
                              data_label: list,
                              data_name: str = 'x',
                              label_name: str = 'Label',
                              bins: int = 50,
                              y_name: str = 'Frequency',
                              sns_theme: str = 'ticks',
                              keep_right_top_outline: bool = True,
                              is_export: bool = False,
                              export_path: str = './figure.png',
                              ax=None):
    # data, a list with element whose type is list.
    # each element is data with the same class
    # an example: data=[[1,2,3],[2,3,4],[3,4,5]] data_label=['dataA','dataB', 'dataC']
    # data_name='Cosine Similarity' label_name='Label'
    if len(data) != len(data_label):
        AssertionError('the class number of data does not match the number of labels.')
    data_list = []
    label_list = []
    for each_class, each_label in zip(data, data_label):
        if is_ndarray(each_class):
            data_list.append(each_class.reshape(-1))
        else:
            data_list.append(np.array(each_class).reshape(-1))
        label_list.extend([each_label] * len(each_class))

    data = np.concatenate(data_list, axis=0)
    df = pd.DataFrame({data_name: data, label_name: label_list})
    # print(df)
    sns.set_theme(style=sns_theme, font_scale=1.6)
    sns.histplot(data=df, x=data_name, hue=label_name, bins=bins, legend=False, ax=ax)
    if ax is None:
        plt.legend(loc='best', labels=data_label)
    else:
        ax.set_ylabel(y_name, fontproperties='Times New Roman', fontsize=20)
        ax.set_xlabel(None)
        ax.set_title(data_name, fontproperties='Times New Roman', fontsize=20)
        legend_font = {"family": "Times New Roman"}
        ax.legend(loc='best',
                  labels=['EARNIE', 'w/o container', 'LexRank', 'TextRank'],
                  fontsize=15,
                  prop=legend_font)
        ax.set_xlim(0.1, 0.5)
        label_format = '{:,.1f}'
        xlabels = ax.get_xticks().tolist()
        # print(xlabels)
        ax.xaxis.set_major_locator(mticker.FixedLocator(xlabels))  # 定位到散点图的x轴
        # print(label_format.format(0.3))
        ax.set_xticklabels([label_format.format(x) for x in xlabels],
                           fontproperties='Times New Roman',
                           fontsize=20)  # 使用列表推导式循环将刻度转换成浮点数

        label_format = '{:,.0f}'
        ylabels = ax.get_yticks().tolist()
        ax.yaxis.set_major_locator(mticker.FixedLocator(ylabels))  # 定位到散点图的y轴
        ax.set_yticklabels([label_format.format(y) for y in ylabels],
                           fontproperties='Times New Roman',
                           fontsize=20)  # 使用列表推导式循环将刻度转换成浮点数

    if keep_right_top_outline:
        sns.despine(top=False, right=False)

    if is_export:
        plt.savefig(export_path, dpi=400)
    elif ax is None:
        plt.show()
    else:
        return


if __name__ == '__main__':
    # a test
    data1 = np.random.randn(2000) + 5
    data2 = np.random.randn(2000) + 3
    data3 = np.random.randn(2000) + 1
    data4 = np.random.randn(2000) + 0
    data_name = 'Cosine Similarity'
    data_label = ['TTRep', 'TextRank', 'LexRank', 'w/o container']
    data = [data1, data2, data3, data4]
    multi_class_data_distplot(data=data,
                              data_label=data_label,
                              data_name='Cosine Similarty',
                              label_name='Method',
                              is_export=True)
