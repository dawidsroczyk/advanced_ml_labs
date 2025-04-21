import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_data(data):
    num_classifiers = data['Bagging']['Train Accuracy'].__len__()
    nc_copy = num_classifiers
    num_classifiers = range(1, num_classifiers+1)

    plot_data = []
    for method in ['Bagging', 'AdaBoost', 'XGBoost', 'Random Forest']:
        for nc in num_classifiers:
            plot_data.append({
                'Method': method,
                'Number of Classifiers': nc,
                'Accuracy': data[method]['Test Accuracy'][nc-1],
                'Type': 'Test'
            })
            plot_data.append({
                'Method': method,
                'Number of Classifiers': nc,
                'Accuracy': data[method]['Train Accuracy'][nc-1],
                'Type': 'Train'
            })

    st_acc = data['Single tree']['Test Accuracy'][0]
    plot_data.append({
        'Method': 'Single Tree',
        'Number of Classifiers': 1,
        'Accuracy': st_acc,
        'Type': 'Test'
    })

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    g = sns.lineplot(data=df[df['Method'] != 'Single Tree'], 
                    x='Number of Classifiers', 
                    y='Accuracy', 
                    hue='Method', 
                    style='Type',
                    markers=True,
                    dashes=False,
                    linewidth=2.5,
                    markersize=10)

    plt.axhline(
        y=st_acc, 
        color='gray', 
        linestyle='--', 
        linewidth=2, 
        label='Single Tree Test Accuracy'
        )

    plt.xlabel('Number of Classifiers', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(1, nc_copy + 1))
    plt.ylim(0, 1.1)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.ylim([0, 1])

def plot_data_separate(data, ylim=[0, 1], main_title=None):
    num_classifiers = data['Bagging']['Train Accuracy'].__len__()
    nc_copy = num_classifiers
    num_classifiers = range(1, num_classifiers+1)

    plot_data = []
    for method in ['Bagging', 'AdaBoost', 'XGBoost', 'Random Forest']:
        for nc in num_classifiers:
            plot_data.append({
                'Method': method,
                'Number of Classifiers': nc,
                'Accuracy': data[method]['Test Accuracy'][nc-1],
                'Type': 'Test'
            })
            plot_data.append({
                'Method': method,
                'Number of Classifiers': nc,
                'Accuracy': data[method]['Train Accuracy'][nc-1],
                'Type': 'Train'
            })

    st_acc = data['Single tree']['Test Accuracy'][0]
    df = pd.DataFrame(plot_data)

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    if main_title:
        fig.suptitle(main_title, fontsize=16, y=1.02)

    sns.lineplot(data=df[(df['Method'] != 'Single Tree') & (df['Type'] == 'Train')], 
                x='Number of Classifiers', 
                y='Accuracy', 
                hue='Method', 
                style='Type',
                markers={'Train': 'X'},
                dashes=False,
                linewidth=2.5,
                markersize=10,
                ax=ax1)
    ax1.set_title('Train Accuracy')
    ax1.set_xlabel('Number of Classifiers', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_xticks(range(1, nc_copy + 1))
    ax1.set_ylim(ylim)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    sns.lineplot(data=df[(df['Method'] != 'Single Tree') & (df['Type'] == 'Test')], 
                x='Number of Classifiers', 
                y='Accuracy', 
                hue='Method', 
                style='Type',
                markers={'Test': 'o'},
                dashes=False,
                linewidth=2.5,
                markersize=10,
                ax=ax2)
    ax2.axhline(y=st_acc, color='gray', linestyle='--', linewidth=2, label='Single Tree Test Accuracy')
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Number of Classifiers', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_xticks(range(1, nc_copy + 1))
    ax2.set_ylim(ylim)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()