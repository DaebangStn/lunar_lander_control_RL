import matplotlib.pyplot as plt


def plot_df(df, chart_name, title, x_label, y_label):
    plt.rcParams.update({'font.size': 17})
    df['rolling_mean'] = df[df.columns[0]].rolling(100).mean()
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()

    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_title(title)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)

    fig = plot.get_figure()
    plt.legend().set_visible(False)
    fig.savefig(chart_name + '.png')


def plot_df2(df, chart_name, title, x_label, y_label, y_limit):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()

    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_title(title)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    plt.ylim(y_limit)

    fig = plot.get_figure()
    fig.savefig(chart_name + '.png')


def plot_experiments(df, chart_name, title, x_label, y_label, y_limit):
    plt.rcParams.update({'font.size': 17})
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()

    plot = df.plot(linewidth=1, figsize=(15, 8))
    plot.set_title(title)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    plt.ylim(y_limit)

    fig = plot.get_figure()
    fig.savefig(chart_name + '.png')
