"""
Plotting
This script contains functions for
1) Plotting results of statistical test
    - plot_test()
2) Plotting final thesis figures
    - final_figure()
    - correlation_figure()
3) Plotting figure subplots
    - plot_extracted_spikes()
    - plot_frames_lfp()
    - plot_frames_MUAe()
    - plot_cors()
    - plot_inter_cors()
    - plot_pca()
    - plot_spont_map()
    - plot_ori_map()
    - plot_hist()

Authors: Matěj Voldřich
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import helper
import quantities as pq


def plot_test(params, tag):
    """
    Plot and save generated spontaneous map, orientation map, test results for
    each (interpolated) channel and overall test result.

    :param params: analysis parameters
    :param tag: test result tag
    :return:
    """
    respath = os.path.join(params['results_path'], f"testing_results_{tag}_{params['remove_PCA_dims']}.pkl")
    with open(respath, "rb") as f:
        result = pickle.load(f)

    if params["control_type"] == "orientation":
        orientations = result["test"]
        spont_map = result["reference"]
    else:
        orientations = result["reference"]
        spont_map = result["test"]

    monkey = tag[0]
    plt.figure(figsize=(13, 12))
    # spontaneous map
    plt.suptitle(f"Testing results {tag}")
    cmap = params['map_colorscheme']
    cmap = plt.get_cmap(cmap)
    cmap.set_under('lightgray')
    cmap.set_over('lightgray')
    plt.subplot(2, 2, 1)
    plt.title(f"Spontaneous map (PCA dims [{params['remove_PCA_dims']}, {params['remove_PCA_dims'] + 1}])")
    plt.pcolormesh(spont_map, cmap=cmap, vmin=0)

    # orientations
    plt.subplot(2, 2, 2)
    plt.title(f"{monkey}{result['Array_ID']} orientations")
    plt.pcolormesh(orientations, cmap=cmap, vmin=0)

    # percentiles
    plt.subplot(2, 2, 3)
    plt.title(f"Scores percentile ({len(result['control_scores'])} controls)")
    cmap = plt.get_cmap("RdBu")
    cmap.set_under('lightgray')
    cmap.set_over('lightgray')
    plt.pcolormesh(result['percentiles'], cmap=cmap, vmin=0, vmax=100)

    # histogram
    plt.subplot(2, 2, 4)
    plt.title(f"Mean score ({result['percentile']})")
    h = plt.hist(result['control_scores'], bins=int(params['permutations'] / 20))[0]
    map_score = result['test_score']
    g = plt.plot([map_score, map_score], [0, 20], color="orange")[0]
    x, y = helper.KDE(result['control_scores'], 1)
    y = y / np.max(y) * np.max(h)
    plt.plot(x, y)
    plt.legend([g], ["Mean test score"])

    plt.savefig(
        f"{params['figures_path']}/testing/{result['method']}/{monkey}{result['Array_ID']}_{result['method']}_{params['remove_PCA_dims']}_{params['control_type'][0]}")
    plt.close()


# Figures in detailed analysis of L13
# figure a) - signal, spikes, frames (MUA and nLFP)
def plot_extracted_spikes(bl, fs, binsize, dur=1):
    """
    Plot frame extraction for MUA or nLFP
    :return:
    """
    # SETUP
    chn_ix = np.random.randint(64)
    t_start = 33

    y = bl.segments[0].analogsignals[0][t_start * fs:int((t_start + dur) * fs), chn_ix]
    y = y / max(y.max(), np.abs(y.min()))
    t_start = t_start * pq.s

    # 1. plot signal with threshold
    offset = 3.5
    signal = y + offset
    x = np.linspace(0, len(signal) / fs, len(signal))
    plt.plot(x, signal, color="black")
    plt.plot([0, x[-1]], [np.mean(signal), np.mean(signal)], color="grey")
    # plt.plot([0, x[-1]], [y_th, y_th], color="red", linestyle="dashed")

    # 2. extracted spikes
    st = bl.segments[0].spiketrains[chn_ix]
    plt.plot([x[0], x[-1]], [2, 2], color="black")
    for spike in st:
        if spike < t_start or spike > (t_start + dur * pq.s):
            continue
        spike = spike - t_start
        plt.plot([spike, spike], [1.6, 2.4], color="black", linewidth=1)

    # 3. convert to frames
    frames = np.zeros(int(x[-1] / binsize))
    for i in range(len(frames)):
        frames[i] = len(np.where(np.logical_and((st - t_start) > (i * binsize), (st - t_start) < (i + 1) * binsize))[0])
    x = np.array(list(range(len(frames) + 1))) / len(frames) * x[-1]
    plt.plot([x[0], x[-1]], [1, 1], color="black")
    plt.plot([x[0], x[-1]], [0, 0], color="black")
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, 1], color="black")
        if i < len(frames):
            plt.annotate(str(frames[i].astype(int)), (x[i] + 0.05, 0.4), size=30)
    plt.ylabel("Frames           Spikes           Signal   ", fontsize=18)
    plt.xticks([])
    plt.yticks([])


def plot_frames_lfp(bl, fs, dt, dur=1):
    # SETUP
    chn_ix = np.random.randint(64)
    t_start = np.random.randint(60)

    y = bl.segments[0].analogsignals[0][t_start * fs:int((t_start + dur) * fs), chn_ix]
    signal = y / max(y.max(), np.abs(y.min()))

    from scipy.signal import welch
    fs = 500
    # 1. raw signal with threshold
    offset = 3.5
    signal = signal + offset
    x = np.linspace(0, len(signal) / fs, len(signal))
    plt.plot(x, signal, color="black")
    plt.plot([0, x[-1]], [offset, offset], color="grey")

    # 2. plot spectrogram
    frames = []
    for i in range(int(x[-1] / dt)):
        signal_part = signal[int(i * dt * fs):int((i + 1) * dt * fs)]
        frames.append(welch(signal_part.reshape(signal_part.shape[0]), fs=fs)[1][:6])
    frames = np.array(frames)
    # 1.1 to 2.4
    cmap = plt.get_cmap("viridis")
    frames = frames - np.min(frames)
    frames = frames / np.max(frames)
    height = (2.4 - 1.3) / frames.shape[1]
    for i in range(frames.shape[0]):
        for j in range(frames.shape[1]):
            x1 = i * dt
            x2 = (i + 1) * dt
            y1 = j * height + 1.3
            y2 = (j + 1) * height + 1.3
            plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], color=cmap(frames[i, j]))

    # 3. convert to frames (total power)
    frames = np.zeros(int(x[-1] / dt))
    for i in range(len(frames)):
        signal_part = signal[int(i * dt * fs):int((i + 1) * dt * fs)]
        frames[i] = np.sum(welch(signal_part.reshape(signal_part.shape[0]), fs=fs)[1])
        frames[i] = int(frames[i] * 1000) / 100
    x = np.array(list(range(len(frames) + 1))) / len(frames) * x[-1]
    plt.plot([x[0], x[-1]], [1, 1], color="black")
    plt.plot([x[0], x[-1]], [0, 0], color="black")
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [0, 1], color="black")
        if i < len(frames):
            plt.annotate(str(frames[i]), (x[i] + 0.05, 0.4), size=23)
    plt.ylabel("Frames    Spectrogram      Signal  ", fontsize=18)
    plt.xticks([])
    plt.yticks([])


def plot_frames_MUAe(bl, fs, dur):
    # SETUP
    chn_ix = np.random.randint(64)
    t_start = np.random.randint(0, 100)

    y = bl.segments[0].analogsignals[0][t_start * fs:int((t_start + dur) * fs), chn_ix]
    signal = y / max(y.max(), np.abs(y.min()))

    # 1. raw signal with threshold
    offset = 0.7
    signal = signal + offset
    x = np.linspace(0, len(signal) / fs, len(signal))
    plt.plot(x, signal, color="black")
    plt.plot([x[0], x[-1]], [np.mean(signal), np.mean(signal)], color="grey")

    # 2. convert to frames (discrete signal)
    signal = signal - offset
    plt.scatter(x, signal, color="black", s=3)

    plt.ylabel("Frames                      Signal", fontsize=20)
    plt.xticks([])
    plt.yticks([])


# figure b) - cors and inter cors
def plot_cors(path, chn):
    """
    Plot correlation map
    :param path:
    :param chn:
    :return:
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    cors = data["correlation_maps"]

    with open("layout.pkl", "rb") as f:
        layout = pickle.load(f)

    corr_map = np.full((8, 8), 0.)
    for x in range(8):
        for y in range(8):
            corr_map[x, y] = cors[chn - 1, layout[x, y] - 1]

    plt.gca().set_aspect('equal', 'box')
    g = plt.pcolormesh(corr_map, cmap="RdBu_r", vmin=-1, vmax=1)
    # plt.colorbar(g, ticks=[-1, 0, 1], shrink=0.5, pad=0.08,
    #              location="bottom", orientation="horizontal", label="correlation")
    for x in range(8):
        for y in range(8):
            if layout[x, y] == chn:
                plt.scatter(y + 0.5, x + 0.5, color="yellow", edgecolors="black", label="seed channel")
    # plt.legend(bbox_to_anchor=(0.5, -0.1), loc='lower center', ncol=1)
    _ = plt.xticks([])
    _ = plt.yticks([])
    return g


def plot_inter_cors(path, chn):
    """
    Plot interpolated correlation map with removed untunned channels
    :param path:
    :param chn:
    :return:
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    inter = data["interpolated_correlation_maps"]
    rins = data["real_channel_inds"]

    plt.gca().set_aspect('equal', 'box')
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_under(color="grey")
    inter_corr_map = inter[:, chn - 1].reshape((65, 65))

    plt.pcolormesh(inter_corr_map.T, cmap=cmap, vmin=-1, vmax=1)
    plt.xticks([])
    plt.yticks([])

    plt.scatter(rins[chn - 1, 0] + 0.5, rins[chn - 1, 1] + 0.5, color="yellow", edgecolors="black")


# figure c) - pca plane
def plot_pca(path):
    """
    Plot projected interpolated correlation vectors onto PCs (PCs defined by path)
    :param path:
    :return:
    """
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    points = data["points_pca_plane"]
    center = points.mean(axis=0)
    p = points - center
    angle = np.arctan(p[:, 0] / p[:, 1])
    angle[p[:, 1] < 0] += np.pi
    angle -= angle.min()
    labels = angle / 2
    g = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="hsv", vmin=0, vmax=np.pi, s=25, edgecolors="grey")
    cb = plt.colorbar(g, ticks=[0, 3.14], shrink=0.8, pad=0.12,
                      location="right", orientation="vertical")
    cb.set_label(label="angle", fontsize=18)
    outline = max(np.max(np.abs(points[:, 0])), np.max(np.abs(points[:, 1])))
    plt.plot([0, 0], [-outline, outline], color="black")
    plt.plot([-outline, outline], [0, 0], color="black")
    plt.xlabel("PC3", fontsize=20)
    plt.ylabel("PC4", fontsize=20)
    plt.gca().set_aspect('equal', 'box')
    plt.xticks([])
    plt.yticks([])


# figure d) - spont. map, OP map, test
def plot_spont_map(path):
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    spont = data["test"]
    cmap = plt.get_cmap("hsv").copy()
    cmap.set_under(color="grey")
    plt.gca().set_aspect('equal', 'box')
    plt.pcolormesh(spont, cmap=cmap, vmin=0)
    plt.title("Spontaneous map", fontsize=20, pad=10)
    plt.xticks([])
    plt.yticks([])


def plot_ori_map(path):
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    ori = data["reference"]
    cmap = plt.get_cmap("hsv").copy()
    cmap.set_under(color="grey")
    plt.gca().set_aspect('equal', 'box')
    plt.pcolormesh(ori, cmap=cmap, vmin=0)
    plt.title("Orientation preference map", fontsize=20, pad=10)
    plt.xticks([])
    plt.yticks([])


# figure e)
def plot_hist(path):
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    plt.title(f"Similarity scores (p={int(data['percentile'] * 100) / (100 * 100)})", fontsize=20, pad=10)
    h = plt.hist(data['control_scores'], bins=100, color="grey")[0]
    plt.plot([data['test_score'], data['test_score']], [0, np.mean(h)], color="black", linewidth=5)
    # x, y = KDE(data['control_scores'], 1)
    # y = y / np.max(y) * np.max(h)
    # plt.plot(x, y, color="black")
    plt.xticks([])
    plt.yticks([])


def final_figure(res_path, method, monkey, arraynr, params):
    """
    Plot figure for each method in detailed analysis of L13
    :param res_path: path to correlation maps folder
    :param method: MUA, MUAe, LFP, nLFP
    :param monkey: L or A
    :param arraynr: Array ID
    :param params:
    :return:
    """
    # SETUP
    cors_path = f"{res_path}/correlation_maps_pool_monkey_{monkey}{arraynr}_{method}_spont.pkl"
    pca_path = f"{res_path}/pca_communities_monkey_{monkey}{arraynr}_{method}_spont.pkl"
    test_path = f"{res_path}/testing/testing_results_{monkey}{arraynr}_{method}_{params['remove_PCA_dims']}.pkl"

    # extract spikes a)
    plt.figure(figsize=(28, 16))
    from extract_signal import LFP, nLFP, MUAe, MUA

    # load signal
    path = "/home/matej/Desktop/Bakalarka/final_figs/recordings/2_final_recording_after_stimulation_005.ns6"
    if method == "MUA":
        fs = 30000
        bl = MUA(path, -3)
        dt = 0.1
    elif method == "nLFP":
        fs = 500
        bl = nLFP(path, -1)
        dt = 0.3
    elif method == "LFP":
        fs = 500
        bl = LFP(path)
        dt = 0.2
    elif method == "MUAe":
        fs = 1000
        bl = MUAe(path)

    # plot extracted frames (a)
    plt.subplot(2, 3, 1)
    if method in ["MUA", "nLFP"]:
        plot_extracted_spikes(bl, fs, dt, dur=1.5)
    elif method == "LFP":
        plot_frames_lfp(bl, fs, dt)
    if method == "MUAe":
        plot_frames_MUAe(bl, fs, 0.1)
        for loc in ["left", "top"]:
            plt.gca().spines[loc].set_visible(False)
    else:
        for loc in ["left", "right", "top", "bottom"]:
            plt.gca().spines[loc].set_visible(False)

    # plot cors and interpolated cors (b)
    chns = [32, 51]
    loc = [3, 9]
    for i in range(2):
        plt.subplot(4, 6, loc[i])
        if i == 0:
            plt.title("Correlation map", fontsize=20)
        g = plot_cors(cors_path, chns[i])
        plt.subplot(4, 6, loc[i] + 1)
        if i == 0:
            plt.title("Interpolated map", fontsize=20)
        plot_inter_cors(cors_path, chns[i])

    plt.subplot(2, 3, 1)
    axs = [plt.subplot(4, 6, loc[i] + j) for i in range(2) for j in range(2)]
    cb = plt.colorbar(g, ax=axs, ticks=[-1, 0, 1], shrink=0.8, pad=0.08,
                      location="bottom", orientation="horizontal")
    cb.set_label(label="Correlation", fontsize=16)

    # plot pca projection (c)
    plt.subplot(2, 3, 3)
    plot_pca(pca_path)

    # plot spontaneous map, orientation map and histogram (d)
    plt.subplot(2, 3, 4)
    plot_spont_map(test_path)
    plt.subplot(2, 3, 5)
    plot_ori_map(test_path)
    plt.subplot(2, 3, 6)
    plot_hist(test_path)

    plt.savefig(f"final_.png")


def correlation_figure(path_to_cors):
    from scipy.stats import pearsonr
    correlation_maps = {}
    for i in range(4):
        m = ["MUA", "nLFP", "MUAe", "LFP"][i]
        with open(f"{path_to_cors}/correlation_maps_pool_monkey_L13_{m}_spont.pkl", "rb") as f:
            correlation_maps[m] = pickle.load(f)

    # load channel coords
    coords = {}
    with open("layout.pkl", "rb") as f:
        layout = pickle.load(f)
    for r in range(8):
        for c in range(8):
            coords[layout[r, c]] = (r, c)

    # load correlation values, CMs correlations and distances
    c, cm, d = {}, {}, {}
    for i in range(4):
        m = ["MUA", "nLFP", "MUAe", "LFP"][i]
        cors = correlation_maps[m]["correlation_maps"]
        for chn1 in range(64):
            for chn2 in range(chn1 + 1, 64):
                if chn1 == 0 and chn2 == 1:
                    c[m] = []
                    cm[m] = []
                    d[m] = []
                c[m].append(cors[chn1, chn2])
                tmp = pearsonr(cors[:, chn1], cors[:, chn2])[0]
                cm[m].append(-2 if np.isnan(tmp) else tmp)
                d[m].append(np.linalg.norm(np.array(coords[chn1 + 1]) - np.array(coords[chn2 + 1])))
        c[m] = np.array(c[m])
        cm[m] = np.array(cm[m])
        d[m] = np.array(d[m])

    # create cors x cm cors line and plot it
    plt.figure(figsize=(26, 10))
    plt.subplot(1, 2, 1)
    x = {}
    y = {}
    std = {}
    segments = 40
    for i in range(4):
        m = ["MUA", "nLFP", "MUAe", "LFP"][i]
        x[m], y[m], std[m] = [], [], []
        x_ = c[m]
        y_ = cm[m]
        bin_x = np.linspace(np.min(x_), np.max(x_), segments)
        for j in range(bin_x.shape[0] - 1):
            tmp = y_[np.where(np.logical_and(x_ >= bin_x[j], x_ < bin_x[j + 1]))]
            if tmp.shape[0] > 0:
                x[m].append(bin_x[j] + ((bin_x[j + 1] - bin_x[j]) / 2))
                y[m].append(np.mean(tmp))
                std[m].append(np.std(tmp))
        x[m], y[m], std[m] = np.array(x[m]), np.array(y[m]), np.array(std[m])
        _ = plt.plot(x[m], y[m], label=m, linewidth=4)
        _ = plt.fill_between(x[m], y[m] - std[m], y[m] + std[m], color=(0.2, 0.4, 0.7, 0.2))
        _ = plt.xlabel("Channel correlations", fontsize=25)
        _ = plt.ylabel("Map correlations", fontsize=25)
        if i == 3:
            _ = plt.legend(fontsize=15)
            plt.ylim(-1, 1)

    # plot cors x distance
    for i in range(4):
        m = ["MUA", "nLFP", "MUAe", "LFP"][i]
        x = np.unique(d[m])
        y = np.zeros((x.shape[0], 2))
        std = np.zeros((x.shape[0], 2))
        for ix in range(x.shape[0]):
            for j in range(2):
                data = [cm[m], c[m]][j][np.where(d[m] == x[ix])]
                data = data[np.where(data >= -1)]
                y[ix, j] = np.mean(data)
                std[ix, j] = np.std(data)
        loc = [3, 4, 7, 8]
        plt.subplot(2, 4, loc[i])
        plt.plot(x, y[:, 0], color="red", label="Map correlations", linewidth=2)
        plt.plot(x, y[:, 1], color="blue", label="Channel correlations", linewidth=2)
        colors = [(255 / 256, 190 / 256, 171 / 256), (177 / 256, 201 / 256, 234 / 256)]
        for j in range(2):
            plt.fill_between(x, y[:, j] - std[:, j], y[:, j] + std[:, j], color=colors[j], alpha=0.5)
        plt.ylim(-0.4, 1)
        x = np.linspace(1, 10, 10)
        plt.xticks(x, (x * 400).astype(int))
        # plt.yticks([-1, -0.5, 0, 0.5, 1])
        plt.title(m, fontsize=20)
        if i == 0:
            plt.ylabel("Correlation values", fontsize=18)
        if i == 1:
            plt.legend(fontsize=18)
        if i == 3:
            plt.xlabel("Distance (um)", fontsize=18)
            plt.savefig("final/dist_.png")
