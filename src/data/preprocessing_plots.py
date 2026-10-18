"""Static research figures for Phase 3, without changing any inputs."""
import numpy as np
import matplotlib.pyplot as plt
from .dataset import load_checked_sample
from .preprocessing import preprocess, resize_pair, normalize_image


def choose_examples(frame):
    examples = frame.sort_values(['class_name', 'PID', 'sample_id']).drop_duplicates(['class_name', 'PID'])
    return examples.groupby('class_name', sort=True).head(2).reset_index(drop=True)


def normalization_figure(examples, output):
    fig, axes = plt.subplots(len(examples), 3, figsize=(11, 3 * len(examples)))
    hist, histogram_axes = plt.subplots(2, 3, figsize=(12, 7))
    for index, record in enumerate(examples.to_dict('records')):
        image, mask = load_checked_sample(record)
        resized, _ = resize_pair(image, mask)
        arrays = [resized, normalize_image(resized, 'minmax'), normalize_image(resized, 'zscore')]
        for ax, values, name in zip(axes[index], arrays, ['Raw (resized)', 'Min-Max', 'Z-score']):
            im = ax.imshow(values, cmap='gray', vmin=float(values.min()), vmax=float(values.max()))
            ax.set_title(f"{record['class_name']} {record['sample_id']} | {name}", fontsize=9)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
        ax = histogram_axes.ravel()[index]
        ax.hist(image.ravel(), bins=100, color='steelblue')
        ax.set_yscale('log')
        ax.set_title(f"{record['class_name']} | sample {record['sample_id']}")
        ax.set_xlabel('Raw MRI intensity'); ax.set_ylabel('Pixel count (log)')
    fig.tight_layout(); hist.tight_layout()
    fig.savefig(output / 'normalization_comparison.png', dpi=220)
    hist.savefig(output / 'representative_histograms.png', dpi=220)
    return fig, hist


def distribution_figure(stats, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    classes = ['Glioma', 'Meningioma', 'Pituitary']
    for ax, key in zip(axes[0], ['maximum', 'std']):
        ax.boxplot([stats.loc[stats.class_name.eq(c), key] for c in classes], tick_labels=classes)
        ax.set_title(f'Per-image raw {key} by class')
    patient = stats.groupby(['PID', 'class_name'])[['mean', 'std']].mean().reset_index()
    for c in classes:
        sub = patient[patient.class_name.eq(c)]
        axes[1, 0].scatter(sub.index, sub['mean'], s=12, label=c)
    axes[1, 0].set_title('Mean raw slice intensity per patient')
    axes[1, 0].set_xlabel('Patient index'); axes[1, 0].legend(fontsize=8)
    axes[1, 1].hist(stats.max_over_p999, bins=40)
    axes[1, 1].set_title('Per-image max / 99.9th percentile')
    axes[1, 1].set_xlabel('Ratio'); axes[1, 1].set_ylabel('Images')
    fig.tight_layout(); fig.savefig(output / 'intensity_distribution.png', dpi=220)
    return fig


def morphology_figure(examples, config, output):
    fig, axes = plt.subplots(len(examples), 4, figsize=(12, 3 * len(examples)))
    for row_axes, record in zip(axes, examples.to_dict('records')):
        image, mask = load_checked_sample(record)
        arrays, _ = preprocess(image, mask, 'test', config)
        for ax, key in zip(row_axes, ['image', 'region', 'boundary', 'distance']):
            ax.imshow(arrays[key][0], cmap='viridis' if key == 'distance' else 'gray',
                      **({} if key == 'image' else {'vmin': 0, 'vmax': 1}))
            ax.set_title(f"{record['class_name']} {record['sample_id']} | {key}", fontsize=9)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'morphology_examples.png', dpi=300)
    fig.savefig(output / 'morphology_examples.pdf')
    return fig


def augmentation_figure(examples, config, output):
    fig, axes = plt.subplots(len(examples), 4, figsize=(12, 3 * len(examples)))
    for index, (row_axes, record) in enumerate(zip(axes, examples.to_dict('records'))):
        image, mask = load_checked_sample(record)
        original, _ = preprocess(image, mask, 'test', config)
        augmented, _ = preprocess(image, mask, 'train', config, np.random.default_rng(42 + index))
        images = [original['image'][0], original['region'][0], augmented['image'][0], augmented['region'][0]]
        for ax, values, title in zip(row_axes, images, ['Original geometry', 'Original mask (resized)', 'Augmented + boundary', 'Augmented mask']):
            ax.imshow(values, cmap='gray')
            ax.set_title(f"{record['class_name']} {record['sample_id']} | {title}", fontsize=8)
            ax.axis('off')
        boundary = augmented['boundary'][0]
        row_axes[2].imshow(np.ma.masked_where(boundary == 0, boundary), cmap='autumn', vmin=0, vmax=1, alpha=.85)
    fig.tight_layout(); fig.savefig(output / 'augmentation_examples.png', dpi=260)
    return fig
