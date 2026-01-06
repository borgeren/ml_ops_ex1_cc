import matplotlib.pyplot as plt
import torch
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from exercise1.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(model_checkpoint: str = "models/model.pth", figure_name: str = "embeddings.png") -> None:
    model: torch.nn.Module = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.fc1 = torch.nn.Identity()

    test_images, test_target = torch.load("data/processed/test_images.pt"), torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)

        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

        if embeddings.shape[1] > 500:
            pca = PCA(n_components=100)
            embeddings = pca.fit_transform(embeddings)
        tsne = TSNE(n_components=2, random_state=1)
        embeddings = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 10))
        for i in range(10):
            idxs = targets == i
            plt.scatter(embeddings[idxs, 0], embeddings[idxs, 1], label=str(i), alpha=0.6)
        plt.legend()
        plt.title("t-SNE visualization of MNIST embeddings")
        plt.savefig(f"reports/figures/{figure_name}")
        print(f"Saved embedding visualization to reports/figures/{figure_name}")


if __name__ == "__main__":
    typer.run(visualize)
