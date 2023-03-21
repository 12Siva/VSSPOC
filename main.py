from fastapi import FastAPI
from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset
from PIL import Image
import numpy as np

model_ckpt = "nateraw/vit-base-beans"
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    print('Hello World - Server')
    return {"message": f"Hello {name}"}


@app.get("/hello/vss/{image}")
async def vss(image: str):
    print(f'Hello World VSS - {image}')
    dataset = load_dataset("beans")

    print(dataset["train"].features)

    seed = 42
    num_samples = 100
    dataset = load_dataset("beans", split="train")
    candidate_dataset = dataset.shuffle(seed=seed).select(range(num_samples))

    test_ds = load_dataset("beans", split="test")

    dataset_with_embeddings = candidate_dataset.map(
        lambda example: {'embeddings': extract_embeddings(example["image"])})
    dataset_with_embeddings.add_faiss_index(column='embeddings')

    def get_neighbors(query_image, top_k=5):
        qi_embedding = model(**extractor(query_image, return_tensors="pt"))
        qi_embedding = qi_embedding.last_hidden_state[:, 0].detach().numpy().squeeze()
        scores, retrieved_examples = dataset_with_embeddings.get_nearest_examples('embeddings', qi_embedding, k=top_k)
        return scores, retrieved_examples

    random_index = np.random.choice(len(test_ds))
    query_image = test_ds[random_index]["image"]
    query_image

    scores, retrieved_examples = get_neighbors(query_image)

    print(scores)

    print(retrieved_examples)

    print(query_image)

    def image_grid(imgs, rows, cols):
        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        for i, img in enumerate(imgs): grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    images = [query_image]
    images.extend(retrieved_examples["image"])

    image_grid(images, 1, len(images))


    return {"message": f"Hello VSS {image}"}

def extract_embeddings(image):
    image_pp = extractor(image, return_tensors="pt")
    features = model(**image_pp).last_hidden_state[:, 0].detach().numpy()
    return features.squeeze()
