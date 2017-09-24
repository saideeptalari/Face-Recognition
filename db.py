import numpy as np
import cPickle

def add_embeddings(embedding, label,
                   embeddings_path="face_embeddings.npy",
                   labels_path="labels.pickle"):
    first_time = False
    try:
        embeddings = np.load(embeddings_path)
        labels = cPickle.load(open(labels_path))
    except IOError:
        first_time = True

    if first_time:
        embeddings = embedding
        labels = [label]
    else:
        embeddings = np.concatenate([embeddings, embedding], axis=0)
        labels.append(label)

    np.save(embeddings_path, embeddings)
    with open(labels_path, "w") as f:
        cPickle.dump(labels, f)

    return True