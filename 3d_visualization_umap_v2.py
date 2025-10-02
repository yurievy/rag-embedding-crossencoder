import json
import os
import numpy as np
import pandas as pd
import umap
import plotly.express as px
import plotly.io as pio
from plotly.colors import qualitative

# -----------------------------
# Plotly renderer
# -----------------------------
pio.renderers.default = "browser"

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "data/intfloat/multilingual-e5-base"
DATASET_FILE = "data/dataset_intfloat-multilingual-e5-base.json"
RAW_FILE = "data/raw_intfloat-multilingual-e5-base.json"
SOURCE_DIR = "source"
N_COMPONENTS = 2 # Set to 2 for 2D visualization or 3 for 3D visualization

# -----------------------------
# Load embeddings dataset
# -----------------------------
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Load raw dataset with question texts
with open(RAW_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Dictionary: question_id -> question text
q_texts = {item["id"]: item["question"] for item in raw_data}

# -----------------------------
# Map question text -> source file
# -----------------------------
question_to_file = {}
for fname in os.listdir(SOURCE_DIR):
    if not fname.endswith(".txt"):
        continue
    fpath = os.path.join(SOURCE_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[query]"):
                q_text = line[len("[query]"):].strip()
                question_to_file[q_text] = fname

# -----------------------------
# Collect embeddings per chunk (no averaging)
# -----------------------------
question_ids = []
questions = []
files = []
embeddings = []

for idx, item in enumerate(data, 1):
    embedding = np.array(item["embedding"])
    question_text = q_texts.get(int(item["question_id"]), item["question_id"])
    file_name = question_to_file.get(question_text, "unknown")
    
    question_ids.append(idx)  # just sequential ID for visualization
    questions.append(item["question_id"] + '. ' + question_text)
    files.append(file_name)
    embeddings.append(embedding)

embeddings = np.array(embeddings)

# Print total questions and chunks
print("Total chunks:", len(data))
print("Total unique questions:", len(set(questions)))

# -----------------------------
# Apply UMAP
# -----------------------------
reducer = umap.UMAP(n_components=N_COMPONENTS, random_state=42)
embedding_nd = reducer.fit_transform(embeddings)

# -----------------------------
# Truncate question text for hover (max 100 chars)
# -----------------------------
short_questions = [q[:100] + "…" if len(q) > 100 else q for q in questions]

# -----------------------------
# Create a consistent color palette for files
# -----------------------------
unique_files = sorted(set(files))
palette = qualitative.Plotly + qualitative.D3 + qualitative.T10
palette = palette[:len(unique_files)]
color_map = {fname: palette[i % len(palette)] for i, fname in enumerate(unique_files)}

# -----------------------------
# Build DataFrame for Plotly
# -----------------------------
df = pd.DataFrame(embedding_nd, columns=[f"UMAP{i+1}" for i in range(N_COMPONENTS)])
df["QuestionID"] = question_ids
df["Question"] = short_questions
df["File"] = files

# -----------------------------
# Plot
# -----------------------------
if N_COMPONENTS == 3:
    fig = px.scatter_3d(
        df,
        x="UMAP1", y="UMAP2", z="UMAP3",
        color="File",
        color_discrete_map=color_map,
        text="QuestionID",
        hover_data=["QuestionID", "Question", "File"]
    )
    fig.update_layout(
        title=f"{MODEL_NAME}: UMAP (3D, colored by source file)",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3"
        )
    )
else:
    fig = px.scatter(
        df,
        x="UMAP1", y="UMAP2",
        color="File",
        color_discrete_map=color_map,
        text="QuestionID",
        hover_data=["QuestionID", "Question", "File"]
    )
    fig.update_layout(
        title=f"{MODEL_NAME}: UMAP (2D, colored by source file)",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2"
    )

# -----------------------------
# Styling
# -----------------------------
fig.update_traces(
    marker=dict(size=3),
    textposition="top center",
    textfont=dict(color="rgba(0,0,0,0)"),
    hovertemplate='%{customdata[0]}<br>%{customdata[1]}<extra></extra>'
)

fig.show()
