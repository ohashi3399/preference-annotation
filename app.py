import gradio as gr
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import json
import os

# Hugging Faceからデータセットを読み込む
dataset = load_dataset("your_dataset_name")
df = pd.DataFrame(dataset["train"])


def annotate(instruction, response1, response2, preference):
    """
    アノテーションを行う関数
    """
    # "preference"カラムを更新する
    if preference == "response1":
        df.loc[index, "preference"] = "response1"
    else:
        df.loc[index, "preference"] = "response2"

    # "annotation_done"カラムを1に更新する
    df.loc[index, "annotation_done"] = 1

    # 更新されたデータフレームをデータセットに反映させる
    dataset["train"].add_item(df.loc[index].to_dict())

    return "Annotation complete!"


def upload_annotated_dataset():
    """
    アノテーション済みのデータセットをHugging Faceにアップロードし、
    ローカルにjsonlファイルとして保存する
    """
    # アノテーション済みのサンプルのみ抽出する
    annotated_df = df[df["annotation_done"] == 1]

    # 新しいデータセットを作成する
    annotated_dataset = Dataset.from_pandas(annotated_df)
    new_dataset = DatasetDict({"train": annotated_dataset})

    # Hugging Faceにアップロードする
    new_dataset.push_to_hub("your_new_dataset_name")

    # ローカルにjsonlファイルとして保存する
    annotated_df.to_json("annotated_dataset.jsonl", orient="records", lines=True)

    # 最初に読み込んだデータセットをHugging Faceにアップロードし直す
    dataset.push_to_hub("your_dataset_name")

    return "Annotated dataset uploaded!"


with gr.Blocks() as demo:
    gr.Markdown("# Annotation App")

    with gr.Row():
        instruction = gr.Textbox(label="Instruction")

    with gr.Row():
        response1 = gr.Textbox(label="Response 1")
        response2 = gr.Textbox(label="Response 2")

    with gr.Row():
        preference = gr.Radio(["response1", "response2"], label="Preference")

    with gr.Row():
        annotate_button = gr.Button("Submit", variant="primary", click=annotate)
        upload_button = gr.Button(
            "Finish Annotation", variant="primary", click=upload_annotated_dataset
        )

    # データセットから未アノテーションのサンプルを取得する
    index = df[df["annotation_done"] == 0].index[0]
    instruction.value = df.loc[index, "instruction"]
    response1.value = df.loc[index, "response1"]
    response2.value = df.loc[index, "response2"]

demo.launch()
