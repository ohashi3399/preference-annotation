import gradio as gr
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict


# Hugging Faceからデータセットを読み込む
dataset_name = "ryota39/preference_test"
dataset = load_dataset("ryota39/preference_test")
df = pd.DataFrame(dataset["train"])

annotated_dataset_name = "ryota39/preference_test_annotated"


def annotate(preference):
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

    # 次のサンプルのデータを取得する
    return display_next_sample()


def display_next_sample():
    """
    未アノテーションのサンプルを表示する関数
    """
    # 未アノテーションのサンプルを取得する
    global index
    unannotated_samples = df[df["annotation_done"] == 0]
    if len(unannotated_samples) > 0:
        index = unannotated_samples.index[0]
        next_instruction = df.loc[index, "instruction"]
        next_response1 = df.loc[index, "response1"]
        next_response2 = df.loc[index, "response2"]
    else:
        next_instruction = "All samples have been annotated."
        next_response1 = ""
        next_response2 = ""

    return next_instruction, next_response1, next_response2


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
    new_dataset.push_to_hub(annotated_dataset_name)

    # ローカルにjsonlファイルとして保存する
    annotated_df.to_json("annotated_dataset.jsonl", orient="records", lines=True)


def handle_response1():
    """
    Response1ボタンが押された時の処理
    """
    return annotate("response1")


def handle_response2():
    """
    Response2ボタンが押された時の処理
    """
    return annotate("response2")


with gr.Blocks() as demo:
    gr.Markdown("# Preference Annotation App")

    with gr.Row():
        instruction = gr.Textbox(label="Instruction")

    with gr.Row():
        response1 = gr.Textbox(label="Response 1")
        response2 = gr.Textbox(label="Response 2")

    with gr.Row():
        btn_response1 = gr.Button("Select Response 1")
        btn_response2 = gr.Button("Select Response 2")

    with gr.Row():
        upload_button = gr.Button("Finish Annotation", variant="primary")

        # ボタンクリック時の処理を設定（出力コンポーネントを指定）
        btn_response1.click(
            fn=handle_response1, outputs=[instruction, response1, response2]
        )
        btn_response2.click(
            fn=handle_response2, outputs=[instruction, response1, response2]
        )
        upload_button.click(fn=upload_annotated_dataset)

    # 最初のサンプルを表示する
    index = df[df["annotation_done"] == 0].index[0]
    instruction.value = df.loc[index, "instruction"]
    response1.value = df.loc[index, "response1"]
    response2.value = df.loc[index, "response2"]

demo.launch()
