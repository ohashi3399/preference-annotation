import os, argparse
import gradio as gr
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict


access_token = os.environ.get("ANNOTATION_TOKEN")
parser = argparse.ArgumentParser(description="Annotation app for preference dataset")
parser.add_argument("--ds_name", type=str, default="Aratako/dataset-for-annotation-v2")
parser.add_argument("--ds_out", type=str, default="preference-team/annotation-test")
args = parser.parse_args()


# Hugging Faceからデータセットを読み込む
dataset = load_dataset(args.ds_name)
df = pd.DataFrame(dataset["train"])

stem = args.ds_out.split("/")[0]
leaf = args.ds_name.split("/")[-1]
annotated_dataset_name = f"{stem}/{leaf}-annotated"


def annotate(preference: str):
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


def skip(preference: str):
    """
    アノテーションを行う関数
    """
    # "preference"カラムを更新する
    if preference == "response1":
        df.loc[index, "preference"] = "response1"
    elif preference == "response2":
        df.loc[index, "preference"] = "response2"
    else:
        df.loc[index, "preference"] = "weird"

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
    new_dataset.push_to_hub(annotated_dataset_name, token=access_token)

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


def handle_trash_bin():
    """
    trash_binボタンが押された時の処理
    """
    return annotate("weird")


with gr.Blocks() as demo:
    gr.Markdown("# Preference Annotation App")

    with gr.Accordion("ルール", open=False):
        annotation_rule = gr.Textbox(label="アノテーションルール")
        message = "- 質問文に書かれた文章に対して、応答文が2つ表示されています\n"
        message += (
            "- 2つの応答文を読んで、左右どちらが好ましいか👈か👉で回答してください\n"
        )
        message += (
            "- 選択が終わると次の質問が表示されるので、続けてアノテーションが行えます\n"
        )
        message += "- 終了するときは終了ボタンを押してください\n"
        annotation_rule.value = message

    with gr.Row():
        instruction = gr.Textbox(label="質問文")

    with gr.Row():
        response1 = gr.Textbox(label="応答文 1")
        response2 = gr.Textbox(label="応答文 2")

    with gr.Row():
        btn_response1 = gr.Button("👈", variant="primary")
        btn_response2 = gr.Button("👉", variant="primary")
        btn_trash_bin = gr.Button("🗑️", variant="secondary")

    with gr.Row():
        upload_button = gr.Button("終了", variant="huggingface")

        # ボタンクリック時の処理を設定（出力コンポーネントを指定）
        btn_response1.click(
            fn=handle_response1, outputs=[instruction, response1, response2]
        )
        btn_response2.click(
            fn=handle_response2, outputs=[instruction, response1, response2]
        )
        btn_trash_bin.click(
            fn=handle_trash_bin, outputs=[instruction, response1, response2]
        )
        upload_button.click(fn=upload_annotated_dataset)

    # 最初のサンプルを表示する
    index = df[df["annotation_done"] == 0].index[0]
    instruction.value = df.loc[index, "instruction"]
    response1.value = df.loc[index, "response1"]
    response2.value = df.loc[index, "response2"]

demo.launch()
