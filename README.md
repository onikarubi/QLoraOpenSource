# QLoraOpenSource

大規模言語モデルとHuggingFaceを用いたファインチューニングを行うためのテンプレートです。このリポジトリでは、Gemma、Llama、Elyzaなどのベースモデルを使用して微調整を行い、カスタムアダプターモデルを生成することができます。

## デモ

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onikarubi/QLoraOpenSource/blob/master/notebook/QLoraOpenSource.ipynb) をクリックして、手軽に試してみてください。

## Features

- Gemma、Llama、Elyzaをベースモデルとした微調整の実行
- 微調整で作成したアダプターモデルの推論

## Requirement

- [wandb](https://www.wandb.jp/) にログインして以下の環境変数を設定してください：
  - `WANDB_API_KEY`: wandbのAPIキー
  - `WANDB_PROJECT`: 任意のプロジェクト名
  - `WANDB_LOG_MODEL`: 'checkpoint' を設定
- [Hugging Face](https://huggingface.co/) にログインして、トークンを `HF_TOKEN` という環境変数に設定してください。
- Google Colabのランタイム設定でGPUを有効にしてください。

## Usage（Google Colabノートブックの場合）

1. リポジトリをクローンします。

    ```bash
    !git clone https://github.com/onikarubi/QLoraOpenSource.git
    ```

    ```bash
    %cd QLoraOpenSource
    ```

2. 必要なパッケージをインストールします。

    ```bash
    !pip install -r requirements.txt
    ```

3. モデルのファインチューニングを実行します。

    ```bash
    !python main.py --task train --dataset_path datasets/sample.jsonl --repo_id google/gemma-2-2b-it --output_dir ./output --epochs 3
    ```

4. 作成したアダプターモデルで推論を行います。

    ```bash
    !python main.py --task run --repo_id google/gemma-2-2b-it --adapter_path ./output --max_tokens 500 --questions "自己紹介をしてくれますか？"
    ```

## License

MIT License
