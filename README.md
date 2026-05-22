# 貪吃蛇AI

基於強化學習 (Reinforcement Learning) 的實作專案，訓練 AI 在經典貪吃蛇遊戲中進行遊玩。
本專案包含並支援了 **PPO (Proximal Policy Optimization)** 以及 **DQN (Deep Q-Network)** 兩種算法的實作。

## 專案特色

*   **高效能自訂環境**：基於 Gymnasium 與 NumPy 實作，支援 `VectorEnv` 平行訓練，並提供 4 通道 12x12 的 One-hot 觀察空間。
*   **啟發式獎勵重塑 (PBRS)**：有效解決貪吃蛇環境中獎勵稀疏與原地轉圈（Reward Hacking）的問題。
*   **PPO 訓練優化**：結合 Actor-Critic 架構、GAE、Value Clipping、動態熵調整與學習率退火，克服梯度爆炸與災難性遺忘。
*   **模型表現良好**：PPO 最佳模型在 12x12 網格中，平均長度達 74.12，最大長度達 112（佔據 77% 空間）。
*   **PPO vs. DQN 深度對比**：實驗證實，面對需多步規劃的環境，PPO 的 GAE 與熵探索機制在收斂速度與最終表現皆優於 DQN。

## 視覺表現

### AI遊玩畫面
| PPO Agent | DQN Agent |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/061ab3ae-ce4d-4087-b7ce-c4cdc2310f22" controls></video> | <video src="https://github.com/user-attachments/assets/bdf30fa5-5b86-4bb7-a1fe-56534f1cac18" controls></video> |


### 訓練表現記錄
| PPO 訓練最大分數記錄 | DQN 訓練最大分數記錄 |
| :---: | :---: |
| ![PPO Max Score](assets/PPO_Max.png) | ![DQN Max Score](assets/DQN_Max.png) |

## 模型架構參考 (PPO)
![PPO Architecture](assets/PPO_architecture.png)

## 安裝

1. 克隆或下載此專案，並進入資料夾：
   ```bash
   git clone https://github.com/Ethanol017/Snake-AI.git
   cd Snake-AI
   ```
2. 安裝必要的依賴套件
   ```bash
   pip install -r requirements.txt
   ```
## 檔案結構

- `model.py`: 定義了神經網路，包含了 `SnakePPO` 和 `SnakeDQN` 兩個網路。
- `train.py`: 訓練模型的主要腳本。
- `test.py`: 測試跟視覺化已訓練好的模型。
- `Gym-Snake/`: 貪吃蛇的自訂 Gym 環境原始碼。
- `models/`: 存放訓練中或訓練完成的 `.pth` 權重檔。
- `logs/`: TensorBoard 的實驗數據與紀錄檔。
- `assets/`: 相關的媒體資源與表現成效圖片。
- `tools`： 各項工具，如：環境測試、視覺化、Tensorboard工具
