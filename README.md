# inference_speedup_Llama-3.2-3B-instruct
## 下載
您可以使用以下指令下載並進到資料夾:
```
git clone https://github.com/sandychinghuang/inference_speedup_Llama-3.2-3B-instruct.git
cd inference_speedup_Llama-3.2-3B-instruct
```
## 環境設置
您可以使用以下指令建置環境(因`requirements.txt`內部有些套間依賴torch，因此需先裝torch):
```
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install gemlite==0.4.4

```
使用模型:https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

## 使用方法
您可以使用以下指令執行重現`result.csv`與`result.png`之結果:
```
python result.py
```