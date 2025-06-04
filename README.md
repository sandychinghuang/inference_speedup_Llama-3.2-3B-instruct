#### 下載
您可以使用以下程式碼下載並進到資料夾:
```
git clone https://github.com/sandychinghuang/inference_speedup_Llama-3.2-3B-instruct.git
cd inference_speedup_Llama-3.2-3B-instruct
```
#### 環境設置
您可以使用以下程式碼建置環境:
```
pip install huggingface-hub[cli]
pip install torch torchvision torchaudio
pip install transformers==4.50.3
pip install timm==1.0.15
pip install datasets==3.5.0
pip install accelerate==1.6.0
pip install gemlite==0.4.4
pip install hqq==0.2.5
pip install triton==3.2.0
pip install tqdm
pip install numpy
```
使用模型:https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

#### 使用方法
您可以使用以下程式碼執行重現`result.csv`與`result.png`之結果:
```
python result.py
```