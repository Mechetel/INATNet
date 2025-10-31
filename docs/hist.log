git clone https://github.com/Mechetel/INATNet.git
cd INATNet
apt install python3-pip unzip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install imageio tqdm reedsolo

cd data
curl -L -o GBRASNET.zip https://www.kaggle.com/api/v1/datasets/download/zapak1010/bossbase-bows2
unzip GBRASNET.zip -d GBRASNET
rm GBRASNET.zip
cd ..

python3 train.py
