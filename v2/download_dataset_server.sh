apt install python3-pip unzip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip3 install imageio tqdm reedsolo

mkdir -p ~/datasets
mkdir -p ~/datasets/ready_to_use

curl -L -o ~/datasets/GBRASNET.zip https://www.kaggle.com/api/v1/datasets/download/zapak1010/bossbase-bows2
unzip ~/datasets/GBRASNET.zip -d ~/datasets
rm ~/datasets/GBRASNET.zip

git clone https://github.com/Mechetel/INATNet.git

python3 ~/INATNet/v2/prepare_dataset.py --source_dir ~/datasets/GBRASNET --output_dir ~/datasets/ready_to_use/GBRASNET

cd INATNet/v2
python3 train.py --cover_train_path ~/datasets/ready_to_use/GBRASNET/BOWS2/cover/train \
                 --stego_train_path ~/datasets/ready_to_use/GBRASNET/BOWS2/stego/WOW/0.2bpp/train \
                 --cover_val_path ~/datasets/ready_to_use/GBRASNET/BOWS2/cover/val \
                 --stego_val_path ~/datasets/ready_to_use/GBRASNET/BOWS2/stego/WOW/0.2bpp/val

