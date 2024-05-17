python -m pip install poetry
poetry config virtualenvs.in-project true

poetry install
poetry shell

cd src/rd4ad
# NOTE: This dataset is for non-commercial purposes only!
mkdir checkpoints
mkdir mvtec
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz -C mvtec/

python main.py