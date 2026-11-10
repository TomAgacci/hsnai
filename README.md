Licensed Under Creative Commons No Derivatives Non-Commercial Open-Source

cd ~/directiorywithfiles

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install "diffusers[torch]" transformers accelerate safetensors pillow


Run
sudo python generator.py description.txt imageoutput.png
or
python generator.py description.txt output.png --image input.jpg
or
python generator.py description.txt output.png --input input.jpg

Source Control
python generator.py description.txt output.png \
  --image input.jpg \
  --strength 0.45

python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

python generator.py description.txt output.png

python3 -m venv .venv
source .venv/bin/activate
pip install torch "diffusers[torch]" transformers accelerate safetensors pillow

DIAGRAM<br>
  description.txt  +  input.jpg<br>
                   |<br>
           +-------+-------+<br>
                   |<br>
                   HSN/ZCMS<br>
                   |<br>
               SDXL img2img<br>
                   |<br>
         ImageMagick<br>
                   |<br>
                   v<br>
          output.png<br>
