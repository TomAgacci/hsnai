Licensed Under Creative Commons No Derivatives Non-Commercial Open-Source

cd ~/z8

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

DIAGRAM<br>
  description.txt  +  input.jpg<br>
  &nbsp|&nbsp&nbsp&nbsp|<br>
  &nbsp+-------+-------+<br>
&nbsp&nbsp&nbsp|<br>
&nbsp&nbspHSN/ZCMS<br>
&nbsp&nbsp&nbsp|<br>
&nbspSDXL img2img<br>
               |
         ImageMagick
               |
               v
          output.png
