from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_ruclip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything
from deep_translator import GoogleTranslator
import os

#What're you creating?
text = 'Ugly Cat'
image_save_dir = 'Russia'
Image_prefix = 'Ugly Cat'
Start_seed = 343
Images_per_res = 1
Seed_batches = 1
Batch_size = 5
Upscale_multiplier = '2x'

Cache_dir = 'C:/Users/Deads/Downloads/art'

Translated = GoogleTranslator(source='auto', target='ru').translate(text)

if not os.path.exists(image_save_dir):
  os.mkdir(image_save_dir)

# prepare models:
device = 'cuda'
dalle = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device, Cache_dir=Cache_dir)
realesrgan = get_realesrgan(Upscale_multiplier, device=device, Cache_dir=Cache_dir)
tokenizer = get_tokenizer(Cache_dir)
vae = get_vae(Cache_dir=Cache_dir).to(device)
ruclip, ruclip_processor = get ruclip('ruclip-vit-base-patch32-384', Cache_dir=Cache_dir)
ruclip = ruclip.to(device)

for seed in range(Start_seed, (Start_seed+Seed_batches)):
    print(f'Your Text: {text}')
    print(f'Russian: {Translated}')
    print(f'Seed: {seed}')


seed_everything(42)
pil_images = []
scores = []
for top_k, top_p, images_num in [
    (2048, 0.995, 24),
]:
    _pil_images, _scores = generate_images(Translated, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, bs=Batch_size, top_p=top_p)
    pil_images += _pil_images
    scores += _scores
sr_images = super_resolution(pil_images, realesrgan)

for i, img in enumerate(sr_images):
img.save(f'{image_save_dir}/{Image_prefix}-{seed}-{i}.jpg')

    print('Image Batch Saved')