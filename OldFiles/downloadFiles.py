import requests
from pathlib import Path

def download_file(url, folder):
    
    local_filename = url.split('/')[-1]
    local_filename = folder + local_filename

    with requests.get(url, stream = True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk:
                    f.write(chunk)
                    

urlsDownload = {
    'main' : "https://raw.githubusercontent.com/AlfredoFilho/Auto-Correction-Tests/master/AutoCorrection.py",
    'media': [
        "https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/media/brain.h5",
        "https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/media/brainModel.json",
        "https://raw.githubusercontent.com/AlfredoFilho/Auto-Correction-Tests/master/media/coordinates.json",
        "https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/media/example.pdf",
        "https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/media/example.png"
    ],
    'modules': [
        "https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/media/modules/__init__.py",
        "https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/media/modules/image.py",
        "https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/media/modules/number.py",
        "https://github.com/AlfredoFilho/Auto-Correction-Tests/raw/master/media/modules/model.py"
    ]
}

Path("media/modules").mkdir(parents=True, exist_ok=True)

print('Download files...')

try:
    for url in urlsDownload['media']:
        download_file(url = url, folder = 'media/')

    for url in urlsDownload['modules']:
        download_file(url = url, folder = 'media/modules/')

    download_file(url = urlsDownload['main'], folder = '')

except:
    print('Ocorreu algum erro, talvez você esteja sem internet. Ou o github está fora do ar.')

print('...100%')