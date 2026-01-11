from pathlib import Path
import urllib.request

URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

def main():
    output_path = Path('data/lid.176.bin')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return
    
    urllib.request.urlretrieve(URL, output_path)


if __name__ == "__main__":
    main()
