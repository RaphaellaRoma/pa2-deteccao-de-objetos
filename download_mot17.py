"""Download oficial; extrai só sequências de treino FRCNN, sem duplicar vídeos."""
import argparse
import hashlib
import io
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import urllib.request
import zipfile

URL = 'https://motchallenge.net/data/'


class RemoteZip(io.RawIOBase):
    """Leitura HTTP Range em blocos persistentes: evita baixar vídeos duplicados."""
    def __init__(self, url, cache):
        self.url, self.cache = url, Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.curl = shutil.which('curl.exe') or shutil.which('curl')
        if not self.curl:
            raise RuntimeError('Download seletivo precisa de curl; use --package full')
        headers = subprocess.check_output([self.curl, '-4', '-f', '-sS', '-I',
                                          '--connect-timeout', '20', '--max-time', '60', url]).decode()
        lengths = [int(l.split(':', 1)[1]) for l in headers.splitlines()
                   if l.lower().startswith('content-length:')]
        if not lengths:
            raise RuntimeError('Servidor sem Content-Length; use --package full')
        self.size, self.position, self.block_size = lengths[-1], 0, 8*1024*1024
        self.headers = headers
        # Nome do cache inclui identidade do arquivo remoto, não só URL.
        identity = [l for l in headers.splitlines()
                    if l.lower().startswith(('etag:', 'last-modified:', 'content-length:'))]
        self.tag = hashlib.sha256((url+'\n'.join(identity)).encode()).hexdigest()[:12]
        self.memory_block, self.memory_data = None, b''

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        new = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if new < 0:
            raise ValueError('Seek negativo')
        self.position = new
        return new

    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size-self.position)
        chunks = []
        while size > 0:
            block = self.position//self.block_size
            start = block*self.block_size
            end = min(self.size, start+self.block_size)-1
            if block != self.memory_block:
                path = self.cache/f'{self.tag}-{block:06d}.bin'
                if not path.exists() or path.stat().st_size != end-start+1:
                    partial = path.with_suffix('.part')
                    status = subprocess.check_output([
                        self.curl, '-4', '-f', '-sS', '-L', '--retry', '2',
                        '--connect-timeout', '20', '--max-time', '120',
                        '-r', f'{start}-{end}', '-o', str(partial), '-w', '%{http_code}', self.url]).decode()
                    if status.strip() != '206' or partial.stat().st_size != end-start+1:
                        raise RuntimeError('Servidor não respeitou HTTP Range; use --package full')
                    partial.replace(path)
                    print(f'Bloco remoto {block} ({end-start+1} bytes)', flush=True)
                self.memory_block, self.memory_data = block, path.read_bytes()
            offset = self.position-start
            piece = self.memory_data[offset:offset+size]
            chunks.append(piece)
            self.position += len(piece)
            size -= len(piece)
        return b''.join(chunks)


def download(url, destination):
    """Arquivo parcial retomável via curl; fallback urllib quando indisponível."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if zipfile.is_zipfile(destination):
            return destination
        raise ValueError(f'Arquivo existente não é ZIP: {destination}')
    partial = destination.with_suffix('.zip.part')
    curl = shutil.which('curl.exe') or shutil.which('curl')
    if curl:
        subprocess.run([curl, '-4', '-f', '-L', '--retry', '2',
                        '--connect-timeout', '20', '-C', '-', '-o', str(partial), url], check=True)
    else:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as stream:
            shutil.copyfileobj(response, stream)
    if not zipfile.is_zipfile(partial):
        raise ValueError(f'Download inválido: {partial}')
    partial.replace(destination)
    return destination


def extract_training(archive, root, sequences=None):
    """Extrai por nome permitido, sem traversal, links simbólicos ou test sem GT."""
    root = Path(root)
    selected = 0
    with zipfile.ZipFile(archive) as z:
        for info in sorted(z.infolist(), key=lambda i: i.header_offset):
            parts = PurePosixPath(info.filename).parts
            if 'train' not in parts or '..' in parts or '\\' in info.filename:
                continue
            parts = parts[parts.index('train')+1:]
            if len(parts) < 2 or not parts[0].endswith('-FRCNN'):
                continue
            if sequences and parts[0].rsplit('-', 1)[0] not in sequences:
                continue
            relative = parts[1:]
            if not (relative[0] in ('gt', 'det', 'img1') or relative == ('seqinfo.ini',)):
                continue
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                continue
            target = root / 'train' / Path(*parts)
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError('Caminho fora do diretório dos dados')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                selected += 1
                continue
            partial = target.with_suffix(target.suffix+'.part')
            with z.open(info) as source, partial.open('wb') as stream:
                shutil.copyfileobj(source, stream)
            partial.replace(target)
            selected += 1
    if not selected:
        raise ValueError('ZIP não contém os arquivos de treino FRCNN solicitados')
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', choices=['labels', 'full', 'frames'], default='labels')
    p.add_argument('--root', default='data/MOT17')
    p.add_argument('--sequences', nargs='+', help='Ex.: MOT17-02 MOT17-09 MOT17-13')
    args = p.parse_args()
    if args.package == 'frames':
        remote = RemoteZip(URL+'MOT17.zip', Path(args.root).parent/'archives/ranges')
        count = extract_training(remote, args.root, args.sequences)
        print(f'{count} arquivos extraídos seletivamente do ZIP oficial', flush=True)
        return
    filename = 'MOT17Labels.zip' if args.package == 'labels' else 'MOT17.zip'
    archive = download(URL+filename, Path(args.root).parent / 'archives' / filename)
    count = extract_training(archive, args.root, args.sequences)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    print(f'{count} arquivos extraídos; SHA256 do ZIP: {digest}', flush=True)


if __name__ == '__main__':
    main()
