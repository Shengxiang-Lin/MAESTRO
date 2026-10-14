"""Lossless adapter from DashScope search responses to the tool hit schema."""
import re


def parse_search_response(obj):
    """Keep complete assistant text and all returned URLs in one evidence item.

    Numeric citations use one-based indexes; [0] is accepted for the first
    source without overwriting the mapping for [1], [2], etc. A missing
    citation stays unattributed rather than assigning an arbitrary source.
    """
    texts, urls, indexed_urls, annotation_urls = [], [], [], []

    def keep_url(url):
        if isinstance(url, str) and url.startswith(('https://', 'http://')):
            if url not in urls:
                urls.append(url)
            return url
        return None

    for block in obj.get('output') or []:
        if not isinstance(block, dict):
            continue
        if block.get('type') == 'web_search_call':
            for source in (block.get('action') or {}).get('sources') or []:
                if isinstance(source, dict):
                    # Keep positions even when a source has no usable URL.
                    indexed_urls.append(keep_url(source.get('url')))
        if block.get('type') != 'message':
            continue
        for part in block.get('content') or []:
            if not isinstance(part, dict) or part.get('type') not in ('text', 'output_text'):
                continue
            text = part.get('text')
            if isinstance(text, str) and text.strip():
                texts.append(text)
            for annotation in part.get('annotations') or []:
                if not isinstance(annotation, dict):
                    continue
                citation = annotation.get('url_citation') or annotation
                if isinstance(citation, dict):
                    url = keep_url(citation.get('url'))
                    if url:
                        annotation_urls.append(url)

    if not texts and isinstance(obj.get('output_text'), str) and obj['output_text'].strip():
        texts.append(obj['output_text'])
    content = '\n\n'.join(texts)
    source = None
    for match in re.finditer(r'\[(\d{1,3})\]', content):
        index = int(match.group(1))
        position = max(0, index - 1)
        if position < len(indexed_urls) and indexed_urls[position]:
            source = indexed_urls[position]
            break
    if not source:
        match = re.search(r'\[[^\]]*\]\((https?://[^)\s]+)\)', content)
        if match:
            source = match.group(1)
        elif annotation_urls:
            source = annotation_urls[0]
    if urls:
        manifest = '\n'.join(f'[{i}] {url}' for i, url in enumerate(indexed_urls, 1) if url)
        extras = [url for url in urls if url not in indexed_urls]
        manifest += ''.join('\n' + url for url in extras)
        content += '\n\nSources returned by web_search (verify attribution against citations above):\n' + manifest.lstrip('\n')
    if not content.strip():
        return [{'content': 'No good Google Search Result was found', 'source': 'None'}]
    return [{'content': content, 'source': source or 'None'}]
