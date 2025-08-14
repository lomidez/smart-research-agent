import agent

def test_sha256_text():
    h1 = agent.sha256_text("hello")
    h2 = agent.sha256_text("hello")
    h3 = agent.sha256_text("world")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64

def test_clean_text():
    s = "  Hello   \n world \t\t !!  "
    assert agent.clean_text(s) == "Hello world !!"

def test_canonicalize_url_removes_utm():
    url = "HTTP://Example.com/Path?q=1&utm_source=abc&fbclid=zzz#frag"
    canon = agent.canonicalize_url(url)
    assert "utm_" not in canon and "fbclid" not in canon
    assert canon.startswith("http://example.com/Path?q=1")

def test_chunk_text_sentence_boundary():
    txt = "A. B. C. " + ("x"*2000) + ". End."
    chunks = agent.chunk_text(txt, target_tokens=100)
    assert len(chunks) >= 2
    for c in chunks[:-1]:
        assert len(c) >= 300

def test_merge_unique_candidates_dedup():
    web = [
        {"url": "https://site/a?utm_source=x", "title": "A"},
        {"url": "https://site/a", "title": "A-dup"},
        {"url": "https://site/b", "title": "B"},
    ]
    arx = [
        {"url": "https://arxiv.org/abs/1234", "title": "P"},
        {"url": "https://site/b?fbclid=zzz", "title": "B-dup"},
    ]
    kept = agent.merge_unique_candidates(web, arx, max_fetch=10)
    assert len(kept) == 3
    urls = [k["url"] for k in kept]
    assert "https://site/a" in urls or "https://site/a?utm_source=x" in urls
    assert "https://site/b" in urls
    assert "https://arxiv.org/abs/1234" in urls

def test_parse_meta_from_html():
    html = """
    <html>
      <head>
        <title>Sample Title</title>
        <meta name="author" content="Alice, Bob">
        <meta property="article:published_time" content="2025-08-01T12:00:00Z">
      </head>
      <body>Hi</body>
    </html>
    """
    title, authors, date = agent.parse_meta_from_html(html)
    assert title == "Sample Title"
    assert authors == "Alice, Bob"
    assert date == "2025-08-01"
