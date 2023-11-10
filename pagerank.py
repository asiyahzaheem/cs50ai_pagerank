import os
import random
import re
import sys


DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    print('corpus', corpus)
    sample_pagerank(corpus, DAMPING, SAMPLES)
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    dist = {}
    links = corpus[page]

    if len(links) == 0:
        for link in corpus:
            dist[link] = 1/len(corpus)
    else:
        for link in corpus:
            dist[link] = (1 - damping_factor) / len(corpus)
        for link in links:
            dist[link] += damping_factor / len(links)
    
    return dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for ea%ch page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    page_rank = {}
    for link in corpus:
        page_rank[link] = 0
    sample = random.choice(list(corpus.keys()))

    for i in range(n):
        page_rank[sample] += 1
        page_dist = transition_model(corpus, sample, damping_factor)
        sample = random.choices(population=list(page_dist.keys()), weights=page_dist.values(), k=1)[0]

    for page in page_rank:
        page_rank[page] = page_rank[page] / n
        
    
    return page_rank
    


def iterate_pagerank(corpus, damping_factor):
    # PR(p) = ( (1-D) / N )  + ( d *  SUM(PR(i) / NumLinks(i) ) )
    N = len(corpus)
    D = damping_factor

    calculated_pr = {}
    page_rank = {}
    condition = True
    
    for page in corpus:
        page_rank[page] = 1/N
        
    while condition:
        for current_page in page_rank: #4
            sum = 0.0
            for page in corpus: #4
                if current_page in corpus[page]:
                    numLinks = len(corpus[page])
                    sum += page_rank[page] / numLinks
                    
                if not corpus[page]:
                    sum += page_rank[page] / N        
            dSum = D * sum
            random = (1-D)/N
            calculated_pr[current_page] = random + dSum
        condition = False
        
        for current_page in page_rank:
            if abs(page_rank[current_page] - calculated_pr[current_page]) > 0.001:
                condition = True
            page_rank[current_page] = calculated_pr[current_page]
        
                    
    return page_rank


#crawl()
if __name__ == "__main__":
    main()
