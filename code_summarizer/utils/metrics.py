import numpy as np
from collections import Counter

def bleu_score(references, hypotheses, max_n=4):
    """
    Calculate BLEU score between references and hypotheses.
    """
    # Placeholder for the actual BLEU implementation
    
    def count_ngrams(s, n):
        return Counter(zip(*[s[i:] for i in range(n)]))
    
    if len(hypotheses) == 0:
        return 0
    
    # Calculate n-gram precision for each n
    precisions = []
    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        
        for hyp, ref in zip(hypotheses, references):
            hyp_ngrams = count_ngrams(hyp, n)
            ref_ngrams = count_ngrams(ref, n)
            
            # Count matches
            for ngram, count in hyp_ngrams.items():
                matches += min(count, ref_ngrams[ngram])
            
            total += sum(hyp_ngrams.values())
        
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(matches / total)
    
    # Calculate brevity penalty
    bp = 1
    if sum(len(hyp) for hyp in hypotheses) < sum(len(ref) for ref in references):
        ratio = sum(len(hyp) for hyp in hypotheses) / sum(len(ref) for ref in references)
        bp = np.exp(1 - 1/ratio)
    
    # Calculate final BLEU score
    if any(p == 0 for p in precisions):
        return 0
    
    bleu = bp * np.exp(sum(np.log(p) for p in precisions) / max_n)
    
    return bleu


def perplexity(probabilities):
    """
    Calculate perplexity from predicted probabilities.
    """
    log_probs = np.log2(probabilities)
    entropy = -np.mean(log_probs)
    return 2 ** entropy