import heapq
import re
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize
import time

@dataclass
class AlignmentState:
    """State representation for A* search"""
    doc1_idx: int
    doc2_idx: int
    cost: float
    path: List[Tuple[str, int, int, float]]
    
    def __lt__(self, other):
        return self.cost < other.cost

class PlagiarismDetector:
    def __init__(self):
        self.skip_penalty = 2.0  # Cost for skipping a sentence
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text: tokenize sentences and normalize"""
        try:
            sentences = sent_tokenize(text)
            processed = []
            for sentence in sentences:
                # Normalize: lowercase, remove punctuation, strip whitespace
                cleaned = re.sub(r'[^\w\s]', '', sentence.lower())
                cleaned = ' '.join(cleaned.split())
                if cleaned:
                    processed.append(cleaned)
            return processed
        except:
            # Fallback simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            processed = []
            for sentence in sentences:
                cleaned = re.sub(r'[^\w\s]', '', sentence.lower().strip())
                cleaned = ' '.join(cleaned.split())
                if cleaned:
                    processed.append(cleaned)
            return processed
    
    def levenshtein_distance(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def normalized_edit_distance(self, s1: str, s2: str) -> float:
        """Calculate normalized edit distance (0-1 scale)"""
        if not s1 and not s2:
            return 0.0
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 0.0
        return self.levenshtein_distance(s1, s2) / max_len
    
    def sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences (0-1, higher is more similar)"""
        if not sent1 or not sent2:
            return 0.0
        
        # Use normalized edit distance converted to similarity
        distance = self.normalized_edit_distance(sent1, sent2)
        similarity = 1.0 - distance
        
        return similarity
    
    def heuristic(self, state: AlignmentState, doc1: List[str], doc2: List[str]) -> float:
        """Admissible heuristic for A* search"""
        remaining_doc1 = len(doc1) - state.doc1_idx
        remaining_doc2 = len(doc2) - state.doc2_idx
        
        # Minimum possible cost for remaining alignment
        min_remaining = min(remaining_doc1, remaining_doc2)
        max_remaining = max(remaining_doc1, remaining_doc2)
        
        # Estimate: best case alignment cost + skip penalties
        estimated_cost = min_remaining * 0.1  # Best case similarity cost
        estimated_cost += (max_remaining - min_remaining) * self.skip_penalty
        
        return estimated_cost
    
    def generate_successors(self, state: AlignmentState, doc1: List[str], doc2: List[str]) -> List[Tuple[str, AlignmentState]]:
        """Generate all possible successor states"""
        successors = []
        i, j = state.doc1_idx, state.doc2_idx
        
        # Option 1: Align current sentences
        if i < len(doc1) and j < len(doc2):
            similarity = self.sentence_similarity(doc1[i], doc2[j])
            cost = 1.0 - similarity  # Convert similarity to cost
            new_cost = state.cost + cost
            new_path = state.path + [('align', i, j, similarity)]
            new_state = AlignmentState(i + 1, j + 1, new_cost, new_path)
            successors.append(('align', new_state))
        
        # Option 2: Skip sentence in document 1
        if i < len(doc1):
            new_cost = state.cost + self.skip_penalty
            new_path = state.path + [('skip_doc1', i, -1, 0.0)]
            new_state = AlignmentState(i + 1, j, new_cost, new_path)
            successors.append(('skip_doc1', new_state))
        
        # Option 3: Skip sentence in document 2
        if j < len(doc2):
            new_cost = state.cost + self.skip_penalty
            new_path = state.path + [('skip_doc2', -1, j, 0.0)]
            new_state = AlignmentState(i, j + 1, new_cost, new_path)
            successors.append(('skip_doc2', new_state))
        
        return successors
    
    def a_star_plagiarism_detect(self, doc1: List[str], doc2: List[str]) -> Tuple[List[Tuple], float, int]:
        """A* search for optimal text alignment"""
        # Priority queue: (f_cost, state)
        open_set = []
        start_state = AlignmentState(0, 0, 0.0, [])
        heapq.heappush(open_set, (0, start_state))
        
        closed_set = set()
        nodes_expanded = 0
        
        while open_set:
            current_f, current_state = heapq.heappop(open_set)
            nodes_expanded += 1
            
            # Check if goal state reached
            if (current_state.doc1_idx == len(doc1) and 
                current_state.doc2_idx == len(doc2)):
                return current_state.path, current_state.cost, nodes_expanded
            
            state_key = (current_state.doc1_idx, current_state.doc2_idx)
            if state_key in closed_set:
                continue
                
            closed_set.add(state_key)
            
            # Generate and process successors
            for move_type, next_state in self.generate_successors(current_state, doc1, doc2):
                next_state_key = (next_state.doc1_idx, next_state.doc2_idx)
                if next_state_key not in closed_set:
                    f_cost = next_state.cost + self.heuristic(next_state, doc1, doc2)
                    heapq.heappush(open_set, (f_cost, next_state))
        
        return [], float('inf'), nodes_expanded
    
    def detect_plagiarism(self, text1: str, text2: str, similarity_threshold: float = 0.8) -> Dict:
        """Main function to detect plagiarism between two texts"""
        print("Preprocessing documents...")
        doc1 = self.preprocess_text(text1)
        doc2 = self.preprocess_text(text2)
        
        print(f"Document 1: {len(doc1)} sentences")
        print(f"Document 2: {len(doc2)} sentences")
        print("Running A* search for optimal alignment...")
        
        start_time = time.time()
        alignment_path, total_cost, nodes_expanded = self.a_star_plagiarism_detect(doc1, doc2)
        end_time = time.time()
        
        # Analyze results
        plagiarism_results = self.analyze_alignment(alignment_path, doc1, doc2, similarity_threshold)
        
        plagiarism_results.update({
            'total_cost': total_cost,
            'computation_time': end_time - start_time,
            'nodes_expanded': nodes_expanded,
            'doc1_processed': doc1,
            'doc2_processed': doc2,
            'alignment_path': alignment_path
        })
        
        return plagiarism_results
    
    def analyze_alignment(self, alignment_path: List[Tuple], doc1: List[str], doc2: List[str], 
                         similarity_threshold: float) -> Dict:
        """Analyze alignment results to detect plagiarism"""
        aligned_pairs = []
        skipped_doc1 = 0
        skipped_doc2 = 0
        total_similarity = 0.0
        high_similarity_pairs = []
        
        for move in alignment_path:
            move_type, i, j, similarity = move
            
            if move_type == 'align':
                aligned_pairs.append((i, j, similarity))
                total_similarity += similarity
                
                if similarity >= similarity_threshold:
                    high_similarity_pairs.append({
                        'doc1_sentence': doc1[i] if i != -1 else "SKIPPED",
                        'doc2_sentence': doc2[j] if j != -1 else "SKIPPED",
                        'similarity': similarity,
                        'doc1_index': i,
                        'doc2_index': j
                    })
            
            elif move_type == 'skip_doc1':
                skipped_doc1 += 1
            elif move_type == 'skip_doc2':
                skipped_doc2 += 1
        
        # Calculate metrics
        total_alignments = len(aligned_pairs)
        avg_similarity = total_similarity / total_alignments if total_alignments > 0 else 0
        plagiarism_score = len(high_similarity_pairs) / total_alignments if total_alignments > 0 else 0
        
        return {
            'total_alignments': total_alignments,
            'skipped_doc1': skipped_doc1,
            'skipped_doc2': skipped_doc2,
            'average_similarity': avg_similarity,
            'high_similarity_pairs': high_similarity_pairs,
            'plagiarism_score': plagiarism_score,
            'plagiarism_percentage': plagiarism_score * 100
        }

def test_plagiarism_cases():
    """Test the plagiarism detector with different test cases"""
    detector = PlagiarismDetector()
    
    # Test Case 1: Identical Documents
    print("TEST CASE 1: Identical Documents")
    print("=" * 50)
    text1 = "The quick brown fox jumps over the lazy dog. Programming is fun and challenging. Artificial intelligence is transforming our world."
    text2 = "The quick brown fox jumps over the lazy dog. Programming is fun and challenging. Artificial intelligence is transforming our world."
    
    results1 = detector.detect_plagiarism(text1, text2)
    print_results(results1)
    
    # Test Case 2: Slightly Modified Document
    print("\nTEST CASE 2: Slightly Modified Document")
    print("=" * 50)
    text3 = "The quick brown fox jumps over the lazy dog. Programming is enjoyable and difficult. AI is changing our world significantly."
    
    results2 = detector.detect_plagiarism(text1, text3)
    print_results(results2)
    
    # Test Case 3: Completely Different Documents
    print("\nTEST CASE 3: Completely Different Documents")
    print("=" * 50)
    text4 = "The weather today is sunny and warm. Cooking requires patience and skill. Music brings joy to people's lives."
    
    results3 = detector.detect_plagiarism(text1, text4)
    print_results(results3)
    
    # Test Case 4: Partial Overlap
    print("\nTEST CASE 4: Partial Overlap")
    print("=" * 50)
    text5 = "The quick brown fox jumps over the lazy dog. Mathematics is fundamental to science. Sports promote physical and mental health."
    
    results4 = detector.detect_plagiarism(text1, text5)
    print_results(results4)

def print_results(results: Dict):
    """Print formatted results"""
    print(f"Alignment Results:")
    print(f"  Total alignments: {results['total_alignments']}")
    print(f"  Skipped sentences (Doc1/Doc2): {results['skipped_doc1']}/{results['skipped_doc2']}")
    print(f"  Average similarity: {results['average_similarity']:.3f}")
    print(f"  Plagiarism percentage: {results['plagiarism_percentage']:.1f}%")
    print(f"  Computation time: {results['computation_time']:.3f}s")
    print(f"  Nodes expanded: {results['nodes_expanded']}")
    print(f"  Total cost: {results['total_cost']:.3f}")
    
    high_similarity = results['high_similarity_pairs']
    if high_similarity:
        print(f"\nHigh similarity pairs (≥0.8):")
        for i, pair in enumerate(high_similarity[:3]):  # Show first 3
            print(f"  Pair {i+1}:")
            print(f"    Doc1: {pair['doc1_sentence'][:50]}...")
            print(f"    Doc2: {pair['doc2_sentence'][:50]}...")
            print(f"    Similarity: {pair['similarity']:.3f}")
    
    # Classification
    plagiarism_score = results['plagiarism_score']
    if plagiarism_score >= 0.7:
        print("\n🔴 HIGH PLAGIARISM DETECTED")
    elif plagiarism_score >= 0.3:
        print("\n🟡 MODERATE PLAGIARISM SUSPECTED")
    else:
        print("\n🟢 LOW PLAGIARISM - LIKELY ORIGINAL")

def demo_custom_texts():
    """Demo with custom text input"""
    detector = PlagiarismDetector()
    
    print("CUSTOM TEXT PLAGIARISM DETECTION")
    print("=" * 50)
    
    # Get custom input
    print("Enter first document:")
    text1 = input("Document 1: ")
    
    print("\nEnter second document:")
    text2 = input("Document 2: ")
    
    print("\nProcessing...")
    results = detector.detect_plagiarism(text1, text2)
    print_results(results)

def performance_analysis():
    """Analyze performance on different document sizes"""
    detector = PlagiarismDetector()
    
    print("PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Generate test documents of different sizes
    base_sentence = "This is a sample sentence for testing plagiarism detection algorithms. "
    
    document_sizes = [5, 10, 15, 20]  # Number of sentences
    
    for size in document_sizes:
        doc1 = [base_sentence + f"Version A. Sentence {i}." for i in range(size)]
        doc2 = [base_sentence + f"Version B. Sentence {i}." for i in range(size)]
        
        text1 = " ".join(doc1)
        text2 = " ".join(doc2)
        
        print(f"\nDocument size: {size} sentences")
        start_time = time.time()
        results = detector.detect_plagiarism(text1, text2)
        end_time = time.time()
        
        print(f"  Time: {end_time - start_time:.3f}s")
        print(f"  Nodes expanded: {results['nodes_expanded']}")
        print(f"  Alignment cost: {results['total_cost']:.3f}")

if __name__ == "__main__":
    print("PLAGIARISM DETECTION USING A* SEARCH")
    print("=" * 60)
    
    # Install nltk if not available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except:
        print("Downloading required NLTK data...")
        import nltk
        nltk.download('punkt')
    
    # Run test cases
    test_plagiarism_cases()
    
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    performance_analysis()
    
    print("\n" + "=" * 60)
    print("CUSTOM TEXT DEMO")
    print("=" * 60)
    
    # Uncomment to test with custom input
    # demo_custom_texts()
    
    print("\nPlagiarism detection completed!")
