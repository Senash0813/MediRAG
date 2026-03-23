from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np

from medirag.domain.models import VerifiedDoc
from medirag.llm.client import LMStudioClient
from medirag.llm.prompts import render_final_prompt
from medirag.retrieval.embedder import Embedder


SECTION_HEADERS = ["Direct answer", "Evidence summary", "Limitations"]
CURRENT_YEAR = datetime.utcnow().year


# ---------------------------
# Section parsing utilities
# ---------------------------

def _split_sections(answer_text: str) -> Dict[str, str]:
	"""Parse answer text into three named sections."""
	text = (answer_text or "").strip()
	pattern = r"(?im)^(Direct answer|Evidence summary|Limitations)\s*:\s*"
	parts = re.split(pattern, text)

	out = {h: "" for h in SECTION_HEADERS}
	if len(parts) < 3:
		out["Direct answer"] = text
		return out

	i = 1
	while i + 1 < len(parts):
		header = parts[i].strip()
		body = parts[i + 1].strip()
		if header in out:
			out[header] = body
		i += 2
	return out


# ---------------------------
# Basic helpers
# ---------------------------

def _humanize_flag(flag: str) -> str:
	"""Convert RISK_FLAG to readable format."""
	return flag.replace("_", " ").lower()


def _as_float(x: Any, default: float = 0.0) -> float:
	"""Safely convert to float."""
	try:
		return float(x)
	except Exception:
		return default


def _get_doc_id(doc: VerifiedDoc) -> str:
	"""Extract document ID from VerifiedDoc."""
	return str(doc.paper_id or doc.qa_id or "UNKNOWN")


def _extract_doc_ids_from_text(text: str) -> Set[str]:
	"""Extract all DOC_ID references from text."""
	if not text:
		return set()
	
	ids: Set[str] = set()
	for m in re.findall(r"\[DOC_ID\s*:\s*([0-9A-Za-z_.-]+)\]", text, flags=re.I):
		ids.add(str(m).strip())
	for m in re.findall(r"\bDOC[_\s]*ID\s*[:#]?\s*([0-9A-Za-z_.-]+)\b", text, flags=re.I):
		ids.add(str(m).strip())
	return ids


# ---------------------------
# Evidence summary bullet parsing
# ---------------------------

def _split_bullets(block: str) -> List[str]:
	"""Extract bullet points from text block."""
	lines = []
	for raw in (block or "").splitlines():
		s = raw.strip()
		if s and (s.startswith("-") or s.startswith("*")):
			lines.append(s)
	return lines


def _extract_doc_ids_from_evidence_line(line: str) -> Set[str]:
	"""Extract DOC_IDs from a single evidence bullet line."""
	ids: Set[str] = set()
	for m in re.findall(r"\bDocument\s*ID\s*:\s*([0-9A-Za-z_.-]+)\b", line, flags=re.I):
		ids.add(str(m))
	for m in re.findall(r"\[DOC_ID\s*:\s*([0-9A-Za-z_.-]+)\]", line, flags=re.I):
		ids.add(str(m))
	for m in re.findall(r"\bDOC[_\s]*ID\s*[:#]?\s*([0-9A-Za-z_.-]+)\b", line, flags=re.I):
		ids.add(str(m))
	return ids


# ---------------------------
# Title replacement utilities
# ---------------------------

def _get_doc_title(doc: VerifiedDoc) -> str:
	"""Get document title."""
	return doc.title.strip() if doc.title else f"Document {doc.paper_id or 'UNKNOWN'}"


def _replace_doc_ids_with_titles(text: str, docs: List[VerifiedDoc]) -> str:
	"""Replace [DOC_ID: xxx] with "Title" in text."""
	if not text:
		return text
	
	id_to_title: Dict[str, str] = {}
	for d in docs:
		if d.paper_id is not None:
			id_to_title[str(d.paper_id)] = _get_doc_title(d)
	
	out = text
	for pid, title in id_to_title.items():
		safe = f'"{title}"'
		out = re.sub(rf"\[DOC_ID\s*:\s*{re.escape(pid)}\]", safe, out, flags=re.I)
		out = re.sub(rf"\bDOC[_\s]*ID\s*[:#]?\s*{re.escape(pid)}\b", safe, out, flags=re.I)
	return out


# ---------------------------
# Quality filtering
# ---------------------------

def _is_high_quality(
	doc: VerifiedDoc,
	min_final_score: float = 0.55,
	allowed_tiers: tuple = ("HIGH", "High", "VERY HIGH", "Very High")
) -> bool:
	"""Check if document meets high quality threshold."""
	tier = doc.quality_tier or ""
	score = _as_float(doc.final_score)
	return tier in allowed_tiers or score >= min_final_score


# ---------------------------
# Risk analysis
# ---------------------------

def _get_doc_year(doc: VerifiedDoc) -> Optional[int]:
	"""Extract publication year from document."""
	try:
		if doc.year:
			return int(doc.year)
	except Exception:
		pass
	return None


def _analyze_risk_profile(display_docs: List[VerifiedDoc]) -> Dict[str, Any]:
	"""Analyze risk flags and age distribution across documents."""
	flags: Set[str] = set()
	ages: List[int] = []
	
	for d in display_docs:
		# Collect risk flags
		flags |= {f.lower().replace("_", " ") for f in d.risk_flags}
		
		# Calculate document age
		y = _get_doc_year(d)
		if y:
			ages.append(CURRENT_YEAR - y)
	
	return {
		"flags": flags,
		"max_age": max(ages) if ages else None,
		"avg_age": int(sum(ages) / len(ages)) if ages else None
	}


# ---------------------------
# Answer cleanup utilities
# ---------------------------

def _remove_defensive_opening(text: str) -> str:
	"""
	Remove overly defensive opening phrases that undermine the answer.
	Move coverage gaps to Limitations section instead.
	"""
	if not text:
		return text
	
	# Patterns that signal defensive openings
	defensive_patterns = [
		r"^The provided (documents?|evidence|context|sources?) (do not|does not) directly (address|support|answer|discuss)",
		r"^The (documents?|evidence|context|sources?) (do not|does not) (directly )?provide specific (information|details|data)",
		r"^Based on the (limited )?evidence provided,?\s*",
		r"^While the evidence (is limited|does not fully address)",
		r"^The available evidence (is limited|does not specifically address)",
	]
	
	# Try to find where actual content starts
	sentences = re.split(r"(?<=[.!?])\s+", text)
	if not sentences:
		return text
	
	first_sentence = sentences[0].strip()
	
	# Check if first sentence is defensive
	for pattern in defensive_patterns:
		if re.search(pattern, first_sentence, flags=re.I):
			# If there's a "However," transition, start from there
			if len(sentences) > 1:
				rest = " ".join(sentences[1:])
				# Remove "However," if it starts the next sentence
				rest = re.sub(r"^However,?\s+", "", rest, flags=re.I)
				rest = re.sub(r"^Nevertheless,?\s+", "", rest, flags=re.I)
				rest = re.sub(r"^But\s+", "", rest, flags=re.I)
				if rest.strip():
					return rest.strip()
			break
	
	return text


# ---------------------------
# Semantic fallback matching
# ---------------------------

def _split_into_sentences(text: str) -> List[str]:
	"""
	Split text into sentences, handling common abbreviations.
	Filters out short/hedged sentences.
	"""
	if not text:
		return []
	
	# Basic sentence splitting (can be improved with nltk if needed)
	sentences = re.split(r'(?<=[.!?])\s+', text)
	
	# Filter out defensive/hedged sentences
	hedge_patterns = [
		r"does not (directly )?(address|support|provide|discuss)",
		r"evidence is (limited|insufficient)",
		r"context does not",
		r"cannot (be )?determined",
	]
	
	substantive = []
	for s in sentences:
		s = s.strip()
		if len(s) < 20:  # Too short to be substantive
			continue
		
		# Check if it's a hedged/refusal sentence
		is_hedged = any(re.search(pattern, s, re.I) for pattern in hedge_patterns)
		if not is_hedged:
			substantive.append(s)
	
	return substantive


def _compute_lexical_overlap(sentence: str, passage: str, min_overlap: int = 2) -> bool:
	"""
	Check if sentence and passage share significant domain terms.
	"""
	# Extract potential domain terms (longer words, excluding common stopwords)
	stopwords = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'these', 'those',
	             'are', 'was', 'were', 'been', 'have', 'has', 'had', 'can', 'could',
	             'will', 'would', 'may', 'might', 'should', 'must', 'also', 'such'}
	
	def extract_terms(text: str) -> Set[str]:
		words = re.findall(r'\b[a-z]{4,}\b', text.lower())  # 4+ char words
		return {w for w in words if w not in stopwords}
	
	sentence_terms = extract_terms(sentence)
	passage_terms = extract_terms(passage)
	
	overlap = sentence_terms & passage_terms
	return len(overlap) >= min_overlap


def _semantic_fallback_match(
	direct_answer: str,
	verified_docs: List[VerifiedDoc],
	embedder: Optional[Embedder],
	sim_threshold: float = 0.60,
	lexical_overlap_min: int = 2,
	max_fallback_docs: int = 2,
) -> List[VerifiedDoc]:
	"""
	Find documents that semantically support claims in direct answer.
	
	For each substantive sentence in direct_answer:
	- Find best matching sentence in each document's passage
	- If similarity >= threshold AND lexical overlap passes, add doc
	
	Args:
		direct_answer: The generated answer text
		verified_docs: Candidate documents to check
		embedder: Embedder for semantic similarity
		sim_threshold: Minimum cosine similarity (0.60 recommended)
		lexical_overlap_min: Minimum shared domain terms
		max_fallback_docs: Maximum docs to add via fallback
	
	Returns:
		List of documents that support claims in the answer
	"""
	if not embedder or not direct_answer or not verified_docs:
		return []
	
	# Step 1: Split direct answer into substantive sentences
	answer_sentences = _split_into_sentences(direct_answer)
	if not answer_sentences:
		return []
	
	# Embed all answer sentences at once
	try:
		answer_embeddings = embedder.embed_queries(answer_sentences)
	except Exception:
		return []  # Fail gracefully if embedding fails
	
	# Step 2: For each doc, find best matching sentence
	supporting_docs: List[tuple[VerifiedDoc, float]] = []  # (doc, max_similarity)
	
	for doc in verified_docs:
		passage = doc.passage_text
		if not passage or len(passage) < 50:
			continue
		
		# Split passage into sentences
		passage_sentences = _split_into_sentences(passage)
		if not passage_sentences:
			continue
		
		try:
			passage_embeddings = embedder.embed_queries(passage_sentences)
		except Exception:
			continue
		
		# Step 3: Compute max similarity between any answer sentence and any passage sentence
		max_sim = 0.0
		best_answer_idx = -1
		best_passage_idx = -1
		
		for i, ans_emb in enumerate(answer_embeddings):
			for j, pass_emb in enumerate(passage_embeddings):
				sim = float(np.dot(ans_emb, pass_emb))  # Already normalized
				if sim > max_sim:
					max_sim = sim
					best_answer_idx = i
					best_passage_idx = j
		
		# Step 4: Check if passes both thresholds
		if max_sim >= sim_threshold and best_answer_idx >= 0 and best_passage_idx >= 0:
			# Also check lexical overlap
			if _compute_lexical_overlap(
				answer_sentences[best_answer_idx],
				passage_sentences[best_passage_idx],
				lexical_overlap_min
			):
				supporting_docs.append((doc, max_sim))
	
	# Sort by similarity and limit to max_fallback_docs
	supporting_docs.sort(key=lambda x: x[1], reverse=True)
	return [doc for doc, _ in supporting_docs[:max_fallback_docs]]


# ---------------------------
# MAIN POST-PROCESSOR
# ---------------------------

def _post_process_answer(
	answer_text: str,
	verified_docs: List[VerifiedDoc],
	*,
	used_docs_k: int = 5,
	min_final_score: float = 0.55,
	enforce_traceability: bool = True,
	embedder: Optional[Embedder] = None,
	semantic_fallback_threshold: float = 0.60,
) -> Dict[str, str]:
	"""
	Post-process LLM answer to ensure traceability, quality, and grounded limitations.
	
	Returns dict with keys: direct_answer, evidence_summary, limitations
	"""
	
	sections = _split_sections(answer_text)
	direct_answer = sections.get("Direct answer", "").strip()
	model_ev = sections.get("Evidence summary", "").strip()
	
	# These are the actual context docs the generator saw
	docs = verified_docs[:used_docs_k]
	id_to_doc = {str(d.paper_id): d for d in docs if d.paper_id is not None}
	
	# -------------------------
	# 1) Determine which docs were USED
	# -------------------------
	used_ids: Set[str] = set()
	used_ids |= _extract_doc_ids_from_text(direct_answer)
	used_ids |= _extract_doc_ids_from_text(model_ev)
	
	used_docs = [id_to_doc[i] for i in used_ids if i in id_to_doc]
	
	# -------------------------
	# 2) Decide display_docs (for evidence and limitations)
	# -------------------------
	if enforce_traceability and used_docs:
		# If model cited docs, align evidence to those
		display_docs = used_docs
		
		# Try semantic fallback if too few docs cited
		if embedder and len(display_docs) < 2:
			fallback_docs = _semantic_fallback_match(
				direct_answer,
				docs,
				embedder,
				sim_threshold=semantic_fallback_threshold,
				max_fallback_docs=2,
			)
			# Add fallback docs that aren't already in display_docs
			existing_ids = {str(d.paper_id) for d in display_docs if d.paper_id}
			for fd in fallback_docs:
				if str(fd.paper_id) not in existing_ids:
					display_docs.append(fd)
	else:
		# Otherwise use all context docs
		display_docs = docs
	
	# -------------------------
	# 3) Evidence summary validation and enhancement
	# -------------------------
	ev_lines = _split_bullets(model_ev)
	allowed_ids = {str(d.paper_id) for d in display_docs if d.paper_id is not None}
	
	# Keep only bullets that cite allowed docs
	kept = []
	for line in ev_lines:
		if _extract_doc_ids_from_evidence_line(line) & allowed_ids:
			kept.append(line)
	
	if kept:
		# Enhance bullets with titles
		id_to_title = {
			str(d.paper_id): _get_doc_title(d)
			for d in display_docs
			if d.paper_id is not None
		}
		
		enhanced_lines = []
		for line in kept:
			ids = _extract_doc_ids_from_evidence_line(line)
			if ids:
				did = list(ids)[0]
				title = id_to_title.get(did)
				# Add title if not already present
				if title and title not in line:
					line = re.sub(
						r"(\[DOC_ID\s*:\s*" + re.escape(did) + r"\])",
						rf"\1 {title}",
						line
					)
			enhanced_lines.append(line)
		
		evidence_block = "\n".join(enhanced_lines)
	else:
		# Generate deterministic bullets with DOC_IDs and titles
		evidence_block = "\n".join(
			f"- [DOC_ID: {d.paper_id}] {_get_doc_title(d)}"
			for d in display_docs
			if d.paper_id is not None
		) or "- Unable to map the provided documents to supporting evidence."
	
	# -------------------------
	# 4) Limitations (grounded in display_docs)
	# -------------------------
	rp = _analyze_risk_profile(display_docs)
	lim = []
	
	# Age-based limitation
	if rp["max_age"] and rp["max_age"] >= 10:
		lim.append(f"Some key sources are {rp['max_age']}+ years old.")
	
	# Risk flag based limitations
	if "weak evidence" in rp["flags"]:
		lim.append("Overall supporting evidence is limited.")
	if "outdated evidence" in rp["flags"]:
		lim.append("Some sources contain outdated evidence.")
	if "low authority" in rp["flags"]:
		lim.append("Some sources have lower authority or citation impact.")
	
	# Answer coverage
	if "does not directly" in direct_answer.lower():
		lim.append("The retrieved context does not directly answer the question.")
	
	limitations_block = (
		"The available evidence has the following limitations:\n- "
		+ "\n- ".join(lim)
		if lim else
		"No major limitations detected in the available sources."
	)
	
	# -------------------------
	# 5) Make Direct answer human-readable
	# -------------------------
	direct_answer = _replace_doc_ids_with_titles(direct_answer, display_docs)
	
	# Remove defensive opening phrases
	direct_answer = _remove_defensive_opening(direct_answer)
	
	return {
		"direct_answer": direct_answer or "No answer generated.",
		"evidence_summary": evidence_block,
		"limitations": limitations_block,
	}


# ---------------------------
# MAIN API FUNCTION
# ---------------------------

def run_answer_llm(
	*,
	query: str,
	verified_docs: List[VerifiedDoc],
	instruction_obj,
	client: LMStudioClient,
	direct_answer_sentences: tuple = (4, 8),
	evidence_bullets: tuple = (2, 6),
	limitations_bullets: tuple = (1, 6),
	enforce_traceability: bool = True,
	embedder: Optional[Embedder] = None,
	semantic_fallback_threshold: float = 0.60,
) -> Dict[str, str]:
	"""
	Call LM Studio with the final prompt and post-process the answer.
	
	Args:
		query: User's question
		verified_docs: List of verified documents to use as context
		instruction_obj: Instructions from instructor LLM
		client: LM Studio client
		direct_answer_sentences: (min, max) sentence range for direct answer
		evidence_bullets: (min, max) bullet range for evidence summary
		limitations_bullets: (min, max) bullet range for limitations
		enforce_traceability: If True, only show docs that were cited in answer
		embedder: Optional embedder for semantic fallback matching
		semantic_fallback_threshold: Similarity threshold for fallback (0.60 recommended)
	
	Returns:
		Dict with keys: direct_answer, evidence_summary, limitations
	"""
	
	final_prompt = render_final_prompt(query, verified_docs, instruction_obj)
	
	# Unpack ranges for system instruction
	_, da_max = direct_answer_sentences
	ev_min, ev_max = evidence_bullets
	lim_min, lim_max = limitations_bullets
	
	system_instruction = (
    "You are a retrieval-grounded medical assistant.\n"
    "Use ONLY the provided CONTEXT in the user's message.\n"
    "Always output EXACTLY three sections named: Direct answer, Evidence summary, Limitations.\n"
    "Each section must start with its label followed by a colon.\n\n"
    
    # 🔥 ADD ANTI-HALLUCINATION RULES
    "CRITICAL RULES:\n"
    "1. Every claim in your Direct answer MUST be explicitly stated in the CONTEXT.\n"
    "2. Do NOT infer, generalize, or use background knowledge.\n"
    "3. If specific details (e.g., mechanisms, percentages, specific effects) are not in CONTEXT, do NOT mention them.\n"
    "4. Prefer saying 'The evidence shows X' rather than making broad claims.\n"
    "5. If context is insufficient, explicitly state what is missing.\n\n"
    
    f"Direct answer: write up to {da_max} sentences (use fewer if appropriate). "
    "Lead with what the evidence DOES show. Be constructive and informative. "
    "Use ONLY facts explicitly stated in CONTEXT passages. "
    "If evidence is incomplete, provide what is available and note gaps in Limitations section. "
    "Do NOT start answers with 'The provided evidence does not...'. "
    "Frame partial answers positively: state what IS known from the evidence.\n"
    f"Evidence summary: write {ev_min}-{ev_max} bullet points. "
    "Each bullet must:\n"
    "  - Cite a doc id like [DOC_ID: 123]\n"
    "  - Quote or closely paraphrase SPECIFIC FINDING from that document\n"
    "  - NOT just repeat the paper title or make generic statements\n"
    
    f"Limitations: write {lim_min}-{lim_max} bullet points. "
    "ONLY mention limitations explicitly present in CONTEXT risk flags. "
    "Also note if retrieved documents are off-topic or from different populations.\n"
	)
	
	# Generate raw answer from LLM
	raw = client.generate_chat(system_instruction, final_prompt)
	
	# Post-process for traceability, quality, and grounded limitations
	result = _post_process_answer(
		raw,
		verified_docs,
		used_docs_k=5,
		enforce_traceability=enforce_traceability,
		embedder=embedder,
		semantic_fallback_threshold=semantic_fallback_threshold,
	)
	
	# Include the rendered prompt in the result
	result["rendered_prompt"] = final_prompt
	
	return result
