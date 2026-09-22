import os
import re
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Force UTF-8 encoding for standard output (Windows support for emojis)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
KNOWLEDGE_FILE = "masvia-rag-knowledge-base.md"


def parse_qa_section(content: str, category_name: str) -> list[Document]:
    """Parses individual Q&A blocks inside Section 8 so each Q&A is indexed as an atomic chunk."""
    qa_docs = []
    # Match blocks starting with #### Q<number>: <Question Title>
    qa_pattern = re.compile(
        r"(####\s+(Q\d+:[^\n]+))\n(.*?)(?=(?:####\s+Q\d+:)|(?:\n###\s+\[)|(?:\n##\s+\d+)|$)",
        re.DOTALL,
    )

    for match in qa_pattern.finditer(content):
        q_header = match.group(1).strip()
        q_body = match.group(3).strip()

        # Extract tags if available
        tags_match = re.search(r"-\s+\*\*Tags:\*\*\s*`?(\[[^\]]+\]|[^`\n]+)`?", q_body)
        tags_str = tags_match.group(1) if tags_match else ""

        # Extract answer if available
        ans_match = re.search(r"-\s+\*\*Jawaban WhatsApp:\*\*\s*\n?(.*)", q_body, re.DOTALL)
        answer_str = ans_match.group(1).strip() if ans_match else q_body

        # Clean question title
        question_title = re.sub(r"^####\s+", "", q_header).strip()

        # Build clean search payload that boosts question + tags matching
        payload = f"KATEGORI: {category_name}\n"
        payload += f"PERTANYAAN: {question_title}\n"
        if tags_str:
            payload += f"KATA KUNCI: {tags_str}\n"
        payload += f"JAWABAN WHATSAPP:\n{answer_str}"

        qa_docs.append(
            Document(
                page_content=payload,
                metadata={
                    "source": KNOWLEDGE_FILE,
                    "type": "qa_pair",
                    "category": category_name,
                    "title": question_title,
                    "tags": tags_str,
                },
            )
        )

    return qa_docs


def parse_masvia_knowledge_base(file_path: str) -> list[Document]:
    """Intelligently parses the Masvia Knowledge Base into structured semantic chunks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge file '{file_path}' not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    documents = []

    # Split by major headings (## )
    sections = re.split(r"\n(?=##\s+\d+\.)", full_text)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract section title
        header_match = re.match(r"##\s+(\d+\.\s+[^\n]+)", section)
        section_title = header_match.group(1).strip() if header_match else "Informasi Umum"

        # Check if this is the Q&A section (Section 8)
        if "BASIS DATA PERTANYAAN KONSUMEN" in section_title:
            # Subdivide by Category (### [KATEGORI ...])
            category_blocks = re.split(r"\n(?=###\s+\[KATEGORI)", section)
            for cat_block in category_blocks:
                cat_block = cat_block.strip()
                cat_header_match = re.match(r"###\s+\[(KATEGORI[^\]]+)\]", cat_block)
                category_name = cat_header_match.group(1).strip() if cat_header_match else "Umum"

                qa_docs = parse_qa_section(cat_block, category_name)
                documents.extend(qa_docs)
        else:
            # For narrative sections (Facts, Bioactive ingredients, Preparation, Policies, Templates):
            # Split by sub-headers (### ) or use recursive text splitter if large
            subsections = re.split(r"\n(?=###\s+)", section)
            for sub in subsections:
                sub = sub.strip()
                if not sub:
                    continue

                sub_header_match = re.match(r"###\s+([^\n]+)", sub)
                sub_title = sub_header_match.group(1).strip() if sub_header_match else section_title

                # If subsection is long, split gently
                if len(sub) > 1200:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=900,
                        chunk_overlap=120,
                        separators=["\n\n", "\n", ". ", " "],
                    )
                    chunks = splitter.split_text(sub)
                    for idx, chunk in enumerate(chunks):
                        documents.append(
                            Document(
                                page_content=f"[{section_title} > {sub_title}]\n{chunk}",
                                metadata={
                                    "source": KNOWLEDGE_FILE,
                                    "type": "narrative",
                                    "section": section_title,
                                    "subsection": sub_title,
                                    "part": idx + 1,
                                },
                            )
                        )
                else:
                    documents.append(
                        Document(
                            page_content=f"[{section_title} > {sub_title}]\n{sub}",
                            metadata={
                                "source": KNOWLEDGE_FILE,
                                "type": "narrative",
                                "section": section_title,
                                "subsection": sub_title,
                            },
                        )
                    )

    return documents


def main():
    print("=" * 60)
    print("🌿 MASVIA RAG CHATBOT - INGESTION PIPELINE")
    print("=" * 60)

    print(f"📖 Reading and parsing '{KNOWLEDGE_FILE}'...")
    documents = parse_masvia_knowledge_base(KNOWLEDGE_FILE)
    print(f"✅ Extracted {len(documents)} high-quality semantic chunks.")

    # Breakdown by chunk type
    qa_count = sum(1 for d in documents if d.metadata.get("type") == "qa_pair")
    narrative_count = sum(1 for d in documents if d.metadata.get("type") == "narrative")
    print(f"   - FAQ / Q&A pairs : {qa_count} chunks")
    print(f"   - Narrative facts : {narrative_count} chunks")

    print(f"\n🧠 Loading local embedding model: '{EMBEDDING_MODEL}'...")
    print("   (Running 100% locally on CPU - zero API costs, zero rate limits)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    import chromadb
    print(f"\n💾 Resetting and storing vectors into ChromaDB ('{CHROMA_DB_DIR}')...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        client.delete_collection("langchain")
        print("   Cleared previous 'langchain' collection...")
    except Exception:
        pass

    vector_db = Chroma(
        client=client,
        embedding_function=embeddings,
    )
    vector_db.add_documents(documents)

    print("🎉 Ingestion complete!")
    print(f"📁 Vector database successfully persisted to: '{CHROMA_DB_DIR}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
